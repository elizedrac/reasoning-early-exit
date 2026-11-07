"""
Extract hidden states at the end of each labeled chunk, for training the correctness probe.

For each trace in a label_chunks.py output file, run one forward pass (transformers, no grad) over prompt_token_ids + token_ids with output_hidden_states=True. For every chunk with
a non-null label, take the vector at position len(prompt_token_ids) + end_token - 1, i.e. the
last token of the chunk.

Layer indexing: hidden_states[0] is the embedding output and hidden_states[i] is the output of
block i, so hidden_states[-1] (= hidden_states[28] for this model) is the last block. Note that
Hugging Face applies the model's final RMSNorm to hidden_states[-1] (it equals last_hidden_state).
    --layers last    -> [28]
    --layers every4  -> [4, 8, 12, 16, 20, 24, 28]

Output: shards <out-dir>/shard_00000.pt, ... each a dict with
    X          float16 tensor [num_chunks, num_layers, 1536]
    layers     list of hidden_states indices
    meta       dict of lists, one entry per row of X:
               id, dataset, split, k, label (1 correct / 0 incorrect), end_token, num_tokens
    trace_ids  all trace IDs processed in this shard (including traces with no labeled chunks)
Resumable: traces already listed in an existing shard are skipped.

Devices: --device cuda (float16), mps (float16 on Apple silicon), cpu (float32). If a trace's
vectors contain NaN/inf in float16, that trace is rerun in float32.

Usage:
    python scripts/extract_hidden.py --traces labeled/gsm8k_train.jsonl --out-dir hidden/gsm8k_train
    python scripts/extract_hidden.py --device mps --traces ... --out-dir ...
Laptop sanity check (2 traces, first 512 tokens, CPU):
    python scripts/extract_hidden.py --traces traces/x.jsonl --out-dir hidden/smoke --cpu-smoke-test
"""
import argparse
import glob
import json
import os
import time

import torch

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LAYER_SETS = {"last": [-1], "every4": [4, 8, 12, 16, 20, 24, 28]}
META_KEYS = ["id", "dataset", "split", "k", "label", "end_token", "num_tokens"]


def load_model(device, dtype):
    import transformers
    from transformers import AutoModel  # base model: no LM head, so no [seq, vocab] logits
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModel.from_pretrained(MODEL, **{dtype_kw: dtype}).to(device).eval()
    return model


def chunk_vectors(model, ids, positions, layers, n_layers, device):
    """Hidden states at `positions` for each layer index: float32 CPU tensor [num_chunks, num_layers, hidden]."""
    with torch.no_grad():
        out = model(torch.tensor([ids], device=device), output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == n_layers + 1, f"expected {n_layers + 1} hidden states, got {len(hs)}"
    pos_t = torch.tensor(positions, device=device)
    x = torch.stack([hs[i][0, pos_t] for i in layers], dim=1).float().cpu()
    del out, hs
    if device == "mps":
        torch.mps.empty_cache()
    return x


def existing_shards(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "shard_*.pt")))
    done = set()
    for path in paths:
        done.update(torch.load(path, weights_only=False)["trace_ids"])
    return len(paths), done


def save_shard(out_dir, idx, X, meta, trace_ids, layers):
    path = os.path.join(out_dir, f"shard_{idx:05d}.pt")
    dim = X[0].shape[-1] if X else 0
    data = {"X": torch.cat(X) if X else torch.zeros(0, len(layers), dim, dtype=torch.float16),
            "layers": layers, "meta": meta, "trace_ids": trace_ids}
    torch.save(data, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leave a half-written shard
    return path, data["X"].shape


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="JSONL from scripts/generate_traces.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers", default="last", choices=list(LAYER_SETS))
    p.add_argument("--shard-size", type=int, default=50, help="traces per shard")
    p.add_argument("--n", type=int, default=None, help="only use the first N traces")
    p.add_argument("--max-len", type=int, default=None,
                   help="truncate inputs to this many tokens; chunks ending past it are skipped")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    p.add_argument("--cpu-smoke-test", action="store_true",
                   help="CPU, float32 compute, 2 traces, --max-len 512")
    args = p.parse_args()

    if args.cpu_smoke_test:
        args.device, args.n, args.max_len = "cpu", 2, 512
    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.float16  # fp16 matmuls on CPU are slow or unsupported
    if device == "cuda":
        assert torch.cuda.is_available(), "no CUDA GPU found (try --device mps or cpu)"
    if device == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"

    os.makedirs(args.out_dir, exist_ok=True)
    shard_idx, done = existing_shards(args.out_dir)
    print(f"{len(done)} traces already extracted in {shard_idx} shards")

    model = load_model(device, dtype)
    fp32_model = None  # loaded on first NaN/inf
    n_layers = model.config.num_hidden_layers
    layers = [n_layers if i == -1 else i for i in LAYER_SETS[args.layers]]
    assert max(layers) <= n_layers

    X, meta, trace_ids = [], {k: [] for k in META_KEYS}, []
    n_seen = n_skipped_chunks = n_fp32_reruns = 0
    t0 = time.time()
    with open(args.traces) as f:
        for line in f:
            if args.n is not None and n_seen >= args.n:
                break
            n_seen += 1
            rec = json.loads(line)
            if rec["id"] in done:
                continue
            n_prompt = len(rec["prompt_token_ids"])
            ids = rec["prompt_token_ids"] + rec["token_ids"]
            if args.max_len is not None:
                ids = ids[:args.max_len]

            positions, rows = [], []
            for c in rec["chunks"]:
                if c["correct"] is None or c["end_token"] is None:
                    continue
                pos = n_prompt + c["end_token"] - 1
                if pos >= len(ids) or c["end_token"] == 0:
                    n_skipped_chunks += 1
                    continue
                positions.append(pos)
                rows.append(c)

            if positions:
                x = chunk_vectors(model, ids, positions, layers, n_layers, device)
                if not torch.isfinite(x).all() and dtype != torch.float32:
                    if fp32_model is None:
                        print("non-finite values in float16; loading a float32 copy for reruns")
                        fp32_model = load_model(device, torch.float32)
                    x = chunk_vectors(fp32_model, ids, positions, layers, n_layers, device)
                    n_fp32_reruns += 1
                assert torch.isfinite(x).all(), f"non-finite hidden states for {rec['id']}"
                X.append(x.to(torch.float16))
                for c in rows:
                    for key, val in zip(META_KEYS, [rec["id"], rec["dataset"], rec["split"], c["k"],
                                                    int(c["correct"]), c["end_token"], rec["num_tokens"]]):
                        meta[key].append(val)
            trace_ids.append(rec["id"])

            if len(trace_ids) >= args.shard_size:
                path, shape = save_shard(args.out_dir, shard_idx, X, meta, trace_ids, layers)
                print(f"wrote {path} X={tuple(shape)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                shard_idx += 1
                X, meta, trace_ids = [], {k: [] for k in META_KEYS}, []

    if trace_ids:
        path, shape = save_shard(args.out_dir, shard_idx, X, meta, trace_ids, layers)
        print(f"wrote {path} X={tuple(shape)}")
    if n_fp32_reruns:
        print(f"{n_fp32_reruns} traces rerun in float32 after NaN/inf in float16")
    if n_skipped_chunks:
        print(f"skipped {n_skipped_chunks} labeled chunks past --max-len or with an empty span")
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    if args.cpu_smoke_test and trace_ids:
        data = torch.load(path, weights_only=False)
        print("smoke test:", {k: v[:3] for k, v in data["meta"].items()})
        print("X dtype", data["X"].dtype, "shape", tuple(data["X"].shape),
              "finite", bool(torch.isfinite(data["X"]).all()))


if __name__ == "__main__":
    main()
