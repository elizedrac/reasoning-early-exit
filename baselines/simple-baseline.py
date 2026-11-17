"""
Simple stopping-rule baselines (no probe).

Rules:
    full         full reasoning: use the trace's final answer (forced answer if the trace was cut
                 off at --max-tokens) and all its tokens
    first        first answer: stop at the end of the first chunk with a non-null answer and
                 answer with it; tokens_used = that chunk's end_token. Traces with no answered
                 chunk fall back to full reasoning
    budget       fixed token budget N: traces with num_tokens <= N use full reasoning; longer
                 ones are cut at N tokens and the model is forced to answer (append
                 "\\n</think>\\n\\nFinal answer: \\\\boxed{" and decode greedily, exactly as
                 generate_traces.py does for truncated traces); tokens_used = N

Input: labeled JSONL from scripts/label_chunks.py (trace fields + `chunks`). `full` and `budget`
also accept plain traces from scripts/generate_traces.py. Only `budget` needs a GPU (or Apple
silicon): it loads DeepSeek-R1-Distill-Qwen-1.5B to force answers.

Output: predictions JSONL for eval/score.py, one line per (question, setting), with
setting = "full", "first", or "budget=N". Several rules and budgets can go in one file.

Usage:
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule full first \
        --out results/simple_gsm8k_test.jsonl
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule budget \
        --budget 512 1024 2048 --backend vllm --out results/budget_gsm8k_test.jsonl
    python eval/score.py --pred results/simple_gsm8k_test.jsonl \
        --gold data/trace_inputs/gsm8k_test.jsonl --full labeled/gsm8k_test.jsonl
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from generate_traces import decode, force_suffix, last_boxed  # noqa: E402


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def full_answer(trace):
    """Final answer after </think>, or the forced answer if the trace hit --max-tokens."""
    return trace.get("forced_answer") if trace.get("truncated") else trace.get("final_answer")


def full_reasoning(trace):
    return {"id": trace["id"], "setting": "full", "answer": full_answer(trace),
            "tokens_used": trace["num_tokens"]}


def first_answer(trace):
    """Stop at the first chunk with an answer; fall back to full reasoning if there is none."""
    if "chunks" not in trace:
        raise ValueError(f"{trace['id']}: no chunks; --rule first needs labeled traces from label_chunks.py")
    for c in trace["chunks"]:
        if c["answer"] is not None:
            if c["end_token"] is None:
                raise ValueError(f"{trace['id']}: chunk {c['k']} has no token span (trace saved without token_ids)")
            return {"id": trace["id"], "setting": "first", "answer": c["answer"], "tokens_used": c["end_token"]}
    return dict(full_reasoning(trace), setting="first")


def needs_forcing(trace, budget):
    return trace["num_tokens"] > budget


def fixed_budget(trace, budget, forced=None):
    """Full reasoning if the trace fits in `budget`; otherwise the answer forced after `budget`
    tokens (`forced`, from force_answers)."""
    if not needs_forcing(trace, budget):
        return dict(full_reasoning(trace), setting=f"budget={budget}")
    return {"id": trace["id"], "setting": f"budget={budget}", "answer": forced, "tokens_used": budget}


def load_backend(name, max_budget, dtype, seed):
    """The generation backend from generate_traces.py, sized for prompt + max_budget tokens."""
    from generate_traces import MLXBackend, VLLMBackend
    args = SimpleNamespace(max_tokens=max_budget, dtype=dtype, seed=seed)
    return (VLLMBackend if name == "vllm" else MLXBackend)(args)


def force_answers(backend, jobs, batch_size=256):
    """jobs: [(trace, budget)]. Cut each trace at `budget` tokens, close the thinking, and decode
    the boxed answer greedily. Returns one answer (or None) per job."""
    answers, t0 = [], time.time()
    for b in range(0, len(jobs), batch_size):
        batch = jobs[b:b + batch_size]
        id_lists = []
        for trace, budget in batch:
            cut = trace["token_ids"][:budget]
            suffix = force_suffix(decode(backend.tok, cut))  # no second </think> if it is already there
            id_lists.append(trace["prompt_token_ids"] + cut +
                            backend.tok.encode(suffix, add_special_tokens=False))
        answers += [last_boxed("\\boxed{" + text) for text in backend.force(id_lists)]
        print(f"forced {len(answers)}/{len(jobs)} answers, {time.time() - t0:.0f} s", flush=True)
    return answers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="labeled JSONL (label_chunks.py) or traces JSONL")
    p.add_argument("--rule", choices=["full", "first", "budget"], nargs="+", required=True)
    p.add_argument("--budget", type=int, nargs="+", default=None, help="token budgets N for --rule budget")
    p.add_argument("--backend", choices=["vllm", "mlx"], default="vllm", help="for --rule budget")
    p.add_argument("--dtype", default="half", choices=["half", "float16", "bfloat16"],
                   help="vLLM only: half for T4; bfloat16 on A100/L4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if "budget" in args.rule and not args.budget:
        p.error("--rule budget needs --budget N [N ...]")

    traces = load_jsonl(args.traces)
    preds = []
    if "full" in args.rule:
        preds += [full_reasoning(t) for t in traces]
    if "first" in args.rule:
        preds += [first_answer(t) for t in traces]
    if "budget" in args.rule:
        jobs = [(t, n) for n in args.budget for t in traces if needs_forcing(t, n)]
        forced = {}
        if jobs:
            missing = [t["id"] for t, _ in jobs if not t.get("token_ids")]
            if missing:
                raise ValueError(f"{len(missing)} traces need forcing but have no token_ids (e.g. {missing[0]})")
            print(f"{len(jobs)} (trace, budget) pairs over budget; loading {args.backend} backend", flush=True)
            backend = load_backend(args.backend, max(args.budget), args.dtype, args.seed)
            for (t, n), ans in zip(jobs, force_answers(backend, jobs)):
                forced[(t["id"], n)] = ans
        preds += [fixed_budget(t, n, forced.get((t["id"], n))) for n in args.budget for t in traces]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in preds)
    print(f"wrote {len(preds)} predictions ({len(traces)} traces x {len(preds) // max(len(traces), 1)} settings) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
