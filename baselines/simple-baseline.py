"""
Simple stopping-rule baselines (no probe).

Rules:
    full         full reasoning: the trace's final answer (the forced answer if it hit the token
                 limit) and all its tokens
    first        first answer: stop at the first chunk with an answer (tokens_used = its end_token);
                 if no chunk has one, fall back to full reasoning
    budget       fixed token budget N: if the trace is longer than N, stop at N and use the answer
                 of the last chunk that ends by token N (none -> wrong); otherwise full reasoning.
                 Uses the answers already extracted for each chunk, no new generation.
                 With --force, over-budget traces are instead cut at N tokens and the model is
                 forced to answer (append "\\n</think>\\n\\nFinal answer: \\\\boxed{" and decode
                 greedily, as generate_traces.py does for truncated traces). Needs token_ids in
                 the traces and a GPU (vLLM) or Apple silicon (MLX).

Input: labeled JSONL from scripts/label_chunks.py. Output: predictions JSONL for eval/score.py.

Usage:
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule first \
        --out results/first_gsm8k_test.jsonl
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule budget --budget 1024 \
        --out results/budget1024_gsm8k_test.jsonl
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule budget --budget 1024 \
        --force --backend vllm --out results/budget1024forced_gsm8k_test.jsonl
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


def full_reasoning(trace):
    # Truncated traces never finished, so use the answer forced at the token limit.
    answer = trace["forced_answer"] if trace["truncated"] else trace["final_answer"]
    return {"answer": answer, "tokens_used": trace["num_tokens"]}


def first_answer(trace):
    # Chunks are in trace order; take the first one that states an answer.
    for c in trace["chunks"]:
        if c["answer"] is not None and c["end_token"]:
            return {"answer": c["answer"], "tokens_used": c["end_token"]}
    return full_reasoning(trace)


def fixed_budget(trace, budget, forced=False, forced_answer=None):
    # The trace fits in the budget, so the budget never stops it.
    if trace["num_tokens"] <= budget:
        return full_reasoning(trace)
    # --force: the answer the model gave when made to answer at token N (from force_answers).
    if forced:
        return {"answer": forced_answer, "tokens_used": budget}
    # Cut at token N: the answer is the latest one stated by then (None if none, scored as wrong).
    answer = None
    for c in trace["chunks"]:
        if c["answer"] is not None and c["end_token"] and c["end_token"] <= budget:
            answer = c["answer"]  # keep the latest answer stated by token N
    return {"answer": answer, "tokens_used": budget}


def load_backend(name, budget, dtype, seed):
    """The generation backend from generate_traces.py, sized for prompt + budget tokens."""
    from generate_traces import MLXBackend, VLLMBackend
    args = SimpleNamespace(max_tokens=budget, dtype=dtype, seed=seed)
    return (VLLMBackend if name == "vllm" else MLXBackend)(args)


def force_answers(backend, traces, budget, batch_size=256):
    """Cut each trace at `budget` tokens, close the thinking, and decode the boxed answer greedily.
    Returns {id: answer or None}."""
    from generate_traces import decode, force_suffix, last_boxed
    answers, t0 = {}, time.time()
    for b in range(0, len(traces), batch_size):
        batch = traces[b:b + batch_size]
        id_lists = []
        for t in batch:
            cut = t["token_ids"][:budget]
            suffix = force_suffix(decode(backend.tok, cut))  # no second </think> if already there
            id_lists.append(t["prompt_token_ids"] + cut + backend.tok.encode(suffix, add_special_tokens=False))
        for t, text in zip(batch, backend.force(id_lists)):
            answers[t["id"]] = last_boxed("\\boxed{" + text)
        print(f"forced {len(answers)}/{len(traces)} answers, {time.time() - t0:.0f} s", flush=True)
    return answers


def predict(trace, rule, budget=None, forced=None):
    # Apply the rule, then add the fields score.py needs (full_tokens for token reduction).
    # forced: {id: answer} from force_answers when --force is set, else None.
    if rule == "budget":
        out = fixed_budget(trace, budget, forced is not None, (forced or {}).get(trace["id"]))
        name = f"budget{budget}" + ("forced" if forced is not None else "")
    else:
        out = {"full": full_reasoning, "first": first_answer}[rule](trace)
        name = rule
    return {"id": trace["id"], "dataset": trace["dataset"], "rule": name,
            **out, "full_tokens": trace["num_tokens"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="labeled JSONL from label_chunks.py")
    p.add_argument("--rule", choices=["full", "first", "budget"], required=True)
    p.add_argument("--budget", type=int, default=None, help="token budget N for --rule budget")
    p.add_argument("--force", action="store_true",
                   help="--rule budget: force the model to answer at token N instead of using chunk answers")
    p.add_argument("--backend", choices=["vllm", "mlx"], default="vllm", help="for --force")
    p.add_argument("--dtype", default="half", choices=["half", "float16", "bfloat16"],
                   help="for --force, vLLM only: half for T4; bfloat16 on A100/L4")
    p.add_argument("--seed", type=int, default=0, help="for --force")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert args.rule != "budget" or args.budget, "--rule budget needs --budget N"
    assert not args.force or args.rule == "budget", "--force only applies to --rule budget"
    with open(args.traces) as f:
        traces = [json.loads(line) for line in f if line.strip()]
    forced = None
    if args.force:
        over = [t for t in traces if t["num_tokens"] > args.budget]
        missing = [t["id"] for t in over if not t.get("token_ids")]
        assert not missing, f"{len(missing)} over-budget traces have no token_ids, e.g. {missing[:3]}"
        forced = {}
        if over:
            print(f"{len(over)} traces over budget; loading {args.backend} backend", flush=True)
            forced = force_answers(load_backend(args.backend, args.budget, args.dtype, args.seed), over, args.budget)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for t in traces:
            f.write(json.dumps(predict(t, args.rule, args.budget, forced)) + "\n")
    print(f"wrote {len(traces)} predictions to {args.out}")


if __name__ == "__main__":
    main()
