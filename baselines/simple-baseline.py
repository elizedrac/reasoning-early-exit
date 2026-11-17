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

Input: labeled JSONL from scripts/label_chunks.py. Output: predictions JSONL for eval/score.py.

Usage:
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule first \
        --out results/first_gsm8k_test.jsonl
    python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule budget --budget 1024 \
        --out results/budget1024_gsm8k_test.jsonl
"""
import argparse
import json
import os


def full_reasoning(trace):
    answer = trace["forced_answer"] if trace["truncated"] else trace["final_answer"]
    return {"answer": answer, "tokens_used": trace["num_tokens"]}


def first_answer(trace):
    for c in trace["chunks"]:
        if c["answer"] is not None and c["end_token"]:
            return {"answer": c["answer"], "tokens_used": c["end_token"]}
    return full_reasoning(trace)


def fixed_budget(trace, budget):
    if trace["num_tokens"] <= budget:
        return full_reasoning(trace)
    answer = None
    for c in trace["chunks"]:
        if c["answer"] is not None and c["end_token"] and c["end_token"] <= budget:
            answer = c["answer"]  # keep the latest answer stated by token N
    return {"answer": answer, "tokens_used": budget}


def predict(trace, rule, budget=None):
    out = {"full": full_reasoning, "first": first_answer}[rule](trace) if rule != "budget" \
        else fixed_budget(trace, budget)
    return {"id": trace["id"], "dataset": trace["dataset"], "rule": rule if rule != "budget" else f"budget{budget}",
            **out, "full_tokens": trace["num_tokens"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="labeled JSONL from label_chunks.py")
    p.add_argument("--rule", choices=["full", "first", "budget"], required=True)
    p.add_argument("--budget", type=int, default=None, help="token budget N for --rule budget")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert args.rule != "budget" or args.budget, "--rule budget needs --budget N"
    with open(args.traces) as f:
        traces = [json.loads(line) for line in f if line.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for t in traces:
            f.write(json.dumps(predict(t, args.rule, args.budget)) + "\n")
    print(f"wrote {len(traces)} predictions to {args.out}")


if __name__ == "__main__":
    main()
