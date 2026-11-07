"""
Simple stopping-rule baselines (no probe).

Rules:
    full         full reasoning: use the trace's final answer and all its tokens
    first        first answer: stop at the first chunk with a non-null answer
    budget       fixed token budget N: cut the trace at N tokens and force an answer
                 (append "\n</think>\n\nFinal answer: \\boxed{" and decode greedily, as in
                 generate_traces.py)

Input: traces JSONL from scripts/generate_traces.py. Output: predictions JSONL for eval/score.py.

Usage:
    python baselines/simple-baseline.py --traces traces/gsm8k_test.jsonl --rule first \
        --out results/first_gsm8k_test.jsonl
"""
import argparse


def full_reasoning(trace):
    """TODO: return {"id", "answer": final_answer (or forced_answer if truncated), "tokens_used": num_tokens}."""
    raise NotImplementedError


def first_answer(trace):
    """TODO: first chunk with answer is not None -> answer, tokens_used = end_token.
    If no chunk has an answer, fall back to full reasoning."""
    raise NotImplementedError


def fixed_budget(trace, budget, llm=None):
    """TODO: if num_tokens <= budget use full reasoning; otherwise truncate token_ids to
    `budget`, force an answer with the model (GPU, vLLM), tokens_used = budget."""
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True)
    p.add_argument("--rule", choices=["full", "first", "budget"], required=True)
    p.add_argument("--budget", type=int, default=None, help="token budget N for --rule budget")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    # TODO: apply the rule to each trace and write predictions JSONL.
    raise NotImplementedError


if __name__ == "__main__":
    main()
