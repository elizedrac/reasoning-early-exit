# Simple baselines

Stopping rules that use no probe. They set the floor the probe has to beat: if stopping at the
first answer already saves most tokens at little accuracy cost, the probe's gain is the gap above
that, not the gap above full reasoning.

## Stopping rules

All rules read the labeled traces from `scripts/label_chunks.py`. Tokens are counted as in
`eval/scoring.md`.

| rule | setting name | answer | tokens_used |
|---|---|---|---|
| Full reasoning | `full` | final answer after `</think>`; forced answer if the trace hit `--max-tokens` | `num_tokens` |
| First answer | `first` | answer of the first chunk with a non-null answer | that chunk's `end_token` |
| Token budget N | `budget=N` | full-reasoning answer if `num_tokens <= N`; otherwise the forced answer after N tokens | `min(num_tokens, N)` |

- **Full reasoning** is the upper bound on tokens and the reference for token reduction.
- **First answer** is the most aggressive rule that needs no model: stop the first time the model
  states an answer, whether or not it is right. Chunks follow the labeling pipeline (split at
  reflection markers, no-answer pieces merged into the next answered piece), so the first answered
  chunk ends at the piece where the first answer appears. Traces with no answered chunk fall back
  to full reasoning.
- **Token budget N** cuts every long trace at the same length, independent of content. For traces
  longer than N, the first N generated tokens are kept, `"\n</think>\n\nFinal answer: \boxed{"` is
  appended (without a second `</think>` if one is already in the first N tokens), and the model
  decodes up to 32 tokens greedily. This is the same forcing `generate_traces.py` uses for traces
  that hit `--max-tokens`. Only this rule needs the reasoning model, so it runs on a GPU (`--backend
  vllm`) or Apple silicon (`--backend mlx`).

## How to run

From the repo root, with `labeled/` from the shared Drive folder:

```bash
# full + first answer (CPU, seconds)
python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule full first \
    --out results/simple_gsm8k_test.jsonl

# token budgets (loads DeepSeek-R1-Distill-Qwen-1.5B to force answers)
python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule budget \
    --budget 512 1024 2048 4096 --backend vllm --out results/budget_gsm8k_test.jsonl

# score
python eval/score.py --pred results/simple_gsm8k_test.jsonl \
    --gold data/trace_inputs/gsm8k_test.jsonl --full labeled/gsm8k_test.jsonl
```

Several rules and budgets can go in one output file: each line has a `setting` field, and
`eval/score.py` prints one row per setting.

## Results
TODO: accuracy, average tokens, % token reduction per dataset (GSM8K, MATH-500, AIME 2024).
