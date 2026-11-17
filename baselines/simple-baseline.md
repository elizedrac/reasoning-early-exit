# Simple baselines

## Stopping rules

All rules reuse the chunk answers already extracted by `scripts/label_chunks.py`; nothing is
regenerated.

- **full**: full reasoning. The trace's final answer (the forced answer if it hit the
  16,384-token limit) and all its tokens.
- **first**: stop at the first chunk that states an answer. If no chunk does, use full reasoning.
- **budgetN**: fixed token budget N. If the trace is longer than N tokens, stop at N and use the
  answer of the last chunk that ends by token N (no answer yet counts as wrong). Shorter traces
  use full reasoning.

## How to run

All generated data (traces, labeled chunks, hidden states) is in our shared Google Drive folder `cis530/`; it is too large for the repo.
`labeled/` is the folder of labeled traces in the shared Drive folder `cis530/` (set
`D` to it as in the README and use `"$D/labeled/..."`). Options: `--traces` (labeled JSONL),
`--rule full|first|budget`, `--budget N` (with `--rule budget`), `--out` (predictions JSONL).

```bash
python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule full \
    --out results/simple/full_gsm8k_test.jsonl
python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule first \
    --out results/simple/first_gsm8k_test.jsonl
python baselines/simple-baseline.py --traces labeled/gsm8k_test.jsonl --rule budget --budget 1024 \
    --out results/simple/budget1024_gsm8k_test.jsonl
```

Score a predictions file (see `eval/scoring.md`):

```bash
python eval/score.py --pred results/simple/first_gsm8k_test.jsonl --gold labeled/gsm8k_test.jsonl \
    --out results/simple/first_gsm8k_test.metrics.json
```

Swap `gsm8k_test` for `math500_test` or `aime2024_test` for the other test sets.

## Results

Test sets. Tokens are generated tokens per question; reduction is relative to full reasoning.

| Rule | GSM8K acc | GSM8K tokens | GSM8K red. | MATH-500 acc | MATH-500 tokens | MATH-500 red. | AIME 2024 acc | AIME 2024 tokens | AIME 2024 red. |
|---|---|---|---|---|---|---|---|---|---|
| full | 84.8% | 2,354 | 0% | 85.2% | 4,885 | 0% | 30.0% | 11,685 | 0% |
| first | 71.8% | 538 | 77% | 73.0% | 1,450 | 70% | 13.3% | 4,553 | 61% |
| budget512 | 55.0% | 512 | 78% | 21.6% | 512 | 90% | 0.0% | 512 | 96% |
| budget1024 | 79.8% | 974 | 59% | 57.2% | 1,019 | 79% | 3.3% | 1,024 | 91% |
| budget2048 | 82.5% | 1,448 | 38% | 73.0% | 1,856 | 62% | 6.7% | 2,038 | 83% |
| budget4096 | 84.0% | 1,840 | 22% | 78.8% | 2,781 | 43% | 16.7% | 3,931 | 66% |

n = 1,319 (GSM8K), 500 (MATH-500), 30 (AIME 2024).
