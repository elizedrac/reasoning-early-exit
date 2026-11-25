# Extension: token-level probe for early exit

> **Status: not implemented yet.** `extensions/token-probe.py` is a skeleton. The commands and
> output format below describe how it is meant to run; the results section is a placeholder.

## Idea
The strong baseline (`baselines/strong-baseline.py`) can only stop at the end of a chunk that
states an answer. This extension trains the same probe on hidden states saved every 8 generated
tokens, so reasoning can stop mid-chunk, as soon as the probe is confident.

## Data
Hidden states come from `scripts/extract_hidden.py --every 8`:

- one row every 8 generated tokens inside each labeled chunk, plus one at each chunk's end, so
  every labeled chunk has at least one row;
- each row gets the label of the chunk it sits in (1 correct, 0 incorrect);
- positions in unlabeled chunks, and positions after the last stated answer, have no row.

```bash
python scripts/extract_hidden.py --every 8 --device mps \
    --traces <cis530 folder>/labeled/gsm8k_train.jsonl \
    --out-dir <cis530 folder>/hidden_tokens/gsm8k_train
```

Run this for each of `gsm8k_train`, `gsm8k_dev`, `gsm8k_test`, `math_train`, `math_dev`,
`math500_test` and `aime2024_test`. In total that is about 1.65M rows (about 5 GB). The output is
in our shared Google Drive folder `cis530/hidden_tokens/`; it is too large for the repo. The
chunk-end states in `hidden/` are not changed.

## Probe
Same as the strong baseline: linear or MLP probe on standardized last-layer states, weighted BCE,
Adam, early stopping on dev loss, and with `--tune` a grid search picked by dev Brier. The training
code is imported from `baselines/strong-baseline.py`.

## Stopping rule
Stop at the first position with probe score >= tau. With `--consecutive m`, stop at the first
position where the last m scores in a row are all >= tau. If nothing fires, use full reasoning.

The answer at the stop:

- **at a chunk end**: that chunk's extracted answer (no model needed);
- **mid-chunk, with `--force`**: cut the trace at the stop token and force the model to answer,
  using `force_answers` from `baselines/simple-baseline.py` (needs MLX on Apple silicon or vLLM on a
  GPU);
- **mid-chunk, without `--force`**: the latest answer stated by that token (None if none, scored
  as wrong).

tau is picked on dev as in the strong baseline: the largest token reduction whose dev accuracy is
within `--max-drop` (default 0.01) of full reasoning.

## How to run

No model needed (mid-chunk stops use the latest stated answer):

```bash
python extensions/token-probe.py --data <cis530 folder> --probe linear --tune \
    --out-dir results/token_probe_linear
```

With forced answers (Mac, MLX):

```bash
python extensions/token-probe.py --data <cis530 folder> --probe linear --tune \
    --force --backend mlx --out-dir results/token_probe_linear_forced
```

Options: `--probe linear|mlp`, `--tune`, `--alpha`, `--lr`, `--weight-decay`, `--hidden`
(probe settings, as in the strong baseline), `--tau` (taus to sweep), `--consecutive m`
(default 1), `--max-drop`, `--force`, `--backend mlx|vllm`, `--train` / `--dev` / `--test`
(file names under `hidden_tokens/` and `labeled/`).

Each run writes `<out-dir>/token_probe_<name>_tau<tau>.jsonl` (predictions for
`eval/score.py`), `<out-dir>/summary.json` (the chosen tau, and dev and test metrics for every
tau) and, with `--force`, `<out-dir>/forced_cache.json` (forced answers, reused across taus and
reruns).

Score one predictions file (probe AUC / ECE / Brier are over positions):

```bash
python eval/score.py --pred results/token_probe_linear/token_probe_gsm8k_test_tau0.85.jsonl \
    --gold <cis530 folder>/labeled/gsm8k_test.jsonl \
    --out results/token_probe_linear/token_probe_gsm8k_test_tau0.85.metrics.json
```

Example output format (the numbers are placeholders, not results):

```
n                1319
accuracy         x.xxxx
avg_tokens       x.xxxx
token_reduction  x.xxxx
roc_auc          x.xxxx
ece              x.xxxx
brier            x.xxxx
```

## Results

To be filled in once the extension runs. Planned tables:

| Probe | GSM8K AUC / ECE / Brier | MATH-500 AUC / ECE / Brier | AIME 2024 AUC / ECE / Brier |
|---|---|---|---|
| token linear | | | |
| token mlp | | | |

Early exit at the dev-picked tau (accuracy, token reduction vs. full reasoning):

| Method | GSM8K | MATH-500 | AIME 2024 |
|---|---|---|---|
| full reasoning | 84.8%, 0% | 85.2%, 0% | 30.0%, 0% |
| strong baseline (probe linear, chunk ends) | 84.3%, 47% | 82.6%, 37% | 30.0%, 1% |
| token probe linear | | | |
| token probe linear, forced | | | |
| token probe mlp | | | |
| token probe mlp, forced | | | |
