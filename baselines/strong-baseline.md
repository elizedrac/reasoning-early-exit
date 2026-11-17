# Strong baseline: hidden-state probe

## Labels
Zhang et al. (2025) used Gemini both to extract intermediate answers from each chunk and to judge
their correctness. We use an open-source LLM (Qwen2.5-7B-Instruct, 4-bit) for extraction only and
`math-verify` for judging (see `scripts/label_chunks.py`).

## Probe
Input: the last-layer hidden state (layer 28, 1,536 dims) at the last token of each chunk that
states an answer, standardized with train statistics. Two probes:

- **linear**: logistic regression, `Linear(1536, 1)`.
- **mlp**: `Linear(1536, h) -> ReLU -> Linear(h, 1)`.

Loss: weighted BCE with positive-class weight `w = alpha * #neg / #pos`. Adam, batch 512, early
stopping on dev loss (patience 10, best weights restored). One probe is trained on GSM8K + MATH
train (20,873 chunks, 63% correct) and tuned on GSM8K + MATH dev (4,254 chunks, 67% correct).

Hyperparameters are grid-searched on dev (`--tune`): learning rate {1e-4, 1e-3, 1e-2}, weight
decay {0, 1e-4, 1e-3}, alpha {0.5, 1, 2}, and for the MLP h {128, 256, 512}. We keep the config
with the lowest dev Brier score, which rewards both ranking and calibration. Selecting by dev
ROC-AUC instead picked alpha = 0.5, with nearly the same AUC but 2.5 to 3 times the test ECE.

| Probe | lr | weight decay | alpha | h | epochs |
|---|---|---|---|---|---|
| linear | 1e-4 | 1e-3 | 2 | | 84 |
| mlp | 1e-4 | 1e-4 | 2 | 256 | 16 |

## Early exit
Stop at the first chunk with probe score >= tau and answer with that chunk's answer
(tokens used = the chunk's last token). If no chunk reaches tau, use full reasoning. tau is
picked on dev: the largest token reduction whose dev accuracy is within 1 point of full
reasoning. This picked tau = 0.85 for both probes.

## How to run

```bash
python baselines/strong-baseline.py --data <cis530 folder> --probe linear --tune --out-dir results/probe_linear
python baselines/strong-baseline.py --data <cis530 folder> --probe mlp --tune --out-dir results/probe_mlp
```

All generated data (traces, labeled chunks, hidden states) is in our shared Google Drive folder `cis530/`; it is too large for the repo. `<cis530 folder>` holds `hidden/` and `labeled/`. With `--tune` each run takes a few minutes on a
laptop CPU; without it, under a minute. Each run writes per-tau
predictions (scored with `eval/score.py`) and `summary.json` (including every grid config's dev
metrics).

Other options: `--alpha`, `--lr`, `--weight-decay`, `--hidden` (set the probe without `--tune`),
`--tau` (fix tau instead of picking it on dev), `--max-drop` (dev accuracy drop allowed when
picking tau, default 0.01), `--train` / `--dev` / `--test` (file names under `hidden/` and
`labeled/`).

Score one predictions file (includes probe AUC / ECE / Brier):

```bash
python eval/score.py --pred results/probe_linear/probe_gsm8k_test_tau0.85.jsonl \
    --gold <cis530 folder>/labeled/gsm8k_test.jsonl \
    --out results/probe_linear/probe_gsm8k_test_tau0.85.metrics.json
```

## Results

Probe quality on test chunks:

| Probe | GSM8K AUC / ECE / Brier | MATH-500 AUC / ECE / Brier | AIME 2024 AUC / ECE / Brier |
|---|---|---|---|
| linear | 0.856 / 0.037 / 0.149 | 0.795 / 0.076 / 0.194 | 0.765 / 0.082 / 0.162 |
| mlp | 0.860 / 0.056 / 0.148 | 0.777 / 0.072 / 0.190 | 0.819 / 0.086 / 0.149 |

Early exit at the dev-picked tau = 0.85 (accuracy, token reduction vs. full reasoning):

| Method | GSM8K | MATH-500 | AIME 2024 |
|---|---|---|---|
| full reasoning | 84.8%, 0% | 85.2%, 0% | 30.0%, 0% |
| probe linear | 84.3%, 47% | 82.6%, 37% | 30.0%, 1% |
| probe mlp | 84.5%, 49% | 81.6%, 41% | 30.0%, 4% |

Accuracy and token reduction for other tau are in `results/probe_*/summary.json`. At similar
savings the probe beats the simple baselines: on MATH-500 the mlp probe at tau = 0.8 gets 80.8%
with 44% fewer tokens, while a fixed 4,096-token budget gets 78.8% with 43% fewer; on GSM8K the
linear probe at tau = 0.7 gets 83.2% with 59% fewer tokens, while a fixed 1,024-token budget gets
79.8% with 59% fewer.

## Comparison with Zhang et al. (2025), R1-Distill-Qwen-1.5B

| | Zhang et al. | Ours (linear / mlp) |
|---|---|---|
| ROC-AUC > 0.7 on all datasets | yes | yes: 0.77 to 0.86 |
| ECE < 0.1 (theirs: 0.04 GSM8K, 0.04 MATH) | yes | yes: GSM8K 0.037 / 0.056, MATH-500 0.076 / 0.072 |
| Tokens saved with no accuracy drop at tau = 0.85 | about 24% | GSM8K: 47% / 49% with a 0.3 to 0.5 point drop; MATH-500: 37% / 41% with a 2.6 to 3.6 point drop |

Looking across tau on test (not a tuned result), the largest savings with no accuracy drop are
38% (linear, tau = 0.9) and 45% (mlp, tau = 0.9) on GSM8K, and 12% (mlp, tau = 0.975) on
MATH-500. Our probe matches the paper on GSM8K and on probe quality, but on MATH-500 it needs a
higher tau than 0.85 to avoid losing accuracy.

Our savings are not directly comparable to the paper's. Our tau = 0.85 matches theirs by
coincidence (it was picked on dev). Our token count also favors early exit: full reasoning counts
every generated token, including the answer text after `</think>`, while early exit counts only up
to the chunk end and leaves out the few tokens needed to write the final answer. Labels
(Qwen + `math-verify` vs. Gemini), chunking, and the 16,384-token limit also differ.
