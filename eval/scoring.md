# Scoring

`eval/score.py` scores a predictions file (one line per question: `id`, `answer`, `tokens_used`,
`full_tokens`, optionally `probe_scores` and `chunk_labels`) against gold answers.

## Metric definitions

### Accuracy
An answer is correct if `math-verify` judges it equal to the gold answer (both wrapped in
`\boxed{}` and parsed). This is the same judge used to label chunks. A missing or unparseable
answer counts as wrong.

### Average tokens
Mean number of generated tokens used before stopping, counted from the start of the model's
output (prompt tokens excluded). For full reasoning this is every generated token, including
the text after `</think>`.

### Token reduction vs. full reasoning
One minus the total tokens used, divided by the total tokens of full reasoning on the same
questions.

### Probe metrics: ROC-AUC, ECE, Brier score
Computed over all labeled chunks, from the probe's predicted probability that the chunk's answer
is correct and the chunk's 0/1 label. ROC-AUC measures ranking, ECE (10 equal-width bins) and
Brier score measure calibration.

## Equations

With questions $i = 1..n$, prediction $\hat a_i$, gold $a_i$, tokens used $t_i$ and full-reasoning
tokens $T_i$:

$$\text{Accuracy} = \frac{1}{n}\sum_i \mathbb{1}[\hat a_i \equiv a_i]
\qquad
\text{Reduction} = 1 - \frac{\sum_i t_i}{\sum_i T_i}$$

With chunks $j = 1..N$, probe score $p_j$, label $y_j$, and bins $B_1..B_{10}$ of $p$:

$$\text{AUC} = P\big(p_j > p_k \mid y_j = 1,\ y_k = 0\big)$$

the probability that a random correct chunk scores higher than a random incorrect one (ties count
one half).


$$\text{ECE} = \sum_b \frac{|B_b|}{N}\,\Big|\overline{y}_{B_b} - \overline{p}_{B_b}\Big|
\qquad
\text{Brier} = \frac{1}{N}\sum_j (p_j - y_j)^2$$

## Citations

- Zhang et al. 2025, "Reasoning Models Know When They're Right."
- Hugging Face `math-verify`, https://github.com/huggingface/Math-Verify
- Fawcett 2006, "An Introduction to ROC Analysis" (ROC-AUC).
- Naeini et al. 2015, "Obtaining Well Calibrated Probabilities Using Bayesian Binning" (ECE).
- Brier 1950, "Verification of Forecasts Expressed in Terms of Probability."

## Example

Options: `--pred` (predictions JSONL), `--gold` (labeled JSONL with the gold `answer`), `--out`
(optional, writes the metrics as JSON). All generated data (traces, labeled chunks, hidden states) is in our shared Google Drive folder `cis530/`; it is too large for the repo. `labeled/` is in that folder.

Simple baseline (no probe scores, so only answer metrics):

```bash
python eval/score.py --pred results/simple/first_gsm8k_test.jsonl --gold labeled/gsm8k_test.jsonl \
    --out results/simple/first_gsm8k_test.metrics.json
```

```
n                1319
accuracy         0.7180
avg_tokens       538.3912
token_reduction  0.7713
```

Probe early exit (adds probe metrics):

```bash
python eval/score.py --pred results/probe_linear/probe_gsm8k_test_tau0.85.jsonl \
    --gold labeled/gsm8k_test.jsonl --out results/probe_linear/probe_gsm8k_test_tau0.85.metrics.json
```

```
n                1319
accuracy         0.8431
avg_tokens       1241.3328
token_reduction  0.4727
roc_auc          0.8558
ece              0.0375
brier            0.1489
```
