# Scoring

`eval/score.py` scores one predictions file (one or more stopping settings) against gold answers.
Every method (full reasoning, first answer, token budget, probe at each τ) writes the same format,
so they are scored the same way.

## Prediction format

One JSON line per question and setting:

```json
{"id": "gsm8k-test-0", "setting": "tau=0.9", "answer": "18", "tokens_used": 412,
 "probe_scores": [0.31, 0.88, 0.95], "chunk_labels": [0, 1, 1]}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | question ID, matching the gold file |
| `answer` | yes | bare answer string (no `\boxed{}`), or `null` if there is none |
| `tokens_used` | yes | generated tokens up to the stopping point (see *Average tokens*) |
| `setting` | no | label for the stopping rule or threshold; rows are grouped by it. Default `"default"` |
| `probe_scores` | no | probe P(correct) for each chunk of the trace, in order |
| `chunk_labels` | no | the chunk's label: 1 correct, 0 incorrect, `null` if the chunk has no answer |

Several settings can share one file (e.g. the τ sweep). Probe scores do not depend on τ, so the
probe metrics count each question once, however many settings it appears in.

## Metric definitions

### Accuracy
A prediction is correct when `math-verify` judges it equal to the gold answer, using the same
`is_correct` as the labeling pipeline (`scripts/generate_traces.py`): both strings are wrapped in
`\boxed{}`, parsed, and compared with `verify(gold, pred)`. The denominator is every question in
the gold file, so a missing prediction, a `null` answer, or an answer that fails to parse counts as
wrong. The script warns when a setting does not cover every gold question.

### Average tokens
Mean `tokens_used` over the predicted questions. Tokens are generated tokens of the
reasoning model, not counting the prompt. Conventions for each method:

- **Full reasoning:** `num_tokens` of the trace, i.e. everything generated, thinking plus the answer
  after `</think>`. Truncated traces count all `--max-tokens` tokens; the ≤32 forced-answer tokens are
  not counted.
- **Early exit (first answer, probe):** `end_token` of the chunk where we stop, i.e. thinking tokens
  up to and including that chunk. The few tokens needed to write the answer after stopping are not
  counted.
- **Token budget N:** `min(num_tokens, N)`.

Because full reasoning also counts the answer section after `</think>` (usually a short summary),
token reduction slightly favors early exit. This matches how Zhang et al. (2025) report savings
against the full trace.

### Token reduction vs. full reasoning
Pooled over the questions that are in both the predictions and the `--full` traces file, so long
traces count for more than short ones. With `--full`, the script also prints full-reasoning
accuracy and average tokens on the same gold questions as a reference row.

### Probe metrics: ROC-AUC, ECE, Brier score
Chunk-level, pooled over every chunk with a non-null label in the predictions file.

- **ROC-AUC:** `sklearn.metrics.roc_auc_score`. Not reported when only one class is present.
- **ECE:** 10 equal-width bins on the probe score (`--bins`); a score of exactly 1.0 falls in the
  last bin.
- **Brier score:** mean squared error between score and label.

## Equations

Let $Q$ be the gold questions, $\hat a_q$ the predicted answer, $a_q$ the gold answer, $t_q$ the
tokens used, $T_q$ the full trace's tokens, and $P \subseteq Q$ the predicted questions.

$$\text{Accuracy} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}[\texttt{verify}(a_q, \hat a_q)]$$

$$\text{Avg tokens} = \frac{1}{|P|}\sum_{q \in P} t_q \qquad
\text{Token reduction} = 1 - \frac{\sum_{q \in P \cap F} t_q}{\sum_{q \in P \cap F} T_q}$$

where $F$ is the set of questions in the `--full` traces file.

For chunk scores $s_i \in [0,1]$ and labels $y_i \in \{0,1\}$, $i = 1..N$, with bins
$B_b = \{i : \lfloor 10\, s_i \rfloor = b\}$ (score 1.0 in $B_9$):

$$\text{ECE} = \sum_{b=0}^{9} \frac{|B_b|}{N}\,\Big|\,\overline{y}_{B_b} - \overline{s}_{B_b}\Big|
\qquad
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (s_i - y_i)^2$$

## Citations

- Zhang, A., Chen, Y., Pan, J., Zhao, C., Panda, A., Li, J., & He, H. (2025). *Reasoning Models Know
  When They're Right: Probing Hidden States for Self-Verification.* arXiv:2504.05419.
- Hugging Face. *Math-Verify.* https://github.com/huggingface/Math-Verify
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural
  Networks.* ICML. (ECE)
- Brier, G. W. (1950). *Verification of forecasts expressed in terms of probability.* Monthly
  Weather Review, 78(1), 1–3.

## Example

```bash
python eval/score.py --pred results/probe_gsm8k_test.jsonl \
    --gold data/trace_inputs/gsm8k_test.jsonl --full traces/gsm8k_test.jsonl \
    --out results/probe_gsm8k_test_metrics.json
```

```
setting  n_pred  accuracy  avg_tokens  token_reduction
tau=0.5  ...
tau=0.9  ...
(full traces)  ...

probe (N chunks, P% positive): ROC-AUC ...  ECE ...  Brier ...
```

`--out` writes the same numbers as JSON: `{"settings": {setting: metrics}, "full_reference": ..., "probe": ...}`.
