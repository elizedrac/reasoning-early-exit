# Scoring

## Metric definitions

### Accuracy
TODO: answer correctness via `math-verify`; how unparseable answers are treated.

### Average tokens
TODO: which tokens count (reasoning tokens only? forced-answer tokens?).

### Token reduction vs. full reasoning
TODO

### Probe metrics: ROC-AUC, ECE, Brier score
TODO

## Equations

TODO: accuracy, token reduction, ECE, Brier.

## Citations

TODO: Zhang et al. 2025 ("Reasoning Models Know When They're Right"), math-verify, ECE (Guo et al. 2017), Brier (1950).

## Example

TODO: example command and output.

```bash
python eval/score.py --pred results/... --gold data/trace_inputs/... --full traces/...
```
