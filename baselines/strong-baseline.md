# Strong baseline: hidden-state probe

## Labels
Zhang et al. (2025) used Gemini both to extract intermediate answers from each chunk and to judge
their correctness. We use an open-source LLM (Qwen2.5-7B-Instruct, 4-bit) for extraction only and
`math-verify` for judging (see `scripts/label_chunks.py`).

## Probe
TODO: logistic regression / 2-layer MLP, input layer, weighted BCE (w = alpha * neg/pos), Adam, early stopping on dev loss.

## Early exit
TODO: stop at the first chunk with probe score >= tau; tau sweep.

## How to run
TODO

## Results
TODO: probe ROC-AUC / ECE / Brier; accuracy vs. tokens for each tau.
