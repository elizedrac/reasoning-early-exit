# Strong baseline: hidden-state probe

## Probe
TODO: logistic regression / 2-layer MLP, input layer, weighted BCE (w = alpha * neg/pos), Adam, early stopping on dev loss.

## Early exit
TODO: stop at the first chunk with probe score >= tau; tau sweep.

## How to run
TODO

## Results
TODO: probe ROC-AUC / ECE / Brier; accuracy vs. tokens for each tau.
