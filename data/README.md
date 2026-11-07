# Data

Question-level splits for GSM8K, MATH (test = MATH-500), and AIME 2024, one JSONL file per split.
See `data.md` for sources, split rules, and the record format, and `stats.json` for exact counts.

## Regenerating the splits

From the repo root:

```bash
pip install datasets
python scripts/make_splits.py --out data --seed 0
```

This downloads the datasets from Hugging Face and rewrites `data/{gsm8k,math,aime}/*.jsonl` and `data/stats.json`.
(`data.md` shows the old path `data/make_splits.py`; the script now lives in `scripts/`.)

## Trace inputs

`data/trace_inputs/` holds the subsampled question sets used for trace generation. Rebuild with:

```bash
python scripts/sample_inputs.py
```
