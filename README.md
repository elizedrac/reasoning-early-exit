# Reasoning Early Exit (CIS 530)

We train a probe on the hidden states of a reasoning model to decide when it can stop thinking
early, reproducing Zhang et al. 2025, "Reasoning Models Know When They're Right."

- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Data:** GSM8K, MATH (test = MATH-500), AIME 2024

Each reasoning trace is split into chunks at reflection markers ("Wait", "Alternatively", ...).
Each chunk's intermediate answer is labeled correct or incorrect with `math-verify`, and the hidden
state at the chunk's last token is saved. A probe trained on these (hidden state, label) pairs
predicts whether the model's current answer is correct. At test time we stop at the first chunk
where the probe is confident enough.

## Repo layout

```
data/          question-level splits (JSONL), stats.json, data.md, README.md (how to regenerate)
  trace_inputs/  subsampled question sets for trace generation
scripts/       make_splits.py, sample_inputs.py, generate_traces.py, extract_hidden.py, make_handcheck.py
eval/          score.py, scoring.md
baselines/     simple-baseline.py/.md (stopping rules), strong-baseline.py/.md (probe)
pilot/         pilot traces (200 training questions, 4,096-token limit), pilot script and Colab notebook, hand-check CSV
results/       outputs of baselines and scoring
```

## Pipeline

Run from the repo root, in this order:

1. `scripts/make_splits.py`: download datasets and build train/dev/test splits (`data/`)
2. `scripts/sample_inputs.py`: pick the questions to generate traces for (`data/trace_inputs/`)
3. `scripts/generate_traces.py`: generate, chunk, and label traces (GPU, vLLM)
4. `scripts/extract_hidden.py`: save chunk-end hidden states (GPU, transformers)
5. `baselines/simple-baseline.py`, `baselines/strong-baseline.py`: stopping rules and probe
6. `eval/score.py`: accuracy, tokens, token reduction, probe metrics

```bash
pip install -r requirements.txt
python scripts/sample_inputs.py
python scripts/generate_traces.py --input data/trace_inputs/gsm8k_train.jsonl --out traces/gsm8k_train.jsonl
python scripts/extract_hidden.py --traces traces/gsm8k_train.jsonl --out-dir hidden/gsm8k_train
```

Both GPU scripts are resumable: rerun the same command after a Colab disconnect and finished
questions are skipped.

## Large files

Traces (`traces/`) and hidden states (`hidden/`, `*.pt`) live in our **shared Google Drive**, not
in git. They are listed in `.gitignore`. On Colab, mount Drive and point `--out` / `--out-dir` there.
