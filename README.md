# Reasoning Early Exit (CIS 530)

We train a probe on the hidden states of a reasoning model to decide when it can stop thinking
early, reproducing Zhang et al. 2025, "Reasoning Models Know When They're Right."

- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Data:** GSM8K, MATH (test = MATH-500), AIME 2024

Each reasoning trace is split into pieces at reflection markers ("Wait", "Alternatively", ...).
An LLM extractor (Qwen2.5-7B-Instruct, 4-bit) reports the answer each piece states (kept only if the
piece itself writes that answer), pieces without an answer are merged into the next one that has one, and `math-verify` labels each chunk's answer
correct or incorrect. The hidden state at each chunk's last token is saved. A probe trained on these
(hidden state, label) pairs predicts whether the model's current answer is correct. At test time we
stop at the first chunk where the probe is confident enough.

## Repo layout

```
data/          question-level splits (JSONL), stats.json, data.md, README.md (how to regenerate)
  trace_inputs/  subsampled question sets for trace generation
scripts/       make_splits.py, sample_inputs.py, generate_traces.py, label_chunks.py, relabel.py,
               extract_hidden.py, make_handcheck.py, run_all.sh
eval/          score.py, scoring.md
baselines/     simple-baseline.py/.md (stopping rules), strong-baseline.py/.md (probe)
pilot/         pilot traces, script, Colab notebook, labeling hand-check (see pilot/README.md)
results/       outputs of baselines and scoring
```

## Pipeline

Run from the repo root, in this order:

1. `scripts/make_splits.py`: download datasets and build train/dev/test splits (`data/`)
2. `scripts/sample_inputs.py`: pick the questions to generate traces for (`data/trace_inputs/`)
3. `scripts/generate_traces.py`: generate traces (`--backend vllm` on NVIDIA, `mlx` on Apple silicon)
4. `scripts/label_chunks.py`: split at reflection markers, extract answers with Qwen2.5-7B-Instruct, judge with math-verify
5. `scripts/relabel.py`: re-ground, re-merge and re-judge labeled files in place, without the LLM. Fixes files labeled before answer grounding existed; harmless on newer ones.
6. `scripts/extract_hidden.py`: save chunk-end hidden states (`--device cuda|mps|cpu`)
7. `baselines/simple-baseline.py`, `baselines/strong-baseline.py`: stopping rules and probe
8. `eval/score.py`: accuracy, tokens, token reduction, probe metrics

`scripts/run_all.sh` runs steps 3 to 6 for every file in `data/trace_inputs/` (train, then dev,
then test). All steps are resumable: rerun the same command after a disconnect and finished
questions are skipped. Extractor replies are cached in `cache/`. To save GPU time, run Colab with
`HIDDEN=0` and extract hidden states on a Mac (`--device mps`, ~2 s per trace) from the same Drive
folder via Google Drive for desktop.

### Full run: Colab (NVIDIA GPU)

> **You probably don't need to run this.** All processed data (traces, labeled chunks, hidden
> states) is in the shared Google Drive folder `cis530/`. The full run takes several days of free
> Colab GPU time, so only run it to add more samples. It resumes from what is already there.

```bash
from google.colab import drive; drive.mount('/content/drive')
!git clone https://<TOKEN>@github.com/elizedrac/reasoning-early-exit.git
%cd reasoning-early-exit
!pip install -q -r requirements.txt
!OUT=/content/drive/MyDrive/cis530 bash scripts/run_all.sh
```

Defaults: `BACKEND=vllm`, `DEVICE=cuda`, `MAX_TOKENS=16384`, `LAYERS=last`. On a T4 the extractor
uses the AWQ 4-bit Qwen2.5-7B-Instruct (fp16 7B doesn't fit).

### Small tests: Mac (Apple silicon)

```bash
pip install -r requirements.txt
BACKEND=mlx DEVICE=mps MAX_TOKENS=2048 INPUTS=<dir with a small input file> bash scripts/run_all.sh
```

Measured on an M4 (24 GB): generation ~29 tokens/s, extraction ~10 s per piece, hidden states
~2 s per trace. The full 4,249 questions would take about 5 to 6 days, so use Colab for that.

### Labeling quality

On 100 manually reviewed pilot chunks, the LLM extractor labels 94% of chunks correctly (regex:
93%) and extracts the right answer 89% of the time (regex: 62%, 40% on MATH), on the Colab T4
(vLLM, AWQ). Each piece's prompt gets only the last 500 tokens of earlier reasoning
(`--context-tokens`) to speed up labeling (full context took ~28 s per GSM8K trace on a T4).
More context didn't help accuracy: 2,000 tokens scored 93%. Details in
`pilot/README.md`.

## Large files

Traces (`traces/`, `labeled/`) and hidden states (`hidden/`, `*.pt`) live in our **shared Google Drive**, not
in git. They are listed in `.gitignore`. On Colab, mount Drive and point `--out` / `--out-dir` there.
