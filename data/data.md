# Data

## Sources

We use three public math reasoning benchmarks, all with short final answers that can be checked automatically.

- **GSM8K** (Cobbe et al., 2021): grade-school math word problems with numeric answers. Hugging Face: `openai/gsm8k` (config `main`).
- **MATH** (Hendrycks et al., 2021): competition math problems across 7 subjects and 5 difficulty levels. Train split from `EleutherAI/hendrycks_math`. Test set is **MATH-500** (`HuggingFaceH4/MATH-500`), the 500-problem subset of the MATH test set introduced by Lightman et al. (2023) and standard in reasoning-model evaluation.
- **AIME 2024**: 30 problems from the 2024 American Invitational Mathematics Examination, integer answers. Hugging Face: `HuggingFaceH4/aime_2024`. Used only as a held-out, harder test set.

## Splits

GSM8K and MATH come with standard train/test splits. We keep the official test sets and carve a 10% dev set out of each train split (random, seed 0). AIME is too small to train on, so it is test only.

| Dataset | Train | Dev | Test |
|---|---|---|---|
| GSM8K | 6,726 | 747 | 1,319 |
| MATH | 6,724 | 747 | 500 (MATH-500) |
| AIME 2024 | 0 | 0 | 30 |

Splits are made at the question level, so no question (or any reasoning chunk derived from it) appears in more than one split. The script checks this explicitly. For MATH train, the answer is extracted from the `\boxed{}` in the reference solution. We drop problems whose solution has no non-empty `\boxed{}` and problems with several distinct boxed parts (multi-part answers), since keeping only one part would mislabel correct model answers. We also strip trailing punctuation from answers, normalize difficulty levels to integers, and remove leading zeros from AIME answers. Exact final counts and the number of dropped problems are written to `data/stats.json`.

## Format

Each split is a JSON Lines file at `data/<dataset>/<split>.jsonl`. One question per line:

| Field | Description |
|---|---|
| `id` | Unique question ID |
| `dataset` | `gsm8k`, `math`, `math500`, or `aime2024` |
| `problem` | Question text |
| `answer` | Ground-truth final answer (string) |
| `solution` | Reference worked solution |
| `subject` | MATH subject (null for GSM8K and AIME) |
| `level` | MATH difficulty level as an integer 1 to 5 (null for GSM8K, AIME, and a few MATH problems with an unknown level) |

Example (GSM8K):

```json
{"id": "gsm8k-train-0", "dataset": "gsm8k",
 "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips to her friends in May. How many clips did Natalia sell altogether in April and May?",
 "answer": "72",
 "solution": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.",
 "subject": null, "level": null}
```

## Reproducing

From the repo root:

```bash
pip install datasets
python scripts/make_splits.py --out data --seed 0
```

`--max-train N` subsamples each train split if we need to cap trace-generation compute.

## Trace inputs

`scripts/sample_inputs.py` (seed 0) picks the questions we generate traces for and writes them to `data/trace_inputs/{dataset}_{split}.jsonl`, adding a `split` field to each record:

| Split | Questions |
|---|---|
| Train | 1,000 GSM8K + 1,000 MATH (random sample) |
| Dev | 200 GSM8K + 200 MATH (random sample) |
| Test | all 1,319 GSM8K + 500 MATH-500 + 30 AIME 2024 |

The script checks that no question ID or problem text appears in more than one file.

## Derived data (to be generated in Milestone 2)

The questions above are the inputs. For each question, we will generate a reasoning trace with DeepSeek-R1-Distill-Qwen-1.5B (vLLM) and split its thinking part at reflection markers ("Wait", "Alternatively", "Let me check", etc.). An LLM extractor (Qwen2.5-7B-Instruct, 4-bit) reads each piece with the question and all earlier reasoning as context and reports the final answer the piece explicitly states, or none. Following Zhang et al., pieces without an answer are merged into the next piece that has one (pieces after the last answer join the last answered chunk). Each chunk's answer is labeled correct or incorrect against `answer` with Hugging Face `math-verify`, and we save the hidden state at the chunk's last token. These chunk-level examples (hidden state, label) are the training data for the probe. They inherit the question-level splits above.

Some MATH answers include units or formatting (for example `13.5\text{ pounds}`, `\$3.47`). We keep them as is and rely on `math-verify` to normalize them during labeling, and will check its handling of these cases in Milestone 2.

**Pilot status:** We tested trace generation and labeling on 200 training questions (100 each from GSM8K and MATH). The model's final answer was correct on 86% of GSM8K and 69% of MATH questions, and traces averaged 5.7 chunks (5.6 labeled) after merging chunks until each states a clear answer. 18.5% of traces reached the 4,096-token limit, so the full run uses a 16,384-token limit. The pilot labeled chunks with a regex extractor; a manual check of 100 pilot chunks found it extracted the right answer only 62% of the time (40% on MATH), so the full run uses the LLM extractor instead (see `pilot/README.md`). Hidden states have not been saved yet. The pilot traces, script (`sample_traces.py`), and Colab notebook (`reasoning-trace-pilot.ipynb`) are in `pilot/`.
