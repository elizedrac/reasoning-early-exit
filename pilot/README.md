# Pilot

Trace generation and labeling on 200 training questions (100 GSM8K, 100 MATH) with a
4,096-token limit, run before the full pipeline existed.

| File | Contents |
|---|---|
| `reasoning-trace-pilot.ipynb` | Colab notebook that ran the pilot |
| `sample_traces.py` | Pilot generation and regex labeling script |
| `gsm8k_sample.jsonl`, `math_sample.jsonl` | Pilot traces with regex-labeled chunks (no token IDs) |
| `recovered_*_sample.jsonl`, `relabeled_chunks.jsonl` | Re-extracted final answers and relabeled chunks (chunks identical to the files above) |
| `handcheck.csv` | 100 sampled labeled chunks (50 GSM8K, 50 MATH, seed 0) with a manual review: `true_answer` (the final answer the chunk states most recently without rejecting it; blank = none) |
| `handcheck_llm_v1.csv` | LLM extractor v1: one call per trace returning a list of answers |
| `handcheck_llm_v2.csv` | v2: one call per chunk, all earlier reasoning as context, 4 examples (mlx) |
| `handcheck_llm_v3.csv` | v3 (tried, reverted): v2 plus "only copy an answer the chunk explicitly states; never compute one" |
| v4 (current; CSV on Colab, not in git) | v2 prompt, but only the last 500 tokens of earlier reasoning as context, for speed (vLLM, AWQ, T4) |

## Hand-check results

Scored against `true_answer` on the same 100 chunks. *Extraction*: the extracted answer equals the
true answer (both none counts as equal). *Label*: correct vs. not, where no answer counts as not
correct. *3-way label*: correct / incorrect / no answer.

| | Regex (pilot) | LLM v1 | LLM v2 | LLM v3 (reverted) | LLM v4 (current) | v4, 2,000-token context |
|---|---|---|---|---|---|---|
| Extraction, all | 62% | 67% | 92% | 91% | 89% | 88% |
| Extraction, GSM8K | 84% | 72% | 90% | 90% | 90% | 86% |
| Extraction, MATH | 40% | 62% | 94% | 92% | 88% | 90% |
| Label, all | 93% | 81% | 96% | 95% | 94% | 93% |
| Label, GSM8K | 98% | 84% | 96% | 96% | 94% | 92% |
| Label, MATH | 88% | 78% | 96% | 94% | 94% | 94% |
| 3-way label, all | 63% | 68% | 93% | 92% | 89% | 89% |

v1 to v3 ran on mlx (Mac); v4 ran on vLLM with the AWQ model on a Colab T4.

v1 often returned lists shifted by one segment. The 8 remaining v2 misses are mostly intermediate
numbers taken as answers or answers stated mid-chunk that it missed. In one case it computed an
answer the chunk didn't state (9240 from "1320 * 7 ounces"). v3 added a rule against computing
answers; it didn't fix that case and cost one MATH chunk, so we kept v2.

v4 cuts the context to the last 500 tokens of earlier reasoning, because full context made
labeling too slow on a T4 (~28 s per GSM8K trace). Its 11 misses are mostly numbers reported for
chunks that state no answer (9240 again), plus one formatting miss (`sqrt(5)`). A 2,000-token
context scored no better, so the drop from v2 comes from the backend or noise (2 points is 2
chunks), not the shorter context.

The review is one person's judgment on 100 chunks; a second reviewer should spot-check it.

Reproduce (Colab T4, ~5 minutes including model load; or `--backend mlx` on a Mac, ~15 minutes):

```bash
python scripts/label_chunks.py --backend vllm --handcheck pilot/handcheck.csv --out pilot/handcheck_llm_v4.csv
```
