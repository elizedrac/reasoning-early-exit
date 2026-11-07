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
| `handcheck_llm_v2.csv` | v2 (current): one call per chunk, all earlier reasoning as context, 4 examples |
| `handcheck_llm_v3.csv` | v3 (tried, reverted): v2 plus "only copy an answer the chunk explicitly states; never compute one" |

## Hand-check results

Scored against `true_answer` on the same 100 chunks. *Extraction*: the extracted answer equals the
true answer (both none counts as equal). *Label*: correct vs. not, where no answer counts as not
correct. *3-way label*: correct / incorrect / no answer.

| | Regex (pilot) | LLM v1 | LLM v2 (current) | LLM v3 (reverted) |
|---|---|---|---|---|
| Extraction, all | 62% | 67% | 92% | 91% |
| Extraction, GSM8K | 84% | 72% | 90% | 90% |
| Extraction, MATH | 40% | 62% | 94% | 92% |
| Label, all | 93% | 81% | 96% | 95% |
| Label, GSM8K | 98% | 84% | 96% | 96% |
| Label, MATH | 88% | 78% | 96% | 94% |
| 3-way label, all | 63% | 68% | 93% | 92% |

v1 often returned lists shifted by one segment. The 8 remaining v2 misses are mostly intermediate
numbers taken as answers or answers stated mid-chunk that it missed. In one case it computed an
answer the chunk didn't state (9240 from "1320 * 7 ounces"). v3 added a rule against computing
answers; it didn't fix that case and cost one MATH chunk, so we kept v2.

The review is one person's judgment on 100 chunks; a second reviewer should spot-check it.

Reproduce (Apple silicon, ~16 minutes; or `--backend vllm` on a GPU):

```bash
python scripts/label_chunks.py --backend mlx --handcheck pilot/handcheck.csv --out pilot/handcheck_llm_v2.csv
```
