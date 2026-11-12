#!/usr/bin/env bash
# Run the full pipeline on every file in data/trace_inputs/ (train, then dev, then test):
#   generate_traces.py -> label_chunks.py -> relabel.py -> extract_hidden.py
# Every step is resumable, so rerunning this script after an interruption picks up where it stopped.
#
# Usage (from the repo root):
#   bash scripts/run_all.sh                                   # NVIDIA defaults (vllm, cuda)
#   BACKEND=mlx DEVICE=mps bash scripts/run_all.sh            # Apple silicon (small tests only;
#                                                             # the full run would take ~5-6 days)
#   OUT=/content/drive/MyDrive/cis530 bash scripts/run_all.sh # write outputs to Google Drive
#   HIDDEN=0 bash scripts/run_all.sh                          # skip hidden states (do them on a Mac
#                                                             # later to save GPU time)
# Any variable below can be overridden from the environment.
set -euo pipefail

BACKEND="${BACKEND:-vllm}"        # vllm (NVIDIA) or mlx (Apple silicon): generation and labeling
DEVICE="${DEVICE:-cuda}"          # cuda, mps, or cpu: hidden-state extraction
MAX_TOKENS="${MAX_TOKENS:-16384}" # generation budget per trace
LAYERS="${LAYERS:-last}"          # last or every4
OUT="${OUT:-.}"                   # traces/, labeled/, hidden/ are created under this directory
HIDDEN="${HIDDEN:-1}"             # 0: stop after relabel.py (do hidden states elsewhere)
PYTHON="${PYTHON:-python}"
INPUTS="${INPUTS:-data/trace_inputs}"

# Keep a Mac awake for the whole run.
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
  export CAFFEINATED=1
  exec caffeinate -i bash "$0" "$@"
fi

cd "$(dirname "$0")/.."
mkdir -p "$OUT/traces" "$OUT/labeled" "$OUT/hidden" "$OUT/cache"

for split in train dev test; do
  for input in "$INPUTS"/*_"$split".jsonl; do
    [[ -e "$input" ]] || continue
    name="$(basename "$input" .jsonl)"
    echo "=== $name ($(date '+%F %T'))"
    "$PYTHON" scripts/generate_traces.py --backend "$BACKEND" --max-tokens "$MAX_TOKENS" \
      --input "$input" --out "$OUT/traces/$name.jsonl"
    "$PYTHON" scripts/label_chunks.py --backend "$BACKEND" --cache "$OUT/cache/label_cache.jsonl" \
      --input "$OUT/traces/$name.jsonl" --out "$OUT/labeled/$name.jsonl"
    "$PYTHON" scripts/relabel.py --force "$OUT/labeled/$name.jsonl"  # just labeled by this script
    [[ "$HIDDEN" == 1 ]] || continue
    "$PYTHON" scripts/extract_hidden.py --device "$DEVICE" --layers "$LAYERS" \
      --traces "$OUT/labeled/$name.jsonl" --out-dir "$OUT/hidden/$name"
  done
done
echo "=== all done ($(date '+%F %T'))"
