#!/bin/bash
# LoRA-rank sweep of the verdict mimic on one benchmark and one held-out design, runs concurrently on one GPU.
#   bash rank_sweep_on_pod.sh <model_dir> <fits_dir> <design> <bench> "<ranks>" [extra train args]
# Each rank writes to <fits_dir>/rank-<r>/<design>; a rank whose held_out csv exists is skipped.
set -euo pipefail
MODEL_DIR=$1; FITS=$2; DESIGN=$3; BENCH=$4; RANKS=$5
shift 5
cd /workspace/mimic
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for rank in $RANKS; do
    OUT=$FITS/rank-$rank/$DESIGN
    [ -f "$OUT/held_out_judge_verdict_from_pair.csv" ] && { echo "done already: $OUT"; continue; }
    mkdir -p "$OUT"
    python train_verdict_mimic.py --data-dir data --model-dir "$MODEL_DIR" --design $DESIGN --benches $BENCH \
        --lora-rank $rank --out-dir "$OUT" "$@" > "$OUT/train.log" 2>&1 &
done
wait
echo SWEEP-DONE
