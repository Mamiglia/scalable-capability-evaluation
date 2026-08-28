#!/bin/bash
# Run the verdict mimic on one pod: the strict folds in FOLDS (default all four) one after another.
#   FOLDS="0 1" DESIGN_PREFIX=model-fold bash run_folds_on_pod.sh <model_dir> <fits_dir> [extra train args]
# DESIGN_PREFIX defaults to new-models-and-items (the strict folds); model-fold holds out models only.
# Skips a fold whose held_out csv already exists, so a restart resumes.
set -euo pipefail
MODEL_DIR=$1
FITS=$2
shift 2
cd /workspace/mimic
for fold in ${FOLDS:-0 1 2 3}; do
    DESIGN=${DESIGN_PREFIX:-new-models-and-items}-$fold
    OUT=$FITS/$DESIGN
    [ -f "$OUT/held_out_judge_verdict_from_pair.csv" ] && { echo "done already: $OUT"; continue; }
    mkdir -p "$OUT"
    python train_verdict_mimic.py --data-dir data --model-dir "$MODEL_DIR" --design $DESIGN --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/train.log"
done
echo ALL-FOLDS-DONE
