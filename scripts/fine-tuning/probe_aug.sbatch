#!/bin/bash
#SBATCH --job-name=jft-aug-probe
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Quick transfer probe: the existing single-domain GPQA 0.6B judges (plain /
# rewrite-aug / combined) scored on the cross-domain eval_mm sets. New files only.
#   sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B probe_aug.sbatch
source hpc_common.sh
CD="$IN/cross_domain"
PO="$IN/cross/aug_probe"; mkdir -p "$PO"

declare -A AD=( [plain]="out_qwen06/adapters/final" \
                [rewriteaug]="out_qwen06/adapters/rewriteaug_full" \
                [combined]="out_combined06/adapters/final" )
for name in plain rewriteaug combined; do
  for d in gpqa aime chess; do
    o="$PO/${name}_${d}.jsonl"
    [ -f "$o" ] && { echo "have $name/$d"; continue; }
    echo "=== $name on $d $(date) ==="
    python pod_verdicts.py --pairs "$CD/pairs_${d}_indomain.parquet" --splits eval_mm \
      --base-model "$MODEL" --adapter "$IN/${AD[$name]}" --tp 1 --out "$o"
  done
done
echo "=== AUG_PROBE_DONE $(date) ==="
