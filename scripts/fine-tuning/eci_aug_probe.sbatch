#!/bin/bash
#SBATCH --job-name=jft-eci-aug
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out
# ECI aug-LOO plain-transfer confirmation: score loo_<d>_eci_aug on d's eci eval_mm.
source hpc_common.sh
PO="$OUT/phase2"; mkdir -p "$PO"
for d in gpqa aime chess; do
  o="$PO/augloo_plain_eci_${d}.jsonl"
  [ -f "$o" ] && { echo "have $d"; continue; }
  python pod_verdicts.py --pairs "$IN/cross_domain/pairs_${d}_eci.parquet" --splits eval_mm \
    --base-model "$MODEL" --adapter "$OUT/loo_${d}_eci_aug/adapters/final" --tp 1 --out "$o"
done
echo "=== ECI_AUG_DONE $(date) ==="
