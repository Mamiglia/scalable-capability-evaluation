#!/bin/bash
#SBATCH --job-name=jft-spoof
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Sandbag-robustness readout: base vs +LoRA verdicts on the same natural /
# spoof_uncertain pairs. One GPU, tiny. Stages skip if output exists.
# Submit: sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=out_qwen06 spoof_verdicts.sbatch
source hpc_common.sh

test -f "$IN/pairs_spoof.parquet" || { echo "MISSING pairs_spoof.parquet"; exit 1; }

echo "=== base $(date) ==="
[ -f "$OUT/spoof_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/spoof_base.jsonl"

echo "=== ft $(date) ==="
[ -f "$OUT/spoof_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --out "$OUT/spoof_ft.jsonl"

echo "=== SPOOF_DONE $(date) ==="
