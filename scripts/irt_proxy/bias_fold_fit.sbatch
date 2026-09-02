#!/bin/bash
#SBATCH --job-name=bias-fold
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19%3
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/bias-fold-%A_%a.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/bias-fold-%A_%a.out
set -euo pipefail

BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
RUN=${RUN:-$BASE/staging/bias_correction_20260817_v1}
SITE=$RUN/site
PYTHON=/usr/bin/python3.12
cd "$RUN"
test -f "$SITE/.ready"
test -f analysis/bias_correction/reference/posterior.npz
test -f analysis/bias_correction/reference/items.csv
test -f analysis/bias_correction/reference/anchors.csv
test -f analysis/bias_correction/reference/diagnostics.csv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_FLAGS=--xla_force_host_platform_device_count=4
export PYTHONPATH="$SITE:$RUN/src:$RUN/scripts/irt_proxy"

FORM_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
INNER=$((SLURM_ARRAY_TASK_ID % 10))
FOLD=$((INNER / 2))
if [ "$FORM_INDEX" -eq 0 ]; then IRT_MODEL=fixed_3pl; else IRT_MODEL=2pl; fi
if [ $((INNER % 2)) -eq 0 ]; then MODEL=corrected; else MODEL=uncorrected; fi
OUT="analysis/bias_correction/folds/$IRT_MODEL/fold-$FOLD/$MODEL.npz"
[ -f "$OUT" ] || "$PYTHON" scripts/irt_proxy/correct_biases.py fit-fold \
  --config scripts/irt_proxy/config.yaml --irt-model "$IRT_MODEL" \
  --fold "$FOLD" --model "$MODEL"
