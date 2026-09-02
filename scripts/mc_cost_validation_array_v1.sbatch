#!/bin/bash
#SBATCH --job-name=mc-cost-validation
#SBATCH --partition=cpu
#SBATCH --array=0-11%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/mc-cost-validation-%A_%a.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/mc-cost-validation-%A_%a.out

set -euo pipefail

run=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/mc_cost_preview_20260815
site="$run/site"
python=/usr/bin/python3.12
cd "$run"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$site:$run/src"

"$python" -m mc_cost.cli validate --config configs/mc_cost/preview.json \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 12
