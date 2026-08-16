#!/bin/bash
#SBATCH --job-name=mc-cost-provisional
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/mc-cost-provisional-%j.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/mc-cost-provisional-%j.out

set -euo pipefail

run=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/mc_cost_preview_20260815
site="$run/site"
python=/usr/bin/python3.12
cd "$run"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$site:$run/src"

if [ ! -f interactive_viewer/data/mc_cost_provisional.json ]; then
  "$python" -m mc_cost.cli export-dashboard \
    --config configs/mc_cost/preview.json --trials 100 --provisional \
    --out interactive_viewer/data/mc_cost_provisional.json
fi
