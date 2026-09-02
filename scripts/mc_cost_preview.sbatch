#!/bin/bash
#SBATCH --job-name=mc-cost-preview
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/mc-cost-preview-%j.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/mc-cost-preview-%j.out

set -euo pipefail

run=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/mc_cost_preview_20260815
venv="$run/venv"
cd "$run"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [ ! -f "$venv/.ready" ]; then
  /usr/bin/python3.12 -m venv "$venv"
  "$venv/bin/python" -m pip install --no-index --find-links "$run/wheelhouse" \
    inspect-ai==0.3.251 jax==0.11.0 jaxlib==0.11.0 numpyro==0.21.0 \
    numpy==2.5.1 pandas==3.0.5 pyarrow==25.0.0 scipy==1.18.0 zipfile-zstd==0.0.4
  "$venv/bin/python" -c "import inspect_ai,jax,numpyro,pandas,pyarrow,scipy"
  touch "$venv/.ready"
fi

if [ ! -f analysis/mc_cost/validation/manifest.json ]; then
  "$venv/bin/python" -m mc_cost.cli validate --config configs/mc_cost/preview.json
fi

if [ ! -f interactive_dashboards/cost_estimator/data/mc_cost.json ]; then
  "$venv/bin/python" -m mc_cost.cli export-dashboard \
    --config configs/mc_cost/preview.json --trials 100 \
    --out interactive_dashboards/cost_estimator/data/mc_cost.json
fi
