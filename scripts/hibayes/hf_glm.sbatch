#!/bin/bash
#SBATCH --job-name=hibayes-hf
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/hibayes-hf-%j.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/hibayes-hf-%j.out
set -euo pipefail

BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
RUN=$BASE/staging/hibayes_hf_20260817_v3
PYTHON=$BASE/code/judge_ft/venv/bin/python
WHEELS=$BASE/staging/mc_cost_preview_20260815/wheelhouse
TASK_TMP=$(mktemp -d "${TMPDIR:-/tmp}/hibayes-hf-${SLURM_JOB_ID}.XXXXXX")
SITE=$TASK_TMP/site
mkdir -p "$SITE" "$TASK_TMP/cache"

for pattern in \
  'jax-0.11.0-*.whl' 'jaxlib-0.11.0-*.whl' 'numpyro-0.21.0-*.whl' \
  'scipy-1.18.0-*.whl' 'ml_dtypes-0.6.0-*.whl' 'opt_einsum-3.4.0-*.whl' \
  'multipledispatch-1.0.0-*.whl'
do
  wheel=$(compgen -G "$WHEELS/$pattern" | head -1)
  test -n "$wheel"
  "$PYTHON" -m zipfile -e "$wheel" "$SITE"
done

export PYTHONPATH="$SITE"
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_force_host_platform_device_count=4
export OMP_NUM_THREADS=2
export XDG_CACHE_HOME="$TASK_TMP/cache"
export PYTHONUNBUFFERED=1
cd "$RUN"
"$PYTHON" -c 'import jax,numpyro,pandas; print(jax.devices())'
[ -f result/DONE ] || "$PYTHON" run_hf_swc.py --data hf_2026_08_17.csv --out result
