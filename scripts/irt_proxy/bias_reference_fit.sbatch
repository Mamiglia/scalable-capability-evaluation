#!/bin/bash
#SBATCH --job-name=bias-reference
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/bias-reference-%j.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/bias-reference-%j.out
set -euo pipefail

BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
RUN=${RUN:-$BASE/staging/bias_correction_20260817_v1}
SITE=$RUN/site
SITE_TMP=$RUN/site.tmp
PYTHON=/usr/bin/python3.12
WHEELS=$BASE/staging/mc_cost_preview_20260815/wheelhouse
cd "$RUN"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORM_NAME=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_FLAGS=--xla_force_host_platform_device_count=4

require_wheel() {
  compgen -G "$WHEELS/$1" >/dev/null || {
    echo "missing offline wheel: $1" >&2
    exit 1
  }
}
if [ ! -f "$SITE/.ready" ]; then
  for wheel in \
    'pip-26.0.1-*.whl' 'inspect_ai-0.3.251-*.whl' 'jax-0.11.0-*.whl' \
    'jaxlib-0.11.0-*.whl' 'numpyro-0.21.0-*.whl' 'numpy-2.5.1-*.whl' \
    'pandas-3.0.5-*.whl' 'pyarrow-25.0.0-*.whl' 'scipy-1.18.0-*.whl' \
    'pyyaml-6.0.3-*.whl' 'zipfile_zstd-0.0.4-*.whl'
  do
    require_wheel "$wheel"
  done
  test ! -e "$SITE"
  mkdir -p "$SITE_TMP"
  PYTHONPATH="$WHEELS/pip-26.0.1-py3-none-any.whl" \
    "$PYTHON" -m pip install --upgrade --target "$SITE_TMP" --no-index --find-links "$WHEELS" \
      inspect-ai==0.3.251 jax==0.11.0 jaxlib==0.11.0 numpyro==0.21.0 \
      numpy==2.5.1 pandas==3.0.5 pyarrow==25.0.0 scipy==1.18.0 \
      pyyaml==6.0.3 zipfile-zstd==0.0.4
  PYTHONPATH="$SITE_TMP:$RUN/src:$RUN/scripts/irt_proxy" \
    "$PYTHON" -c "import inspect_ai,jax,numpyro,pandas,pyarrow,scipy,yaml"
  touch "$SITE_TMP/.ready"
  mv "$SITE_TMP" "$SITE"
fi
export PYTHONPATH="$SITE:$RUN/src:$RUN/scripts/irt_proxy"

[ -f analysis/bias_correction/synthetic_recovery.csv ] || \
  "$PYTHON" scripts/irt_proxy/correct_biases.py validate-model --config scripts/irt_proxy/config.yaml
[ -f analysis/bias_correction/input/manifest.json ] || \
  "$PYTHON" scripts/irt_proxy/correct_biases.py prepare --config scripts/irt_proxy/config.yaml
[ -f analysis/bias_correction/reference/posterior.npz ] || \
  "$PYTHON" scripts/irt_proxy/correct_biases.py fit-reference --config scripts/irt_proxy/config.yaml
