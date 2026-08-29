#!/usr/bin/env bash
#SBATCH --job-name=judge-expanded
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

# From a new staging directory containing repo/ and logs/:
#   sbatch repo/scripts/judge_verdict_analysis/expanded.sbatch \
#     build BASELINE_CONFIG CANONICAL_CONFIG EXPANDED_MATRIX_ROOT
#   sbatch --array=0-3%4 repo/scripts/judge_verdict_analysis/expanded.sbatch \
#     fit VARIANT EXPANDED_MATRIX_ROOT RESULTS_ROOT
set -euo pipefail

mode=${1:?usage: expanded.sbatch build BASELINE_CONFIG CANONICAL_CONFIG MATRIX_ROOT | fit VARIANT MATRIX_ROOT RESULTS_ROOT}
staging=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
repo=${JV_CODE_DIR:-$staging/repo}
python=${JV_PYTHON:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python}
packages=${JV_PACKAGES:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages}

[[ -n ${SLURM_JOB_ID:-} ]] || { echo "expanded matrices and fits require a Slurm allocation" >&2; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || {
  echo "refusing expanded execution on the SWC gateway" >&2
  exit 2
}

export PYTHONPATH="$repo:$repo/src:$packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"

case "$mode" in
  build)
    [[ -z ${SLURM_ARRAY_TASK_ID:-} ]] || { echo "build must be a non-array job" >&2; exit 2; }
    baseline_config=${2:?build requires BASELINE_CONFIG CANONICAL_CONFIG MATRIX_ROOT}
    source_config=${3:?build requires BASELINE_CONFIG CANONICAL_CONFIG MATRIX_ROOT}
    matrix_root=${4:?build requires BASELINE_CONFIG CANONICAL_CONFIG MATRIX_ROOT}
    "$python" -m scripts.judge_verdict_analysis.expanded build \
      "$baseline_config" "$source_config" "$matrix_root"
    ;;
  fit)
    : "${SLURM_ARRAY_TASK_ID:?fit requires the 0-3 Slurm array}"
    variant=${2:?fit requires VARIANT MATRIX_ROOT RESULTS_ROOT}
    matrix_root=${3:?fit requires VARIANT MATRIX_ROOT RESULTS_ROOT}
    results_root=${4:?fit requires VARIANT MATRIX_ROOT RESULTS_ROOT}
    "$python" -m scripts.judge_verdict_analysis.expanded fit \
      "$variant" "$matrix_root" "$results_root" "$SLURM_ARRAY_TASK_ID"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
