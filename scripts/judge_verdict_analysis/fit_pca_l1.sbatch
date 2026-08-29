#!/usr/bin/env bash
#SBATCH --job-name=judge-pca-l1
#SBATCH --partition=cpu
#SBATCH --array=0-15%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

# From a staging directory containing repo/ and logs/:
#   sbatch repo/scripts/judge_verdict_analysis/fit_pca_l1.sbatch fold MATRIX_ROOT RESULTS_DIR
#   sbatch --array=0-0 --dependency=afterok:<fold-job> \
#     repo/scripts/judge_verdict_analysis/fit_pca_l1.sbatch aggregate MATRIX_ROOT RESULTS_DIR
set -euo pipefail

mode=${1:?usage: fit_pca_l1.sbatch fold|aggregate MATRIX_ROOT RESULTS_DIR}
matrix_root=${2:?usage: fit_pca_l1.sbatch fold|aggregate MATRIX_ROOT RESULTS_DIR}
results_dir=${3:?usage: fit_pca_l1.sbatch fold|aggregate MATRIX_ROOT RESULTS_DIR}
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "PCA-L1 fits require a Slurm allocation" >&2; exit 2; }
staging=$SLURM_SUBMIT_DIR
repo=${JV_CODE_DIR:-$staging/repo}
python=${JV_PYTHON:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python}
packages=${JV_PACKAGES:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages}

[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || {
  echo "refusing PCA-L1 execution on the SWC gateway" >&2
  exit 2
}
for required in manifest.json design-matrix.yaml feature-dictionary.parquet pair.parquet; do
  [[ -f $matrix_root/$required ]] || {
    echo "staged design matrix is missing $required: $matrix_root" >&2
    exit 2
  }
done

export PYTHONPATH="$repo:$repo/src:$packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"

case "$mode" in
  fold)
    : "${SLURM_ARRAY_TASK_ID:?fold mode requires the 0-15 Slurm array}"
    "$python" -m scripts.judge_verdict_analysis.fit_pca_l1 fold \
      --fold-index "$SLURM_ARRAY_TASK_ID" \
      --processed-matrix-root "$matrix_root" \
      --results-dir "$results_dir"
    ;;
  aggregate)
    [[ ${SLURM_ARRAY_TASK_ID:-0} == 0 ]] || { echo "aggregate uses --array=0-0" >&2; exit 2; }
    "$python" -m scripts.judge_verdict_analysis.fit_pca_l1 aggregate \
      --results-dir "$results_dir"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
