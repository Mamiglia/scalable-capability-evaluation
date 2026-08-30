#!/usr/bin/env bash
#SBATCH --job-name=judge-lesions
#SBATCH --partition=cpu
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

# From a new staging directory containing repo/ and logs/:
#   sbatch repo/scripts/judge_verdict_analysis/lesioning.sbatch prepare MATRIX_ROOT RESULTS_DIR raw l1
#   repo/scripts/judge_verdict_analysis/submit_lesion_specs.sh MATRIX_ROOT RESULTS_DIR raw l1
#   sbatch --dependency=afterok:<spec-job> ... finalize ...
#   sbatch --array=0-0 --dependency=afterok:<finalize-job> \
#     repo/scripts/judge_verdict_analysis/lesioning.sbatch summarize MATRIX_ROOT RESULTS_DIR raw l1
set -euo pipefail

mode=${1:?usage: lesioning.sbatch prepare|finalize|summarize MATRIX_ROOT RESULTS_DIR raw|pc l1|ridge [PC_BASIS_ROOT] [PC_COMPONENTS]}
matrix_root=${2:?missing MATRIX_ROOT}
results_dir=${3:?missing RESULTS_DIR}
feature_space=${4:?missing raw|pc}
penalty=${5:?missing l1|ridge}
pc_basis_root=${6:-}
pc_components=${7:-30}
staging=$SLURM_SUBMIT_DIR
repo=${JV_CODE_DIR:-$staging/repo}
python=${JV_PYTHON:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python}
packages=${JV_PACKAGES:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages}

[[ -n ${SLURM_JOB_ID:-} ]] || { echo "lesions require a Slurm allocation" >&2; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || {
  echo "refusing lesion execution on the SWC gateway" >&2
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
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID
cd "$repo"
manifest_sha=$(sha256sum "$matrix_root/manifest.json" | cut -d' ' -f1)
variant_args=(--feature-space "$feature_space" --penalty "$penalty" --manifest-sha256 "$manifest_sha")
if [[ $feature_space == pc ]]; then
  [[ -d $pc_basis_root ]] || { echo "missing global PC basis: $pc_basis_root" >&2; exit 2; }
  variant_args+=(--pc-basis-root "$pc_basis_root" --pc-components "$pc_components")
fi

case "$mode" in
  prepare)
    : "${SLURM_ARRAY_TASK_ID:?prepare mode requires the 0-15 Slurm array}"
    "$python" -m scripts.judge_verdict_analysis.lesioning prepare \
      --fold-index "$SLURM_ARRAY_TASK_ID" \
      --processed-matrix-root "$matrix_root" \
      --results-dir "$results_dir" "${variant_args[@]}"
    ;;
  finalize)
    : "${SLURM_ARRAY_TASK_ID:?finalize mode requires the 0-15 Slurm array}"
    "$python" -m scripts.judge_verdict_analysis.lesioning finalize \
      --fold-index "$SLURM_ARRAY_TASK_ID" \
      --processed-matrix-root "$matrix_root" \
      --results-dir "$results_dir" "${variant_args[@]}"
    ;;
  summarize)
    [[ ${SLURM_ARRAY_TASK_ID:-0} == 0 ]] || { echo "skipping nonzero summarize task"; exit 0; }
    "$python" -m scripts.judge_verdict_analysis.lesioning summarize \
      --processed-matrix-root "$matrix_root" \
      --results-dir "$results_dir" "${variant_args[@]}"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
