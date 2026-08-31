#!/usr/bin/env bash
#SBATCH --job-name=judge-sparse
#SBATCH --partition=cpu
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
# Submit aggregate with --array=0-0; nonzero default-array tasks safely no-op.
mode=${1:?usage: sparse.sbatch fold|aggregate|capability-fold|capability-aggregate MATRIX_ROOT RESULTS_DIR raw|pc l1|ridge [PC_BASIS_ROOT] [PC_COMPONENTS]}
matrix_root=${2:?missing MATRIX_ROOT}
results_dir=${3:?missing RESULTS_DIR}
feature_space=${4:?missing raw|pc}
penalty=${5:?missing l1|ridge}
pc_basis_root=${6:-}
pc_components=${7:-}
[[ $feature_space == raw || $feature_space == pc ]] || { echo "invalid feature space" >&2; exit 2; }
[[ $penalty == l1 || $penalty == ridge ]] || { echo "invalid penalty" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "sparse fits require Slurm" >&2; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo "refusing gateway execution" >&2; exit 2; }

repo=${JV_CODE_DIR:-$SLURM_SUBMIT_DIR/repo}
python=${JV_PYTHON:?set JV_PYTHON to the staged environment Python}
packages=${JV_PACKAGES:?set JV_PACKAGES to the staged Python package directory}
manifest_sha=$(sha256sum "$matrix_root/manifest.json" | cut -d' ' -f1)
export PYTHONPATH="$repo:$repo/src:$packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"

args=(--results-dir "$results_dir" --feature-space "$feature_space" --penalty "$penalty" --manifest-sha256 "$manifest_sha")
if [[ $feature_space == pc ]]; then
  [[ -d $pc_basis_root ]] || { echo "missing global PC basis: $pc_basis_root" >&2; exit 2; }
  [[ $pc_components =~ ^[1-9][0-9]*$ ]] || { echo "PC_COMPONENTS must be explicit and positive" >&2; exit 2; }
  args+=(--pc-basis-root "$pc_basis_root" --pc-components "$pc_components")
elif [[ -n $pc_basis_root || -n $pc_components ]]; then
  echo "PC arguments require feature space pc" >&2
  exit 2
fi
case "$mode" in
  fold)
    : "${SLURM_ARRAY_TASK_ID:?fold mode requires an array task}"
    "$python" -m scripts.judge_verdict_analysis.fit fold \
      --fold-index "$SLURM_ARRAY_TASK_ID" --processed-matrix-root "$matrix_root" "${args[@]}"
    ;;
  aggregate)
    [[ ${SLURM_ARRAY_TASK_ID:-0} == 0 ]] || { echo "skipping nonzero aggregate task"; exit 0; }
    "$python" -m scripts.judge_verdict_analysis.fit aggregate "${args[@]}"
    ;;
  capability-fold)
    : "${SLURM_ARRAY_TASK_ID:?capability-fold mode requires an array task}"
    "$python" -m scripts.judge_verdict_analysis.capability fold \
      --fold-index "$SLURM_ARRAY_TASK_ID" --processed-matrix-root "$matrix_root" "${args[@]}"
    ;;
  capability-aggregate)
    [[ ${SLURM_ARRAY_TASK_ID:-0} == 0 ]] || { echo "skipping nonzero capability aggregate task"; exit 0; }
    "$python" -m scripts.judge_verdict_analysis.capability aggregate "${args[@]}"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
