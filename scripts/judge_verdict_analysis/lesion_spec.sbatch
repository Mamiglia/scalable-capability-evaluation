#!/usr/bin/env bash
#SBATCH --job-name=judge-lesion-spec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
matrix_root=${1:?usage: lesion_spec.sbatch MATRIX_ROOT RESULTS_DIR raw|pc l1|ridge}
results_dir=${2:?missing RESULTS_DIR}
feature_space=${3:?missing raw|pc}
penalty=${4:?missing l1|ridge}
: "${SLURM_ARRAY_TASK_ID:?submit through submit_lesion_specs.sh}"
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "lesions require Slurm" >&2; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo "refusing gateway execution" >&2; exit 2; }

repo=${JV_CODE_DIR:-$SLURM_SUBMIT_DIR/repo}
python=${JV_PYTHON:?set JV_PYTHON}
packages=${JV_PACKAGES:?set JV_PACKAGES}
manifest_sha=$(sha256sum "$matrix_root/manifest.json" | cut -d' ' -f1)
export PYTHONPATH="$repo:$repo/src:$packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"
"$python" -m scripts.judge_verdict_analysis.lesioning spec \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --processed-matrix-root "$matrix_root" \
  --results-dir "$results_dir" \
  --feature-space "$feature_space" --penalty "$penalty" \
  --manifest-sha256 "$manifest_sha"
