#!/usr/bin/env bash
#SBATCH --job-name=capability-lesions
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

# Usage: sbatch --array=0-N repo/capability_predictions_glm/pairwise_wrapper_lesions.sbatch prepare|spec|finalize [CLI arguments]
#        sbatch repo/capability_predictions_glm/pairwise_wrapper_lesions.sbatch summarize [CLI arguments]
set -euo pipefail

command=${1:?usage: pairwise_wrapper_lesions.sbatch prepare|spec|finalize|summarize [CLI arguments]}
shift
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "capability lesions require Slurm" >&2; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || {
  echo "refusing capability lesions on the SWC gateway" >&2
  exit 2
}
case $command in
  prepare|finalize) index=(--fold-index "${SLURM_ARRAY_TASK_ID:?array task required}") ;;
  spec) index=(--task-index "${SLURM_ARRAY_TASK_ID:?array task required}") ;;
  summarize) index=() ;;
  *) echo "unknown command: $command" >&2; exit 2 ;;
esac

staging=$SLURM_SUBMIT_DIR
repo=${CAPABILITY_CODE_DIR:-$staging/repo}
python=${CAPABILITY_PYTHON:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python}
packages=${CAPABILITY_PACKAGES:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages}
export PYTHONPATH="$repo:$repo/src:$packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"
"$python" -m capability_predictions_glm.pairwise_wrapper_lesions \
  "$command" "${index[@]}" "$@"
