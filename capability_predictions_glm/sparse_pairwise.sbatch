#!/usr/bin/env bash
#SBATCH --job-name=capability-sparse
#SBATCH --partition=cpu
#SBATCH --array=0-15%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM_JOB_ID is required" >&2
  exit 2
fi

case "$(hostname -s | tr '[:upper:]' '[:lower:]')" in
  gateway*|sgw*)
    echo "refusing capability fit on a gateway host" >&2
    exit 2
    ;;
esac

mode=""
previous=""
for argument in "$@"; do
  [[ $previous == --mode ]] && mode=$argument
  [[ $argument == --mode=* ]] && mode=${argument#--mode=}
  previous=$argument
done
if [[ $mode == aggregate && ${SLURM_ARRAY_TASK_ID:-0} != 0 ]]; then
  echo "skipping nonzero aggregate task"
  exit 0
fi

exec python -m capability_predictions_glm.sparse_pairwise "$@"
