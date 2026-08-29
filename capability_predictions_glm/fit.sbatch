#!/usr/bin/env bash
#SBATCH --job-name=capability-glm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
inputs=${1:?usage: fit.sbatch INPUT_BUNDLE OUTPUT_DIR}
output=${2:?usage: fit.sbatch INPUT_BUNDLE OUTPUT_DIR}
staging=$SLURM_SUBMIT_DIR
venv=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv
packages=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages

[ -f "$output/manifest.json" ] && { echo "already done: $output"; exit 0; }
[ -f "$inputs/manifest.json" ] || { echo "prepared input bundle missing: $inputs" >&2; exit 2; }

export PYTHONPATH="$staging/repo:$staging/repo/src:$packages"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID

cd "$staging/repo"
"$venv/bin/python" -m capability_predictions_glm --input-root "$inputs" --output-root "$output" --execute --full-only
