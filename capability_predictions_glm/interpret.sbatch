#!/usr/bin/env bash
#SBATCH --job-name=cap-interpret
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
staging=$SLURM_SUBMIT_DIR
venv=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv
packages=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages
spec=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$staging/repo/capability_predictions_glm/specifications.txt")
output="$staging/outputs/capability-interpret-v1/$spec"

[[ -n $spec ]] || { echo "no specification for array index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
[[ -f "$output/manifest.json" ]] && { echo "already done: $spec"; exit 0; }

export PYTHONPATH="$staging/repo:$staging/repo/src:$packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$staging/repo"

"$venv/bin/python" -m capability_predictions_glm.interpret \
  --input-root "$staging/inputs/capability-no-common-support" \
  --full-results "$staging/inputs/capability-full-expanded" \
  --specification "$spec" --output "$output"
echo "complete: $spec"
