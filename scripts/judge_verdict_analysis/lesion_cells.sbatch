#!/usr/bin/env bash
#SBATCH --job-name=judge-lesion-cells
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# From a staging directory containing repo/ and logs/:
#   sbatch repo/scripts/judge_verdict_analysis/lesion_cells.sbatch ARTIFACT_INDEX OUTPUT_DIR [BOOTSTRAP_ITERATIONS] [SEED] [write-item-deltas] EXPECTED_LABELLED EXPECTED_UNIQUE
set -euo pipefail

artifact_index=${1:?usage: lesion_cells.sbatch ARTIFACT_INDEX OUTPUT_DIR [BOOTSTRAP_ITERATIONS] [SEED] [write-item-deltas]}
output_dir=${2:?usage: lesion_cells.sbatch ARTIFACT_INDEX OUTPUT_DIR [BOOTSTRAP_ITERATIONS] [SEED] [write-item-deltas]}
bootstrap_iterations=${3:-1000}
seed=${4:-0}
item_deltas=${5:-}
expected_labelled=${6:?supply expected labelled specifications: raw=252, pc=60}
expected_unique=${7:?supply expected unique artifacts: raw=250, pc=60}
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "lesion-cell summaries require a Slurm allocation" >&2; exit 2; }
staging=$SLURM_SUBMIT_DIR
[[ $artifact_index == /* ]] || artifact_index=$staging/$artifact_index
[[ $output_dir == /* ]] || output_dir=$staging/$output_dir
repo=${JV_CODE_DIR:-$staging/repo}
python=${JV_PYTHON:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python}
packages=${JV_PACKAGES:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages}

[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || {
  echo "refusing lesion-cell summary on the SWC gateway" >&2
  exit 2
}
[[ -f $artifact_index ]] || { echo "artifact index not found: $artifact_index" >&2; exit 2; }
[[ -z $item_deltas || $item_deltas == write-item-deltas ]] || {
  echo "fifth argument must be write-item-deltas when supplied" >&2
  exit 2
}

export PYTHONPATH="$repo:$repo/src:$packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"

command=(
  "$python" -m scripts.judge_verdict_analysis.lesion_cells
  --artifact-index "$artifact_index" --output-dir "$output_dir"
  --bootstrap-iterations "$bootstrap_iterations" --seed "$seed"
  --expected-labelled "$expected_labelled" --expected-unique "$expected_unique"
)
[[ $item_deltas == write-item-deltas ]] && command+=(--write-item-deltas)
"${command[@]}"
