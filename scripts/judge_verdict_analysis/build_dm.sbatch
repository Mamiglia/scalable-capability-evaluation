#!/usr/bin/env bash
#SBATCH --job-name=judge-composites
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source_config=${1:?usage: build_dm.sbatch SOURCE_CONFIG OUTPUT_ROOT SOURCE_REVISION}
output_root=${2:?missing OUTPUT_ROOT}
source_revision=${3:?missing SOURCE_REVISION}
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "matrix build requires Slurm" >&2; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo "refusing gateway execution" >&2; exit 2; }

repo=${JV_CODE_DIR:-$SLURM_SUBMIT_DIR/repo}
python=${JV_PYTHON:?set JV_PYTHON to the staged environment Python}
export PYTHONPATH="$repo:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID
cd "$repo"
"$python" -m scripts.judge_verdict_analysis.build_dm \
  --source-config "$source_config" \
  --output-root "$output_root" \
  --source-revision "$source_revision"
