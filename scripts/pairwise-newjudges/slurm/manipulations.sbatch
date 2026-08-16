#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

# Judging campaign for one explicit transcript-manipulation condition,
# local vLLM judges only - no API cost. Adapted from
# scripts/analyses/direct-judge/slurm/serve_and_score.sbatch.
#
# usage: sbatch -J <name> --array=0-2 --export=ALL,CONFIG=configs/manipulations/<name>.yaml manipulations.sbatch
# Array index picks the judge - one array task per judge, each serving its
# own vllm and judging only CONFIG. Condition logs land
# under $PROJ/repo/logs/pairwise-newjudges/manipulations/ (run.py resolves
# against the repo root); pair dirs with a success log are skipped, so a
# requeued or re-launched task resumes rather than redoing finished pairs.
set -euo pipefail

BASE=${BASE:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge}
PROJ=${PROJ:-$BASE/code/pairwise_newjudges}
VENV=${VENV:-$BASE/code/judge_variation/venv}            # reused read-only, same run.py deps
VLLM_BIN=${VLLM_BIN:-$BASE/code/judge_ft/venv/bin/vllm}  # reused read-only, own shebang

# Explicit judge list, one array task each - not indexed off whatever is on
# disk. Override a weights dir at submit time if the staged path differs,
# e.g. sbatch --export=ALL,WEIGHTS_GPT_OSS_20B=/other/path ...
JUDGES=(qwen3-8b qwen3-32b gpt-oss-20b)
WEIGHTS_DIRS=(
  "${WEIGHTS_QWEN3_8B:-$BASE/models/Qwen3-8B}"
  "${WEIGHTS_QWEN3_32B:-$BASE/models/Qwen3-32B}"
  "${WEIGHTS_GPT_OSS_20B:-$BASE/models/gpt-oss-20b}"
)
MODES=(native native gptoss)

: "${SLURM_ARRAY_TASK_ID:?run with --array=0-2, one array task per judge}"
JUDGE=${JUDGES[$SLURM_ARRAY_TASK_ID]}
WEIGHTS=${WEIGHTS_DIRS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

# Requiring one path prevents a resumed/new campaign from silently picking up
# old configs or whatever else happens to be present in the directory.
: "${CONFIG:?submit with --export=ALL,CONFIG=configs/manipulations/<condition>.yaml}"
CONFIGS=("$CONFIG")

source "$VENV/bin/activate"
export PYTHONPATH="$PROJ/repo/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export LOCAL_API_KEY=none
RUN_ARGS=${RUN_ARGS:-}   # e.g. sbatch --export=ALL,RUN_ARGS="--limit 10" for pilots
QUANT=${QUANT:-}

SERVE_ARGS=()
case "$MODE" in
  native) SERVE_ARGS=(--max-model-len 32768 --reasoning-parser qwen3) ;;
  gptoss) SERVE_ARGS=(--max-model-len 131072 --reasoning-parser openai_gptoss) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

# Per-job port: gpu_lowp packs several 1-GPU jobs per node; a shared port
# means every job silently talks to whichever vllm bound it first.
PORT=$((10000 + SLURM_JOB_ID % 20000))

[ -n "$QUANT" ] && SERVE_ARGS+=(--quantization "$QUANT")
"$VLLM_BIN" serve "$WEIGHTS" --served-model-name judge --port "$PORT" \
  --gpu-memory-utilization 0.92 "${SERVE_ARGS[@]}" &
VLLM_PID=$!

for i in $(seq 1 120); do
  kill -0 $VLLM_PID || { echo "vllm died during startup"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "vllm never came up"; exit 1; }

cd "$PROJ/repo/scripts/pairwise-newjudges"
for CFG in "${CONFIGS[@]}"; do
  echo "== $(date -Is) $JUDGE $MODE $CFG"
  python run.py --config "$CFG" --judge-model openai-api/local/judge --judge-short "$JUDGE" \
    --base-url "http://127.0.0.1:$PORT/v1" $RUN_ARGS
done

kill $VLLM_PID
