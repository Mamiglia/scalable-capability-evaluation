#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

# usage: sbatch -J <name> serve_and_judge.sbatch <weights_dir> <served_name> <qwen|llama> <config yaml...>
# Serves one judge on a per-job port, then runs every config against it with
# run.py --judge <served_name>. Pair dirs with a success log are skipped on
# requeue. Judge logs land under $PROJ/repo/logs/eval-variation/.
set -euo pipefail

BASE=${BASE:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge}
PROJ=${PROJ:-$BASE/code/eval_variation}
# judge_variation's venv, used read-only; PYTHONPATH points imports at OUR repo
VENV=${VENV:-$BASE/code/judge_variation/venv}
WEIGHTS=$1; SERVED=$2; MODE=$3; shift 3

source "$VENV/bin/activate"
export PYTHONPATH="$PROJ/repo/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export LOCAL_API_KEY=none
RUN_ARGS=${RUN_ARGS:-}   # e.g. sbatch --export=ALL,RUN_ARGS="--limit 10" for pilots

SERVE_ARGS=(--max-model-len 32768)
[ "$MODE" = "qwen" ] && SERVE_ARGS+=(--reasoning-parser qwen3)

# per-job port: several 1-GPU jobs share a node; a fixed port cross-wires them
PORT=$((10000 + SLURM_JOB_ID % 20000))

VLLM_BIN=$BASE/code/judge_ft/venv/bin/vllm
"$VLLM_BIN" serve "$WEIGHTS" --served-model-name "$SERVED" --port "$PORT" \
  --gpu-memory-utilization 0.92 "${SERVE_ARGS[@]}" &
VLLM_PID=$!

for i in $(seq 1 120); do
  kill -0 $VLLM_PID || { echo "vllm died during startup"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 5
done

cd "$PROJ/repo"
for cfg in "$@"; do
  python analyses/eval-variation/run.py "$cfg" --judge "$SERVED" \
    --model-base-url "http://127.0.0.1:$PORT/v1" $RUN_ARGS
done
kill $VLLM_PID
