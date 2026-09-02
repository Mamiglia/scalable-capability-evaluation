#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

# usage: sbatch -J <name> attribution.sbatch <weights_dir> <judge_short> <native|llama>
# Family-attribution side task on gpqa + otis-aime; resumes if logs exist.
set -euo pipefail

BASE=${BASE:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge}
PROJ=${PROJ:-$BASE/code/judge_variation}
VENV=${VENV:-$PROJ/venv}
WEIGHTS=$1; JUDGE=$2; MODE=$3

source "$VENV/bin/activate"
export PYTHONPATH="$PROJ/repo/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export LOCAL_API_KEY=none
PORT=$((10000 + SLURM_JOB_ID % 20000))

SERVE_ARGS=(--max-model-len 32768)
[ "$MODE" = native ] && SERVE_ARGS+=(--reasoning-parser qwen3)

VLLM_BIN=$BASE/code/judge_ft/venv/bin/vllm
"$VLLM_BIN" serve "$WEIGHTS" --served-model-name judge --port "$PORT" \
  --gpu-memory-utilization 0.92 "${SERVE_ARGS[@]}" &
VLLM_PID=$!

for i in $(seq 1 120); do
  kill -0 $VLLM_PID || { echo "vllm died during startup"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "vllm never came up"; exit 1; }

cd "$PROJ/repo/analyses/judge-variation"
for BENCH in gpqa otis-aime; do
  echo "== $(date -Is) attribution $JUDGE $BENCH"
  python attribution.py --bench "$BENCH" --judge-model openai-api/local/judge \
    --judge-short "$JUDGE" --base-url "http://127.0.0.1:$PORT/v1"
done

kill $VLLM_PID
