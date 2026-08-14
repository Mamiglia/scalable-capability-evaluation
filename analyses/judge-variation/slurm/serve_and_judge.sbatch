#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

# usage: sbatch -J <name> serve_and_judge.sbatch <weights_dir> <judge_short> <native|yarn|llama> <config yaml...>
# Judge logs land under $PROJ/repo/logs/judging-variations/ (run.py resolves
# against the repo root); pair dirs with a success log are skipped on requeue.
set -euo pipefail

BASE=${BASE:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge}
PROJ=${PROJ:-$BASE/code/judge_variation}
VENV=${VENV:-$PROJ/venv}
WEIGHTS=$1; JUDGE=$2; MODE=$3; shift 3

source "$VENV/bin/activate"
export PYTHONPATH="$PROJ/repo/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export LOCAL_API_KEY=none
RUN_ARGS=${RUN_ARGS:-}   # e.g. sbatch --export=ALL,RUN_ARGS="--limit 10" for pilots
QUANT=${QUANT:-}         # e.g. QUANT=fp8 to fit a 32B on an L40S

SERVE_ARGS=()
case "$MODE" in
  native) SERVE_ARGS=(--max-model-len 32768 --reasoning-parser qwen3) ;;
  yarn)   SERVE_ARGS=(--max-model-len 131072 --reasoning-parser qwen3
                      --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}') ;;
  llama)  SERVE_ARGS=(--max-model-len 131072) ;;
  gptoss) SERVE_ARGS=(--max-model-len 131072 --reasoning-parser openai_gptoss) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

# Per-job port: gpu_lowp packs several 1-GPU jobs per node; a shared port
# means every job silently talks to whichever vllm bound it first.
PORT=$((10000 + SLURM_JOB_ID % 20000))

# judge_ft's vllm, used read-only via its own shebang interpreter
VLLM_BIN=$BASE/code/judge_ft/venv/bin/vllm
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

cd "$PROJ/repo/analyses/judge-variation"
for CFG in "$@"; do
  echo "== $(date -Is) $JUDGE $MODE $CFG"
  python run.py --config "$CFG" --judge-model openai-api/local/judge --judge-short "$JUDGE" \
    --base-url "http://127.0.0.1:$PORT/v1" $RUN_ARGS
done

kill $VLLM_PID
