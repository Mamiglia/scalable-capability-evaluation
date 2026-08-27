#!/usr/bin/env bash
#SBATCH --job-name=intermediate-errors
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
# Serve Qwen3-32B with vLLM on a per-job port, annotate intermediate errors, stop the server.
# From the staging dir:  sbatch repo/scripts/verdict_dissection/annotate/intermediate_errors.sbatch
set -euo pipefail
staging=$SLURM_SUBMIT_DIR
base=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
model_dir=$base/models/Qwen3-32B
vllm=$base/code/judge_ft/venv/bin/vllm
python=$base/code/judge_ft/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
port=$((10000 + SLURM_JOB_ID % 20000))
cd "$staging"
"$vllm" serve "$model_dir" --served-model-name Qwen3-32B --port "$port" --max-model-len 20000 \
    --gpu-memory-utilization 0.95 --reasoning-parser qwen3 --dtype bfloat16 > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
server=$!
for i in $(seq 1 120); do
    sleep 10
    kill -0 $server 2>/dev/null || { echo "vllm died"; tail -30 "logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
    curl -sf "http://localhost:$port/health" > /dev/null && break
done
curl -sf "http://localhost:$port/health" > /dev/null || { echo "server never became healthy"; exit 1; }
"$python" repo/scripts/verdict_dissection/annotate/intermediate_errors.py \
    --input data/transcripts_gpqa_aime.parquet --output outputs/intermediate_errors.jsonl \
    --base-url "http://localhost:$port/v1" --model Qwen3-32B --workers 32 "$@"
kill $server
