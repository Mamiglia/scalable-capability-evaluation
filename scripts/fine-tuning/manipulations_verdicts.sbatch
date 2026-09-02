#!/bin/bash
#SBATCH --job-name=jft-manip
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-15%16
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%A_%a.out

# Four variants x four condition shards. Every task is local vLLM inference;
# no API provider or API key is used.
set -euo pipefail

C=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
CODE=$C/code/judge_ft
PAIRWISE=$C/code/pairwise_newjudges/repo
PAIRS=$C/staging/judge_ft/manipulations/pairs
RESULTS=$C/staging/judge_ft/manipulations/results_r1

VARIANTS=(qwen3-8b-base qwen3-8b-ft llama-3.2-3b-base llama-3.2-3b-ft)
MODELS=(
  "$C/models/Qwen3-8B"
  "$C/models/Qwen3-8B"
  "$C/models/Llama-3.2-3B-Instruct"
  "$C/models/Llama-3.2-3B-Instruct"
)
ADAPTERS=(
  ""
  "$C/staging/judge_ft/cross_r1/Qwen3-8B/gpqa_indomain/adapters/final"
  ""
  "$C/staging/judge_ft/cross_r1/Llama-3.2-3B-Instruct/gpqa_indomain/adapters/final"
)

variant_index=$((SLURM_ARRAY_TASK_ID / 4))
shard=$((SLURM_ARRAY_TASK_ID % 4))
variant=${VARIANTS[$variant_index]}
model=${MODELS[$variant_index]}
adapter=${ADAPTERS[$variant_index]}

source "$CODE/venv/bin/activate"
export PYTHONPATH="$PAIRWISE/src:$CODE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0

args=(
  --base-model "$model"
  --variant "$variant"
  --shard "$shard"
  --pairs-dir "$PAIRS"
  --out-root "$RESULTS"
)
[ -n "$adapter" ] && args+=(--adapter "$adapter")

cd "$CODE"
python manipulations_verdicts.py "${args[@]}"
