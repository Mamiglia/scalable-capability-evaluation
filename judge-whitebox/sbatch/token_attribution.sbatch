#!/bin/bash
#SBATCH --job-name=wb-token-attribution
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Submit from a directory holding wb_common.sh.
# The split must match the recording's split, or the table and the states describe different prompts.
# sbatch --export=ALL,WB_MODEL=Qwen3-8B,WB_FOLD=A,WB_SPLIT=train,WB_TAG=attr_v1 token_attribution.sbatch
# Split the prompts over an array: WB_SHARDS must equal the array size, one shard per task.
# sbatch --array=0-7 --export=ALL,WB_SHARDS=8,WB_MODEL=Qwen3-8B,WB_FOLD=all,\
#   WB_SPLIT=train,WB_TAG=attr_v1,WB_STEPS=256 token_attribution.sbatch
# Set WB_SWEEP_STEPS (comma-separated) to measure where the integral converges instead;
# run that first and take WB_STEPS from it. Its commas would split a --export list, so
# export it in the environment:
# WB_SWEEP_STEPS=64,128,256,512,1024,2048 sbatch --export=ALL,WB_MODEL=Qwen3-8B,WB_FOLD=A,\
#   WB_SPLIT=train,WB_TAG=sweep_v1,WB_SWEEP_PROMPTS=20 token_attribution.sbatch
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD" >&2; exit 3; }

MODEL="${WB_MODEL:?set WB_MODEL}"
FOLD="${WB_FOLD:?set WB_FOLD=A|B|all}"
TAG="${WB_TAG:?set a new WB_TAG}"

# One shard per array task; a plain submission is the single shard 0 of 1.
SHARDS="${WB_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SUFFIX=""
[ "$SHARDS" -gt 1 ] && SHARD_SUFFIX="_shard${SHARD}of${SHARDS}"

ARGS=(--model "$MODEL" --fold "$FOLD" --split "${WB_SPLIT:?set WB_SPLIT}"
      --shard-index "$SHARD" --shard-count "$SHARDS")
[ -n "${WB_STEP_BATCH:-}" ] && ARGS+=(--step-batch "$WB_STEP_BATCH")
if [ -n "${WB_SWEEP_STEPS:-}" ]; then
  DEST="$OUT/token_attribution/$TAG/completeness_sweep_${MODEL}_${FOLD}${SHARD_SUFFIX}.parquet"
  ARGS+=(--sweep-steps "$WB_SWEEP_STEPS")
  [ -n "${WB_DTYPE:-}" ] && ARGS+=(--dtype "$WB_DTYPE")
  [ -n "${WB_SWEEP_PROMPTS:-}" ] && ARGS+=(--sweep-prompts "$WB_SWEEP_PROMPTS")
  [ -n "${WB_SWEEP_SEED:-}" ] && ARGS+=(--sweep-seed "$WB_SWEEP_SEED")
else
  DEST="$OUT/token_attribution/$TAG/attributions_${MODEL}_${FOLD}${SHARD_SUFFIX}.parquet"
  ARGS+=(--steps "${WB_STEPS:?set WB_STEPS from the sweep}")
  [ -n "${WB_DTYPE:-}" ] && ARGS+=(--dtype "$WB_DTYPE")
fi
[ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
mkdir -p "$(dirname "$DEST")"

ARGS+=(--out "$DEST")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
python token_attribution.py "${ARGS[@]}"
