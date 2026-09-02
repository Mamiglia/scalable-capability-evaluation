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
# Additivity probe (WB_ADDITIVITY_KS is comma-valued, so export it the same way):
# WB_ADDITIVITY_KS=1,2,4,8,16,32,64,128 sbatch --export=ALL,WB_MODEL=Llama-3.2-3B-Instruct,\
#   WB_FOLD=all,WB_SPLIT=train,WB_TAG=addprobe_v1,WB_LIMIT=10 token_attribution.sbatch
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
# Part-directory modes resume into an existing destination; only the
# single-file modes must refuse to overwrite.
RESUMABLE=0
if [ -n "${WB_SWEEP_STEPS:-}" ]; then
  DEST="$OUT/token_attribution/$TAG/completeness_sweep_${MODEL}_${FOLD}${SHARD_SUFFIX}.parquet"
  ARGS+=(--sweep-steps "$WB_SWEEP_STEPS")
  [ -n "${WB_DTYPE:-}" ] && ARGS+=(--dtype "$WB_DTYPE")
  [ -n "${WB_SWEEP_PROMPTS:-}" ] && ARGS+=(--sweep-prompts "$WB_SWEEP_PROMPTS")
  [ -n "${WB_SWEEP_SEED:-}" ] && ARGS+=(--sweep-seed "$WB_SWEEP_SEED")
elif [ -n "${WB_BLANK_GROUPS:-}" ]; then
  DEST="$OUT/token_attribution/$TAG/blank_groups_${MODEL}_${FOLD}${SHARD_SUFFIX}"
  RESUMABLE=1
  ARGS+=(--blank-groups --batch-spans "${WB_BATCH_SPANS:-8}")
  [ -n "${WB_DTYPE:-}" ] && ARGS+=(--dtype "$WB_DTYPE")
elif [ -n "${WB_ADDITIVITY_KS:-}" ]; then
  # Comma-valued: export WB_ADDITIVITY_KS in the submitting shell, --export=ALL only.
  DEST="$OUT/token_attribution/$TAG/additivity_${MODEL}_${FOLD}${SHARD_SUFFIX}"
  RESUMABLE=1
  ARGS+=(--additivity-ks "$WB_ADDITIVITY_KS"
         --additivity-triples "${WB_ADDITIVITY_TRIPLES:-64}"
         --batch-spans "${WB_BATCH_SPANS:-8}")
  [ -n "${WB_DTYPE:-}" ] && ARGS+=(--dtype "$WB_DTYPE")
else
  DEST="$OUT/token_attribution/$TAG/attributions_${MODEL}_${FOLD}${SHARD_SUFFIX}.parquet"
  if [ -n "${WB_MASK_DRAWS:-}" ]; then
    RESUMABLE=1
    ARGS+=(--mask-draws "$WB_MASK_DRAWS" --mask-size "${WB_MASK_SIZE:-32}"
           --mask-seed "${WB_MASK_SEED:-0}" --batch-spans "${WB_BATCH_SPANS:-8}")
  elif [ -n "${WB_OCCLUDE_SPAN:-}" ]; then
    RESUMABLE=1
    ARGS+=(--occlude-span "$WB_OCCLUDE_SPAN" --batch-spans "${WB_BATCH_SPANS:-8}")
  else
    ARGS+=(--steps "${WB_STEPS:?set WB_STEPS from the sweep}")
  fi
  [ -n "${WB_DTYPE:-}" ] && ARGS+=(--dtype "$WB_DTYPE")
fi
[ "$RESUMABLE" = 1 ] || [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
mkdir -p "$(dirname "$DEST")"

ARGS+=(--out "$DEST")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
[ -n "${WB_LIMIT_SAMPLING:-}" ] && ARGS+=(--limit-sampling "$WB_LIMIT_SAMPLING")
[ -n "${WB_MIN_REASONING:-}" ] && ARGS+=(--min-reasoning-chars "$WB_MIN_REASONING")
python token_attribution.py "${ARGS[@]}"
