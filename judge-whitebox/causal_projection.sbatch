#!/bin/bash
#SBATCH --job-name=wb-causal-projection
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Submit from the ceph code dir; directions_L14.npz must sit next to this file.
# Smoke:
# sbatch --export=ALL,WB_MODEL=Llama-3.2-3B-Instruct,WB_SPLIT=train,WB_TAG=causal_smoke1,\
#   WB_LIMIT=2,WB_LIMIT_SAMPLING=balanced,WB_MIN_REASONING=100 causal_projection.sbatch
# Campaign:
# sbatch --array=0-3 --export=ALL,WB_SHARDS=4,WB_MODEL=Llama-3.2-3B-Instruct,WB_SPLIT=train,\
#   WB_TAG=causal_v1,WB_LIMIT=1136,WB_LIMIT_SAMPLING=balanced,WB_MIN_REASONING=100 \
#   causal_projection.sbatch
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD" >&2; exit 3; }

MODEL="${WB_MODEL:?set WB_MODEL}"
TAG="${WB_TAG:?set a new WB_TAG}"
SHARDS="${WB_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

DEST="$OUT/token_attribution/$TAG/projection_${MODEL}_shard${SHARD}of${SHARDS}"
mkdir -p "$(dirname "$DEST")"

ARGS=(--model "$MODEL" --directions ./directions_L14.npz
      --split "${WB_SPLIT:?set WB_SPLIT}"
      --shard-index "$SHARD" --shard-count "$SHARDS" --out "$DEST")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
[ -n "${WB_LIMIT_SAMPLING:-}" ] && ARGS+=(--limit-sampling "$WB_LIMIT_SAMPLING")
[ -n "${WB_MIN_REASONING:-}" ] && ARGS+=(--min-reasoning-chars "$WB_MIN_REASONING")
[ -n "${WB_BATCH:-}" ] && ARGS+=(--batch "$WB_BATCH")
# comma-valued: export in the submitting shell, --export=ALL only
[ -n "${WB_RESTRICTIONS:-}" ] && ARGS+=(--restrictions "$WB_RESTRICTIONS")
[ -n "${WB_ONLY_DIRECTIONS:-}" ] && ARGS+=(--only-directions "$WB_ONLY_DIRECTIONS")
[ -n "${WB_DOSES:-}" ] && ARGS+=(--doses "$WB_DOSES")
[ -n "${WB_N_RANDOM:-}" ] && ARGS+=(--n-random "$WB_N_RANDOM")
python causal_projection.py "${ARGS[@]}"
