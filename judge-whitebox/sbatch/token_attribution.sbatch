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
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD" >&2; exit 3; }

MODEL="${WB_MODEL:?set WB_MODEL}"
FOLD="${WB_FOLD:?set WB_FOLD=A|B}"
TAG="${WB_TAG:?set a new WB_TAG}"
DEST="$OUT/token_attribution/$TAG/attributions_${MODEL}_${FOLD}.parquet"
[ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
mkdir -p "$(dirname "$DEST")"

ARGS=(--model "$MODEL" --fold "$FOLD" --split "${WB_SPLIT:?set WB_SPLIT}" \
      --steps "${WB_STEPS:-64}" --out "$DEST")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
python token_attribution.py "${ARGS[@]}"
