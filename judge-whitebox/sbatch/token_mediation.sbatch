#!/bin/bash
#SBATCH --job-name=wb-token-mediation
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Sentence-feature extraction from a residual recording. Submit from the code dir.
# sbatch --export=ALL,WB_MODEL=Llama-3.2-3B-Instruct,WB_TAG=tokattr_valid1106,\
#   WB_FEAT_TAG=feat_valid1106 token_mediation.sbatch
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD" >&2; exit 3; }

NAME="${WB_MODEL:?set WB_MODEL}"
TAG="${WB_TAG:?set WB_TAG of the recording}"
FEAT="${WB_FEAT_TAG:?set a WB_FEAT_TAG}"
MANIFEST="$OUT/token_manifold/$TAG/record_${NAME}_all/manifest.json"
DEST="$OUT/token_mediation/$FEAT/$NAME"
[ ! -e "$DEST/index.parquet" ] || { echo "destination complete: $DEST"; exit 0; }
mkdir -p "$DEST"

ARGS=(--manifest "$MANIFEST" --tokenizer "$C/models/$NAME" --out "$DEST")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
python token_mediation.py "${ARGS[@]}"
