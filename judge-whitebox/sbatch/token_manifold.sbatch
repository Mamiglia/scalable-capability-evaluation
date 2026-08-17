#!/bin/bash
#SBATCH --job-name=wb-token-manifold
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# User-gated stages; submit from staging/whitebox/code after reviewing the storage estimate.
# sbatch --export=ALL,WB_STAGE=record,WB_TAG=pilot,WB_MODEL=Qwen3-8B,WB_FOLD=A token_manifold.sbatch
# sbatch --export=ALL,WB_STAGE=timeline,WB_TAG=pilot,WB_MANIFEST=/path/manifest.json,WB_MANIFOLD_VENV=/path/venv token_manifold.sbatch
# sbatch --export=ALL,WB_STAGE=bakeoff,WB_TAG=pilot,WB_MANIFEST=/path/manifest.json,WB_LAYER=12,WB_MANIFOLD_VENV=/path/venv token_manifold.sbatch
source wb_common.sh

STAGE="${WB_STAGE:?set WB_STAGE=record|timeline|bakeoff}"
TAG="${WB_TAG:?set a new WB_TAG}"
ROOT="$OUT/token_manifold/$TAG"
mkdir -p "$ROOT"

case "$STAGE" in
  record)
    MODEL="${WB_MODEL:?set WB_MODEL}"
    FOLD="${WB_FOLD:?set WB_FOLD=A|B}"
    DEST="$ROOT/record_${MODEL}_${FOLD}"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    ARGS=(--model "$MODEL" --fold "$FOLD" --layers "${WB_LAYERS:-auto}"
          --shard-prompts "${WB_SHARD_PROMPTS:-8}" --out "$DEST")
    [ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
    python token_manifold_record.py "${ARGS[@]}"
    ;;
  timeline)
    source "${WB_MANIFOLD_VENV:?set WB_MANIFOLD_VENV}/bin/activate"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    DEST="$ROOT/timeline.csv"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    python token_manifold_discover.py timeline --manifest "$MANIFEST" --out "$DEST"
    ;;
  bakeoff)
    source "${WB_MANIFOLD_VENV:?set WB_MANIFOLD_VENV}/bin/activate"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    LAYER="${WB_LAYER:?set WB_LAYER}"
    DEST="$ROOT/bakeoff_L${LAYER}"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    python token_manifold_discover.py bakeoff --manifest "$MANIFEST" --layer "$LAYER" \
      --methods "${WB_METHODS:-cebra,bsf,sae}" --seeds "${WB_SEEDS:-0,1,2,3,4}" --out "$DEST"
    ;;
  *)
    echo "unknown WB_STAGE: $STAGE" >&2
    exit 2
    ;;
esac
