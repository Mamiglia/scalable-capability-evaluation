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
# timeline holds every layer in host RAM at once: override --mem well above the 80G default.
# sbatch --mem=200G --export=ALL,WB_STAGE=timeline,WB_TAG=pilot,WB_RUN=qwen,WB_MANIFEST=/path/manifest.json,WB_MANIFOLD_OVERLAY=/path/overlay token_manifold.sbatch
# sbatch --export=ALL,WB_STAGE=bakeoff,WB_TAG=pilot,WB_RUN=qwen,WB_MANIFEST=/path/manifest.json,WB_LAYER=12,WB_MANIFOLD_OVERLAY=/path/overlay token_manifold.sbatch
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD; submit from a dir that has it" >&2; exit 3; }

STAGE="${WB_STAGE:?set WB_STAGE=estimate|record|timeline|bakeoff|trajectory|trace|crossfit}"
TAG="${WB_TAG:?set a new WB_TAG}"
ROOT="$OUT/token_manifold/$TAG"
mkdir -p "$ROOT"

case "$STAGE" in
  estimate)
    MODEL="${WB_MODEL:?set WB_MODEL}"
    FOLD="${WB_FOLD:?set WB_FOLD=A|B}"
    ARGS=(--model "$MODEL" --fold "$FOLD" --layers "${WB_LAYERS:-auto}"
          --out /unused --estimate-only)
    [ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
    python token_manifold_record.py "${ARGS[@]}"
    ;;
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
    export PYTHONPATH="${WB_MANIFOLD_OVERLAY:?set WB_MANIFOLD_OVERLAY to the manifold dependency overlay dir}${PYTHONPATH:+:$PYTHONPATH}"
    RUN="${WB_RUN:?set a model-specific WB_RUN}"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    DEST="$ROOT/timeline_${RUN}.csv"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    python token_manifold_discover.py timeline --manifest "$MANIFEST" --out "$DEST"
    ;;
  bakeoff)
    export PYTHONPATH="${WB_MANIFOLD_OVERLAY:?set WB_MANIFOLD_OVERLAY to the manifold dependency overlay dir}${PYTHONPATH:+:$PYTHONPATH}"
    RUN="${WB_RUN:?set a model-specific WB_RUN}"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    LAYER="${WB_LAYER:?set WB_LAYER}"
    DEST="$ROOT/bakeoff_${RUN}_L${LAYER}"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    python token_manifold_discover.py bakeoff --manifest "$MANIFEST" --layer "$LAYER" \
      --methods "${WB_METHODS:-cebra,bsf,sae}" --seeds "${WB_SEEDS:-0,1,2,3,4}" --out "$DEST"
    ;;
  trajectory)
    export PYTHONPATH="${WB_MANIFOLD_OVERLAY:?set WB_MANIFOLD_OVERLAY to the manifold dependency overlay dir}${PYTHONPATH:+:$PYTHONPATH}"
    RUN="${WB_RUN:?set a model-specific WB_RUN}"
    METHOD="${WB_METHOD:?set WB_METHOD=cebra|bsf|sae}"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    LAYERS="${WB_LAYERS:?set comma-separated physical WB_LAYERS}"
    DEST="$ROOT/trajectory_${RUN}_${METHOD}"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    ARGS=(trajectory --manifest "$MANIFEST" --layers "$LAYERS" --method "$METHOD"
          --seeds "${WB_SEEDS:-0,1,2}" --positions-per-prompt "${WB_POSITIONS:-64}"
          --cv-folds "${WB_CV_FOLDS:-3}" --out "$DEST")
    [ "${WB_QUICK:-0}" = 1 ] && ARGS+=(--quick)
    python token_manifold_discover.py "${ARGS[@]}"
    ;;
  trace)
    export PYTHONPATH="${WB_MANIFOLD_OVERLAY:?set WB_MANIFOLD_OVERLAY to the manifold dependency overlay dir}${PYTHONPATH:+:$PYTHONPATH}"
    RUN="${WB_RUN:?set a model-specific WB_RUN}"
    METHOD="${WB_METHOD:?set WB_METHOD=cebra|bsf|sae}"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    LAYER="${WB_LAYER:?set one physical WB_LAYER}"
    DEST="$ROOT/trace_${RUN}_${METHOD}_L${LAYER}"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    ARGS=(trace --manifest "$MANIFEST" --layer "$LAYER" --method "$METHOD"
          --seed "${WB_SEED:-0}" --positions-per-prompt "${WB_POSITIONS:-64}"
          --out "$DEST")
    [ "${WB_QUICK:-0}" = 1 ] && ARGS+=(--quick)
    python token_manifold_discover.py "${ARGS[@]}"
    ;;
  crossfit)
    export PYTHONPATH="${WB_MANIFOLD_OVERLAY:?set WB_MANIFOLD_OVERLAY to the manifold dependency overlay dir}${PYTHONPATH:+:$PYTHONPATH}"
    RUN="${WB_RUN:?set a model-specific WB_RUN}"
    MANIFEST="${WB_MANIFEST:?set WB_MANIFEST}"
    LAYER="${WB_LAYER:?set one physical WB_LAYER}"
    DEST="$ROOT/crossfit_${RUN}_cebra_L${LAYER}"
    [ ! -e "$DEST" ] || { echo "destination exists: $DEST"; exit 1; }
    ARGS=(crossfit-trace --manifest "$MANIFEST" --layer "$LAYER"
          --seeds "${WB_SEEDS:-0,1,2}" --positions-per-prompt "${WB_POSITIONS:-64}"
          --cv-folds "${WB_CV_FOLDS:-2}" --out "$DEST")
    [ "${WB_QUICK:-0}" = 1 ] && ARGS+=(--quick)
    python token_manifold_discover.py "${ARGS[@]}"
    ;;
  *)
    echo "unknown WB_STAGE: $STAGE" >&2
    exit 2
    ;;
esac
