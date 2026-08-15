#!/bin/bash
#SBATCH --job-name=wb-token-ablate
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Token-resolved causal ablation, batch-1. WB_MODE=base|ft. ft needs WB_ADAPTER (the LoRA to
# probe); base runs the plain model (no adapter). Full eval_mm split unless WB_LIMIT set. Pinned
# h100 for the 8B; override --gres=gpu:l40s:1 for the 3B. Submit from judge-whitebox/ on ceph.
#   sbatch --export=ALL,WB_MODEL=Qwen3-8B,WB_MODE=ft,WB_ADAPTER=/path/to/loo_gpqa adapter token_ablate.sbatch
#   sbatch --export=ALL,WB_MODEL=Qwen3-8B,WB_MODE=base token_ablate.sbatch
source wb_common.sh

MNAME="${WB_MODEL:-Qwen3-8B}"
MODE="${WB_MODE:?set WB_MODE=base|ft}"

mkdir -p "$OUT/token_ablate"
DEST="$OUT/token_ablate/token_ablate_${MNAME}_${MODE}.csv"

ARGS=(--model "$MNAME" --mode "$MODE" --out "$DEST")
if [ "$MODE" = "ft" ]; then ARGS+=(--adapter "${WB_ADAPTER:?set WB_ADAPTER for mode ft}"); fi
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")

echo "=== token_ablate $MNAME $MODE $(date) ==="
python token_ablate.py "${ARGS[@]}"
echo "=== done $(date) ==="
