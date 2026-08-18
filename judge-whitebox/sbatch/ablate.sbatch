#!/bin/bash
#SBATCH --job-name=wb-ablate
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%A.out

# Full single-mode battery (verdict acc/position term/flip rate + control NLL) for every top-N
# component of one model/variant, on the full eval_mm split + control corpus (no --limit).
# Needs screen.sbatch's decomp cache and importance_report.py's CSV already built for this
# model/variant. base uses no adapter; ft needs WB_ADAPTER:
#   sbatch --export=ALL,WB_MODEL=Qwen3-0.6B,WB_VARIANT=base,WB_COMPONENTS=$OUT/importance_base06.csv ablate.sbatch
#   sbatch --export=ALL,WB_MODEL=Qwen3-8B,WB_VARIANT=ft,WB_ADAPTER=/path/to/adapter,WB_COMPONENTS=$OUT/importance_ft8b.csv ablate.sbatch
# Submit from judge-whitebox/ on ceph (holds ablate.py, config.yaml, wb_common.sh).
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD; submit from a dir that has it" >&2; exit 3; }

MNAME="${WB_MODEL:-Qwen3-8B}"
VARIANT="${WB_VARIANT:?set WB_VARIANT=base|ft}"
COMPONENTS="${WB_COMPONENTS:?set WB_COMPONENTS=path to importance_x.csv}"

mkdir -p "$OUT/decomps" "$OUT/ablate"
DECOMP="$OUT/decomps/${MNAME}_${VARIANT}.pt"
DEST="$OUT/ablate/ablate_${MNAME}_${VARIANT}.jsonl"

ARGS=(--model "$MNAME" --components "$COMPONENTS" --mode single --decomp "$DECOMP" --out "$DEST")
[ "$VARIANT" = "ft" ] && ARGS+=(--adapter "${WB_ADAPTER:?set WB_ADAPTER for ft variant}")

echo "=== ablate $MNAME $VARIANT $(date) ==="
python ablate.py "${ARGS[@]}"
echo "=== done $MNAME $VARIANT $(date) ==="
