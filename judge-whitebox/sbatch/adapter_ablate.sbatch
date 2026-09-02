#!/bin/bash
#SBATCH --job-name=wb-adapter-ablate
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Per-layer-group LoRA ablation (base/group_only/full/full_minus + derived sufficiency/
# necessity/interaction), full eval_mm split (no --limit). Pinned to h100: the 8B forward
# OOM'd a 40GB A100 before. WB_ADAPTER is required -- this ablates one specific fine-tune.
#   sbatch --export=ALL,WB_MODEL=Qwen3-8B,WB_ADAPTER=/path/to/adapter adapter_ablate.sbatch
# Submit from judge-whitebox/ on ceph (holds adapter_ablate.py, config.yaml, wb_common.sh).
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD; submit from a dir that has it" >&2; exit 3; }

MNAME="${WB_MODEL:-Qwen3-8B}"
ADAPTER="${WB_ADAPTER:?set WB_ADAPTER=path to the LoRA adapter}"

mkdir -p "$OUT/adapter_ablate"
DEST="$OUT/adapter_ablate/adapter_ablate_${MNAME}.csv"

ARGS=(--model "$MNAME" --adapter "$ADAPTER" --out "$DEST")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")

echo "=== adapter_ablate $MNAME $(date) ==="
python adapter_ablate.py "${ARGS[@]}"
echo "=== done $(date) ==="
