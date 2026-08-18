#!/bin/bash
#SBATCH --job-name=wb-verdicts
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Verdict runs. Array 0-1 = normal / inverted.
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD; submit from a dir that has it" >&2; exit 3; }
MNAME="${WB_MODEL:-Qwen3-8B}"
VAR="base"; [ -n "${WB_ADAPTER:-}" ] && VAR="ft"
INV=(); TAG="norm"
[ "$SLURM_ARRAY_TASK_ID" = "1" ] && { INV=(--invert); TAG="inv"; }
mkdir -p "$OUT/verdicts"
ARGS=(--model "$MNAME" "${INV[@]}" --out "$OUT/verdicts/verdicts_${MNAME}_${VAR}_${TAG}.jsonl")
[ -n "${WB_ADAPTER:-}" ] && ARGS+=(--adapter "$WB_ADAPTER")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
echo "=== verdicts $MNAME $VAR $TAG $(date) ==="
python verdicts.py "${ARGS[@]}"
echo "=== done $(date) ==="
