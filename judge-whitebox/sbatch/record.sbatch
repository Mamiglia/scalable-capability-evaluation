#!/bin/bash
#SBATCH --job-name=wb-record
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Residual-stream recording, forward-only.
#   pairwise (default): array 0-1 = normal / inverted verdict prompt.
#   WB_SINGLE=1: single-transcript neutral read frame (geometry); one task, no inversion.
# Submit from staging/whitebox/code: sbatch --array=0-1 --export=ALL,WB_MODEL=...,WB_LIMIT=...,WB_PAIRS=... [WB_ADAPTER=...] sbatch/record.sbatch
#                                    sbatch          --export=ALL,WB_SINGLE=1,WB_MODEL=...,WB_PAIRS=... [WB_ADAPTER=...] sbatch/record.sbatch
source wb_common.sh
MNAME="${WB_MODEL:-Qwen3-8B}"
VAR="base"; [ -n "${WB_ADAPTER:-}" ] && VAR="ft"
mkdir -p "$OUT/resid"
PTAG=$(basename "${WB_PAIRS:-default}" .parquet)
if [ -n "${WB_SINGLE:-}" ]; then
  TAG="single"
  ARGS=(--model "$MNAME" --single-transcript --out "$OUT/resid/resid_${MNAME}_${VAR}_${TAG}_${PTAG}.pt")
else
  INV=(); TAG="norm"
  [ "$SLURM_ARRAY_TASK_ID" = "1" ] && { INV=(--invert); TAG="inv"; }
  ARGS=(--model "$MNAME" "${INV[@]}" --out "$OUT/resid/resid_${MNAME}_${VAR}_${TAG}_${PTAG}.pt")
fi
[ -n "${WB_ADAPTER:-}" ] && ARGS+=(--adapter "$WB_ADAPTER")
[ -n "${WB_LIMIT:-}" ] && ARGS+=(--limit "$WB_LIMIT")
echo "=== record $MNAME $VAR $TAG $(date) ==="
python record_resid.py "${ARGS[@]}"
echo "=== done $(date) ==="
