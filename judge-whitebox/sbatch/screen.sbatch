#!/bin/bash
#SBATCH --job-name=wb-screen
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-3
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%A_%a.out

# Array of 4: {base,ft} x {judging,control}, one model per submission. Full eval_mm
# judging set + full control set (no --limit). ft uses the adapter at WB_ADAPTER
# (default below is a placeholder — override once fine-tuning selects a final adapter):
#   sbatch --export=ALL,WB_MODEL=Qwen3-0.6B screen.sbatch
#   sbatch --export=ALL,WB_MODEL=Qwen3-0.6B,WB_ADAPTER=/path/to/adapter screen.sbatch
# Submit from judge-whitebox/ on ceph (holds screen.py, config.yaml, wb_common.sh).
source ./wb_common.sh || { echo "wb_common.sh not found in $PWD; submit from a dir that has it" >&2; exit 3; }

MNAME="${WB_MODEL:-Qwen3-8B}"
ADAPTER="${WB_ADAPTER:-$C/staging/judge_ft/out/adapters/final}"

VARIANTS=(base base ft ft)
SETS=(judging control judging control)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SET="${SETS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$OUT/decomps" "$OUT/screens"
# pairs-tag the screen output so runs on different pair universes (eval_mm vs the
# gpqa_indomain sweep split) never collide on one filename. Decomp stays pairs-agnostic
# (function of the weights only; its identity guard catches a model/adapter mismatch).
PTAG=$(basename "${WB_PAIRS:-default}" .parquet)
# WB_INVERT=1: render the inverted (pick-weaker) verdict prompt; judging set only, tagged _inv.
# Decomp is unchanged (function of weights only), so inverted screens reuse the banked cache.
INV_ARG=(); INV_TAG=""
if [ -n "${WB_INVERT:-}" ] && [ "$SET" = "judging" ]; then INV_ARG=(--invert); INV_TAG="_inv"; fi
DECOMP="$OUT/decomps/${MNAME}_${VARIANT}.pt"
DEST="$OUT/screens/screen_${MNAME}_${VARIANT}_${SET}${INV_TAG}_${PTAG}.pt"

ARGS=(--model "$MNAME" --set "$SET" --decomp "$DECOMP" --out "$DEST" "${INV_ARG[@]}")
[ "$VARIANT" = "ft" ] && ARGS+=(--adapter "$ADAPTER")
[ -n "$WB_LIMIT" ] && ARGS+=(--limit "$WB_LIMIT")

echo "=== [task $SLURM_ARRAY_TASK_ID] screen $MNAME $VARIANT $SET $(date) ==="
python screen.py "${ARGS[@]}"
echo "=== done $MNAME $VARIANT $SET $(date) ==="
