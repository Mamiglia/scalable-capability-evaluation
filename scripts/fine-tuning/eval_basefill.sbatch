#!/bin/bash
#SBATCH --job-name=jft-basefill8b
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Pair-match the base/plain-8B judges to the combined-8B held-out sets so the
# base/plain/combined comparison figures are on identical pairs. Combined already
# has eval2_spoof/eval2_style (pairs_spoof_heldout / pairs_style_heldout); plain
# already has style_heldout_plain. Fill the rest. New files only (Rule 1).
#   sbatch eval_basefill.sbatch
source hpc_common.sh
A="$OUT/adapters/final"   # 8B plain-FT adapter

echo "=== base-8B on spoof heldout $(date) ==="
[ -f "$OUT/spoof_heldout_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/spoof_heldout_base.jsonl"

echo "=== base-8B on style heldout $(date) ==="
[ -f "$OUT/style_heldout_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_style_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/style_heldout_base.jsonl"

echo "=== plain-FT-8B on spoof heldout $(date) ==="
[ -f "$OUT/spoof_heldout_plain.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/spoof_heldout_plain.jsonl"

echo "=== BASEFILL_DONE $(date) ==="
