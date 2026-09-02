#!/bin/bash
#SBATCH --job-name=jft-combined-final
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Force a specific (stable) final cell for the combined set + 3-way eval, when the
# grid pick chose an lr that diverges on full data (grid cells too short to show it).
# sbatch --export=ALL,JFT_MODEL=Qwen3-8B,JFT_OUT=out_combined8b,JFT_LR=1e-4,JFT_RANK=16 combined_final.sbatch
source hpc_common.sh
A="$OUT/adapters/final_stable"
test -f "$IN/pairs_combined.parquet" || { echo "MISSING pairs_combined.parquet"; exit 1; }

echo "=== final train lr=$JFT_LR rank=$JFT_RANK $(date) ==="
[ -f "$A/adapter_config.json" ] || \
python train.py --pairs "$IN/pairs_combined.parquet" --base-model "$MODEL" \
  --lr "$JFT_LR" --lora-rank "$JFT_RANK" --splits train,val --out-dir "$A"

echo "=== 3-way eval $(date) ==="
[ -f "$OUT/eval2_plain.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --splits eval_mm --out "$OUT/eval2_plain.jsonl"
[ -f "$OUT/eval2_spoof.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/eval2_spoof.jsonl"
[ -f "$OUT/eval2_style.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_style_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/eval2_style.jsonl"

echo "=== COMBINED_FINAL_DONE $(date) ==="
