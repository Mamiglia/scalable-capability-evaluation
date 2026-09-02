#!/bin/bash
#SBATCH --job-name=jft-rewriteaug
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# 3rd SFT set: epoch-pool train + 260 rewrite-augmented pairs (neutral + cross-model
# style, voice-independent of label). Train 0.6B, then verdict-eval on eval_mm
# (base-task, did it degrade?) and on the held-out spoof pairs (does style-training
# transfer to confidence-robustness?). Grace 10000 (config). Stages skip if done.
# Submit: sbatch --partition=gpu_lowp --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=out_qwen06 rewrite_retrain.sbatch
source hpc_common.sh
A="$OUT/adapters/rewriteaug"
test -f "$IN/pairs_rewriteaug.parquet" || { echo "MISSING pairs_rewriteaug.parquet"; exit 1; }

echo "=== train rewriteaug $(date) ==="
[ -f "$A/adapter_config.json" ] || \
python train.py --pairs "$IN/pairs_rewriteaug.parquet" --base-model "$MODEL" \
  --lr 1e-4 --lora-rank 64 --splits train,val --out-dir "$A"

echo "=== eval base task (eval_mm) $(date) ==="
[ -f "$OUT/rewriteaug_eval_mm.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --splits eval_mm --out "$OUT/rewriteaug_eval_mm.jsonl"

echo "=== eval held-out spoof (cross-benefit) $(date) ==="
[ -f "$OUT/rewriteaug_spoof_heldout.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A" --out "$OUT/rewriteaug_spoof_heldout.jsonl"

echo "=== REWRITEAUG_DONE $(date) ==="
