#!/bin/bash
#SBATCH --job-name=jft-spoofaug
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Spoof-augmented 0.6B: train on pairs_spoofaug (epoch pool + tone-crossed spoof
# pairs), then verdict-eval plain-FT vs spoof-trained on the HELD-OUT spoof pairs.
# Same hyperparams as the 0.6B final (lr 1e-4, r64). Stages skip if output exists.
# Submit: sbatch --partition=gpu_lowp --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=out_qwen06 spoof_retrain.sbatch
source hpc_common.sh
A_PLAIN="$OUT/adapters/final"
A_AUG="$OUT/adapters/spoofaug"
test -f "$IN/pairs_spoofaug.parquet" || { echo "MISSING pairs_spoofaug.parquet"; exit 1; }

echo "=== train spoofaug $(date) ==="
[ -f "$A_AUG/adapter_config.json" ] || \
python train.py --pairs "$IN/pairs_spoofaug.parquet" --base-model "$MODEL" \
  --lr 1e-4 --lora-rank 64 --splits train,val --out-dir "$A_AUG"

echo "=== held-out spoof eval: plain-FT $(date) ==="
[ -f "$OUT/spoof_heldout_plain.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A_PLAIN" --out "$OUT/spoof_heldout_plain.jsonl"

echo "=== held-out spoof eval: spoof-trained $(date) ==="
[ -f "$OUT/spoof_heldout_spoofaug.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A_AUG" --out "$OUT/spoof_heldout_spoofaug.jsonl"

echo "=== SPOOFAUG_DONE $(date) ==="
