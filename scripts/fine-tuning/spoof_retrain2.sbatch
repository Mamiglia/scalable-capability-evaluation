#!/bin/bash
#SBATCH --job-name=jft-spoofaug2
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Re-run of the spoof-augmented 0.6B after the v1 run false-aborted at step 1000
# (config abort_check_steps now 10000). New adapter dir (v1's stays; ceph is
# append-only). Plain-FT held-out eval already exists (spoof_heldout_plain.jsonl).
# Submit: sbatch --partition=gpu_lowp --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=out_qwen06 spoof_retrain2.sbatch
source hpc_common.sh
A_AUG="$OUT/adapters/spoofaug_v2"
test -f "$IN/pairs_spoofaug.parquet" || { echo "MISSING pairs_spoofaug.parquet"; exit 1; }

echo "=== train spoofaug_v2 (grace 10000) $(date) ==="
[ -f "$A_AUG/adapter_config.json" ] || \
python train.py --pairs "$IN/pairs_spoofaug.parquet" --base-model "$MODEL" \
  --lr 1e-4 --lora-rank 64 --splits train,val --out-dir "$A_AUG"

echo "=== held-out spoof eval: spoof-trained v2 $(date) ==="
[ -f "$OUT/spoof_heldout_spoofaug_v2.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_spoof_heldout.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$A_AUG" --out "$OUT/spoof_heldout_spoofaug_v2.jsonl"

echo "=== SPOOFAUG2_DONE $(date) ==="
