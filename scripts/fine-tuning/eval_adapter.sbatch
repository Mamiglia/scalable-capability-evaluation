#!/bin/bash
#SBATCH --job-name=jft-evaladapter
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Verdicts for one adapter on one pairs file (baseline evals).
# sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=out_qwen06,JFT_ADAPTER=<abs>,JFT_EVALPAIRS=pairs_style_heldout.parquet,JFT_EVALOUT=style_heldout_plain eval_adapter.sbatch
source hpc_common.sh
[ -f "$OUT/${JFT_EVALOUT}.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/$JFT_EVALPAIRS" --base-model "$MODEL" --tp 1 \
  --adapter "$JFT_ADAPTER" --out "$OUT/${JFT_EVALOUT}.jsonl"
echo "=== EVAL_ADAPTER_DONE $(date) ==="
