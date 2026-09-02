#!/bin/bash
#SBATCH --job-name=jft-phase2-verdicts
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Phase-2 verdicts: base / plain-LOO / aug-LOO on each held-out domain's plain +
# style-swap eval. New jsonls only.
#   sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=cross/Qwen3-0.6B verdicts_phase2.sbatch
source hpc_common.sh
python verdicts_phase2.py --base-model "$MODEL" --out-root "$OUT" \
  --pairs-dir "$IN/cross_domain" --tp 1
