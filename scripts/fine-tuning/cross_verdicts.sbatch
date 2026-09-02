#!/bin/bash
#SBATCH --job-name=jft-cross-verdicts
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Score every trained cell adapter (+ base floor) on each test domain's held-out
# eval_mm, for ONE model size. One model load. New jsonls only; resumable.
#   sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=cross/Qwen3-0.6B cross_verdicts.sbatch
#   sbatch --export=ALL,JFT_MODEL=Qwen3-8B,JFT_OUT=cross/Qwen3-8B cross_verdicts.sbatch
source hpc_common.sh
python cross_verdicts.py --base-model "$MODEL" --out-root "$OUT" \
  --pairs-dir "$IN/cross_domain" --tp 1
