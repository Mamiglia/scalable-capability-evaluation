#!/bin/bash
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/code/verdict_dissection/freeze-v1-f722885fd8226009/rolling-h32b32-h7168b1024-v1/%x-%j.out
set -euo pipefail

SCORER=${1:?usage: sbatch multiscale.sbatch SCORER}
BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
FREEZE=$BASE/code/verdict_dissection/freeze-v1-f722885fd8226009
RUN=$FREEZE/rolling-h32b32-h7168b1024-v1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
$BASE/code/judge_interp/venv/bin/python \
  $RUN/repo/scripts/verdict_dissection/llm_multiscale.py \
  $FREEZE/transcript_store.parquet \
  $FREEZE/rolling_surprisal_h32b32_h7168b1024_v1 \
  "$SCORER" \
  --prefix-cache $FREEZE/surprisal_cache_content
