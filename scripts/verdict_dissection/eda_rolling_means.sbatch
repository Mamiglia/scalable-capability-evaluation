#!/bin/bash
#SBATCH --job-name=rolling-means-eda
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/code/verdict_dissection/freeze-v1-f722885fd8226009/rolling-h32b32-h7168b1024-v1/%x-%j.out
set -euo pipefail

BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
FREEZE=$BASE/code/verdict_dissection/freeze-v1-f722885fd8226009
RUN=$FREEZE/rolling-h32b32-h7168b1024-v1
DATA=$FREEZE/dataframes-rolling-h32b32-h7168b1024-v1

$BASE/code/judge_interp/venv/bin/python \
  $RUN/export-repo/scripts/verdict_dissection/eda_rolling_means.py \
  $DATA/raw/token-predictability-rolling \
  $DATA/raw/transcript-features.parquet \
  $DATA/normalised/rolling-position
