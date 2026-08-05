#!/bin/bash
#SBATCH --job-name=record-acts
#SBATCH --partition=a100,gpu_lowp
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Two GPUs suffice for the hook-based recorder: peak memory stays near
# the bf16 weights (~66 GB; A100s here are 40 GB, every gpu_lowp card
# is >= 48 GB; device_map=auto shards). Submit from the directory holding
# record_activations.py + config.yaml, with a venv set up once via:
#   python3 -m venv venv && venv/bin/pip install -r requirements-hpc.txt
# Resubmitting after a timeout or preemption is safe: recorded entities
# are skipped.

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv/bin/activate
python record_activations.py
