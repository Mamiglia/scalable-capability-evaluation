# Shared paths/env for the whitebox sbatch jobs. Sourced from the job's
# working dir (judge-whitebox on ceph).
set -euo pipefail
export WB_HPC=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# no CUDA toolkit (nvcc) on compute nodes: flashinfer's JIT sampler cannot build
export VLLM_USE_FLASHINFER_SAMPLER=0
C=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
OUT=$C/staging/whitebox
MODEL=$C/models/${WB_MODEL:-Qwen3-8B}
mkdir -p "$OUT"
source $C/code/judge_ft/venv/bin/activate
