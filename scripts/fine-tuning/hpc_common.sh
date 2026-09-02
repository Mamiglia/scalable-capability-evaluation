# Shared paths/env for the judge_ft sbatch jobs. Sourced from the job's
# working dir (code/judge_ft on ceph).
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# no CUDA toolkit (nvcc) on compute nodes: flashinfer's JIT sampler cannot build
export VLLM_USE_FLASHINFER_SAMPLER=0
C=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
IN=$C/staging/judge_ft
# per-model runs: sbatch --export=ALL,JFT_MODEL=<dir under models/>,JFT_OUT=<subdir>
OUT=$C/staging/judge_ft/${JFT_OUT:-out}
MODEL=$C/models/${JFT_MODEL:-Qwen3-8B}
mkdir -p "$OUT"
source venv/bin/activate
