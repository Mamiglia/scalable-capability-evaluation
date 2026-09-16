#!/bin/bash
# GPU runner for scripts/embeddings/embed.py: the Qwen3-Embedding-8B snapshot comes from the shared
# read-only HF cache on /fast (offline, no token needed); arguments pass through to embed.py.
set -euo pipefail
cd /home/jyeon/LASR/small-judge
export HF_HOME=/fast/jyeon/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "no usable GPU on $(hostname)"; exit 6; }
echo "data -> $(readlink -f data)"
exec /home/jyeon/.venvs/smalljudge-embed/bin/python scripts/embeddings/embed.py "$@"
