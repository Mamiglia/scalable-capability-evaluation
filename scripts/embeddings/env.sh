#!/bin/bash
# Build /home/jyeon/.venvs/smalljudge-embed: the evaluation stack (inspect-ai, draccus, pandas,
# scikit-learn) plus torch/transformers for Qwen3-Embedding-8B. The frozen venvs
# (smalljudge-glm313, smalljudge-judge) are not touched. Runs as an HTCondor CPU job: the
# download size is unknown and uv's cache needs node-local flock (Condor scratch).
set -euo pipefail
export UV_CACHE_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/uv"
export UV_LINK_MODE=copy
export PATH=/home/jyeon/.local/bin:$PATH
VENV=/home/jyeon/.venvs/smalljudge-embed
REPO=/home/jyeon/LASR/small-judge

echo "=== python 3.13 venv ==="
UV_VENV_CLEAR=1 uv venv --python 3.13 "$VENV" 2>&1 | tail -2

echo "=== install ==="
VIRTUAL_ENV="$VENV" uv pip install \
    "inspect-ai==0.3.255" "draccus>=0.11.6" python-dotenv pyyaml "pandas>=3.0.5" numpy \
    "scikit-learn>=1.7" scipy pyarrow pytest pytest-asyncio zipfile-zstd \
    torch transformers "huggingface_hub[hf_transfer]" "numpyro>=0.15" jax 2>&1 | tail -8

echo "=== versions ==="
"$VENV/bin/python" -c "import torch, transformers, huggingface_hub, inspect_ai, sklearn, pandas; print('torch', torch.__version__, '| transformers', transformers.__version__, '| hub', huggingface_hub.__version__, '| inspect_ai', inspect_ai.__version__, '| sklearn', sklearn.__version__, '| pandas', pandas.__version__)"

echo "=== evaluation import path ==="
cd "$REPO"
PYTHONPATH=src "$VENV/bin/python" -c "import capability.evaluate, protocols, evals.transcripts; print('imports OK')"

echo "=== hugging face auth (username only) and repo sizes ==="
PYTHONPATH=scripts/judging_behavior "$VENV/bin/python" - <<'PY'
from hf_download import secret
from huggingface_hub import HfApi
api = HfApi(token=secret("HF_TOKEN"))
print("user", api.whoami()["name"])
for repo in ("LASR-G5/benchmarks", "LASR-G5/transcript-embeddings"):
    info = api.dataset_info(repo, files_metadata=True)
    print(repo, "sha", info.sha, "files", len(info.siblings), "GB", round(sum((s.size or 0) for s in info.siblings) / 1e9, 2))
PY

echo "=== unit tests ==="
"$VENV/bin/python" -m pytest -q tests/test_protocols.py
