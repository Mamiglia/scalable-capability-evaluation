#!/bin/bash
set -euo pipefail

stage=${1:?usage: sae.sh check|embed|features}
repo=${_CONDOR_JOB_IWD:?HTCondor did not set _CONDOR_JOB_IWD}
venv=${SMALL_JUDGE_GLM_VENV:-/home/jyeon/.venvs/smalljudge-glm}
cd "$repo"

if [[ "$stage" != embed ]]; then
  exec "$venv/bin/python" scripts/hibayes/sae_judge.py --stage "$stage"
fi

job_cache=${_CONDOR_SCRATCH_DIR:?}/small-judge-sae
export HF_HOME="$job_cache/huggingface"
export XDG_CACHE_HOME="$job_cache/xdg"
mkdir -p "$HF_HOME/hub" "$XDG_CACHE_HOME"

# The shared cache is optional. A missing pinned snapshot is downloaded into
# Condor scratch for this job only.
shared_cache=${SMALL_JUDGE_HF_CACHE:-/fast/jyeon/hf_cache}
snapshot=models--Qwen--Qwen3-Embedding-8B
revision=1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
if [[ -d "$shared_cache/hub/$snapshot/snapshots/$revision" ]]; then
  cp -r "$shared_cache/hub/$snapshot" "$HF_HOME/hub/"
  echo "seeded the job cache from $shared_cache"
fi

"$venv/bin/python" scripts/hibayes/sae_judge.py --stage embed --batch-size 8
