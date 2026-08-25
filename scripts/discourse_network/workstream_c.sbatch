#!/usr/bin/env bash
#SBATCH --job-name=clause-network
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=run_logs/%x-%j.out

set -euo pipefail

stage=${1:?usage: sbatch workstream_c.sbatch probe|segment|nodes|embeddings|edges|features|finish}
: "${C_CODE_DIR:?set C_CODE_DIR to the staged repository}"
: "${C_PYTHON:?set C_PYTHON to the Linux compute-node Python}"
: "${C_PACKAGES:?set C_PACKAGES to the staged Python packages}"
: "${C_HF_HOME:?set C_HF_HOME to the staged Hugging Face cache}"
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'clause-network stages are Slurm-only'; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo 'refusing gateway execution'; exit 2; }

export PYTHONPATH="$C_PACKAGES:$C_CODE_DIR/scripts"
export HF_HOME="$C_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$C_CODE_DIR"
config="${SLURM_TMPDIR:-/tmp}/workstream_c_${SLURM_JOB_ID}.yaml"
sed 's/device: mps/device: cuda/' configs/workstream_c.yaml > "$config"

case "$stage" in
  probe)
    "$C_PYTHON" -c 'import scipy, sklearn, sentence_transformers; print("analysis imports ok")'
    "$C_PYTHON" -c 'from isanlp_rst.parser import Parser; p=Parser(hf_model_name="tchewik/isanlp_rst_v3", hf_model_version="gumrrg", cuda_device=-1); print(p("One clause, because another follows.")["rst"][0])'
    "$C_PYTHON" -c 'from sentence_transformers import SentenceTransformer; m=SentenceTransformer("Alibaba-NLP/gte-modernbert-base", revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22", device="cpu"); print(m.encode(["probe"]).shape)'
    ;;
  segment)
    pids=()
    for shard in {0..5}; do
      "$C_PYTHON" scripts/discourse_network/segment.py --config "$config" --shard "$shard" --n-shards 6 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    (( status == 0 )) || exit "$status"
    ;;
  nodes|embeddings|edges|features)
    "$C_PYTHON" scripts/discourse_network/build.py --config "$config" --stop-after "$stage"
    ;;
  finish)
    "$C_PYTHON" scripts/discourse_network/build.py --config "$config"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac

echo "$stage complete"
