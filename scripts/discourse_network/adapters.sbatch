#!/usr/bin/env bash
#SBATCH --job-name=workstream-c-adapter
#SBATCH --partition=a100,gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=run_logs/workstream-c-adapter-%x-%j.out

set -euo pipefail

kind=${1:?usage: sbatch adapters.sbatch segment|embed}
: "${C_CODE_DIR:?set C_CODE_DIR to the staged repository}"
: "${C_ADAPTER_VENV:?set C_ADAPTER_VENV to the pinned compute-node environment}"
: "${C_ADAPTER_INPUT:?set C_ADAPTER_INPUT to the immutable input parquet}"
: "${C_ADAPTER_OUTPUT:?set C_ADAPTER_OUTPUT to a new adapter artifact directory}"
: "${C_ADAPTER_ASSETS:?set C_ADAPTER_ASSETS to staged offline assets}"
: "${C_ADAPTER_REVISION:?set C_ADAPTER_REVISION to the pinned revision}"
: "${C_CONFIG_HASH:?set C_CONFIG_HASH to the frozen C config SHA-256}"
: "${C_FREEZE_SHA256:?set C_FREEZE_SHA256 to the computed A manifest SHA-256}"
: "${C_CONTENT_HASH:?set C_CONTENT_HASH to the canonical content hash}"
: "${C_APPROVAL:?set C_APPROVAL to the frozen approval artifact}"
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'canonical adapters are Slurm-only'; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo 'refusing gateway execution'; exit 2; }
[[ ! -e "$C_ADAPTER_OUTPUT" ]] || {
  echo 'refusing existing adapter output'; exit 2;
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
cd "$C_CODE_DIR"
command=("$C_ADAPTER_VENV/bin/python" -m scripts.discourse_network.adapters "$kind"
  --input "$C_ADAPTER_INPUT" --output "$C_ADAPTER_OUTPUT" --assets "$C_ADAPTER_ASSETS"
  --revision "$C_ADAPTER_REVISION" --config-hash "$C_CONFIG_HASH"
  --freeze-manifest-sha256 "$C_FREEZE_SHA256" --canonical-content-hash "$C_CONTENT_HASH"
  --approval "$C_APPROVAL")
[[ $kind == segment ]] && command+=(--gate-metrics "${C_GATE_METRICS:?set C_GATE_METRICS}")
[[ $kind == segment ]] && command+=(--system "${C_SEGMENTER_SYSTEM:-isanlp_rst}")
[[ $kind == segment ]] && command+=(--freeze-manifest "${C_FREEZE:?set C_FREEZE}" \
  --split "${C_SPLIT:?set C_SPLIT}" --judge-util "${C_JUDGE_UTIL:?set C_JUDGE_UTIL}" \
  --reconciliation "${C_RECONCILIATION:?set C_RECONCILIATION}" --c-config "${C_CONFIG:?set C_CONFIG}")
[[ $kind == embed ]] && command+=(--nodes-manifest-sha256 "${C_NODES_MANIFEST_SHA256:?set C_NODES_MANIFEST_SHA256}")
[[ $kind == embed ]] && command+=(--nodes-manifest "${C_NODES_MANIFEST:?set C_NODES_MANIFEST}" --c-code-hash "${C_CODE_HASH:?set C_CODE_HASH}")
[[ ${C_CONFIRMATION_APPLY:-NO} == YES ]] && command+=(--confirmation-apply)
"${command[@]}"
