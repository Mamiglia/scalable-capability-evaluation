#!/usr/bin/env bash
#SBATCH --job-name=workstream-c
#SBATCH --partition=a100,gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=run_logs/workstream-c-%x-%j.out

set -euo pipefail

stage=${1:?usage: sbatch workstream_c.sbatch nodes|attachments|edge-views|circuits}
: "${C_CODE_DIR:?set C_CODE_DIR to the staged repository}"
: "${C_VENV:?set C_VENV to the compute-node virtualenv}"
: "${C_FREEZE:?set C_FREEZE to the frozen Phase 1 manifest}"
: "${C_STORE:?set C_STORE to the frozen transcript store}"
: "${C_SPLIT:?set C_SPLIT to the frozen item split}"
: "${C_RECONCILIATION:?set C_RECONCILIATION to the hashed verdict reconciliation}"
: "${C_APPROVAL:?set C_APPROVAL to the immutable design/annotator/budget approval}"
: "${C_CONFIG:?set C_CONFIG to configs/workstream_c.yaml}"
: "${C_OUTPUT_ROOT:?set C_OUTPUT_ROOT to a new hash-versioned Ceph directory}"
: "${C_STAGE_INPUT:?set C_STAGE_INPUT to the immutable inputs for this stage}"
: "${C_CODE_COMMIT:?set C_CODE_COMMIT to the staged code commit}"
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Workstream C canonical stages are Slurm-only'; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo 'refusing gateway execution'; exit 2; }
[[ -f "$C_CONFIG" ]] || { echo 'missing Workstream C config'; exit 2; }
[[ -f "$C_OUTPUT_ROOT/$stage/artifact_manifest.json" ]] && exit 0

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$C_CODE_DIR"

case "$stage" in
  nodes|attachments|edge-views|circuits)
    command=("$C_VENV/bin/python" -m scripts.discourse_network.stages "$stage" \
      --freeze "$C_FREEZE" --store "$C_STORE" --split "$C_SPLIT" \
      --judge-util src/judge/util.py --reconciliation "$C_RECONCILIATION" \
      --approval "$C_APPROVAL" --config "$C_CONFIG" \
      --input "$C_STAGE_INPUT" --output "$C_OUTPUT_ROOT/$stage" \
      --code-commit "$C_CODE_COMMIT")
    [[ ${C_CONFIRMATION_APPLY:-NO} == YES ]] && command+=(--confirmation-apply --apply-from "${C_APPLY_FROM:?set C_APPLY_FROM for confirmation}")
    [[ ${C_CONFIRMATION_APPLY:-NO} == YES ]] && command+=(--apply-from-manifest-sha256 "${C_APPLY_FROM_MANIFEST_SHA256:?set C_APPLY_FROM_MANIFEST_SHA256}")
    [[ -n ${C_VERDICTS:-} ]] && command+=(--verdicts "$C_VERDICTS")
    [[ -n ${C_ADAPTER_INPUT:-} ]] && command+=(--adapter-input "$C_ADAPTER_INPUT")
    [[ $stage != nodes ]] && command+=(--parent-manifest "${C_PARENT_MANIFEST:?set C_PARENT_MANIFEST}" --parent-manifest-sha256 "${C_PARENT_MANIFEST_SHA256:?set C_PARENT_MANIFEST_SHA256}")
    "${command[@]}"
    ;;
  *)
    echo "unknown Workstream C stage: $stage" >&2
    exit 2
    ;;
esac

echo "$stage complete"
