#!/usr/bin/env bash
#SBATCH --job-name=workstream-b
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=run_logs/workstream-b-%x-%j.out

set -euo pipefail

stage=${1:?usage: sbatch workstream_b.sbatch STAGE}
: "${B_CODE_DIR:?set B_CODE_DIR to the staged repository}"
: "${B_VENV:?set B_VENV to the compute-node virtualenv}"
: "${B_STORE:?set B_STORE to the frozen transcript store}"
: "${B_SPLIT:?set B_SPLIT to the frozen item split}"
: "${B_FREEZE:?set B_FREEZE to the Phase 1 manifest}"
: "${B_OUTPUT_ROOT:?set B_OUTPUT_ROOT to a new Workstream B output directory}"
: "${B_QWEN_MODEL:?set B_QWEN_MODEL to the staged immutable Qwen revision}"
: "${B_BGE_MODEL:=}"
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'GPU stages must run under Slurm'; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo 'refusing gateway execution'; exit 2; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$B_CODE_DIR"

# every stage sees all canonical-correct transcripts (all three split roles)
embed=("$B_VENV/bin/python" scripts/verdict_dissection/transcript_embeddings.py
       --store "$B_STORE" --split "$B_SPLIT" --freeze "$B_FREEZE"
       --roles representation_train representation_validation confirmation)

artifact_ok() {
  "$B_VENV/bin/python" -c \
    'import json,sys; sys.path.insert(0,"scripts/verdict_dissection"); from transcript_embeddings import verify_artifact_files; p=__import__("pathlib").Path(sys.argv[1]); verify_artifact_files(p,json.loads((p/"manifest.json").read_text()))' \
    "$1"
}

run_artifact() {
  local out=$1
  shift
  if [[ -e "$out" ]]; then
    artifact_ok "$out" || { echo "invalid existing artifact: $out" >&2; return 2; }
  else
    "$@"
  fi
}

family=${stage%%-*}
case "$family" in
  qwen) model=$B_QWEN_MODEL ;;
  bge) model=$B_BGE_MODEL ;;
  *) model= ;;
esac
probe_keys="$B_OUTPUT_ROOT/dense-qwen-probe-1/rows.parquet"

case "$stage" in
  qwen-probe-1|qwen-probe-2)
    out="$B_OUTPUT_ROOT/dense-$stage"
    run_artifact "$out" "${embed[@]}" --family qwen --model-path "$model" --probe --output "$out"
    ;;
  bge-probe-1|bge-probe-2)
    out="$B_OUTPUT_ROOT/dense-$stage"
    artifact_ok "$B_OUTPUT_ROOT/dense-qwen-probe-1" || { echo 'run qwen-probe-1 first'; exit 2; }
    run_artifact "$out" "${embed[@]}" --family bge --model-path "$model" --probe-keys "$probe_keys" --output "$out"
    ;;
  qwen-overlap-half|qwen-overlap-double|bge-overlap-half|bge-overlap-double)
    # overlap sensitivities are the family's predeclared half/double values
    case "$stage" in
      qwen-overlap-half) overlap=512 ;;
      qwen-overlap-double) overlap=2048 ;;
      bge-overlap-half) overlap=256 ;;
      bge-overlap-double) overlap=1024 ;;
    esac
    out="$B_OUTPUT_ROOT/dense-$stage"
    artifact_ok "$B_OUTPUT_ROOT/dense-qwen-probe-1" || { echo 'run qwen-probe-1 first'; exit 2; }
    run_artifact "$out" "${embed[@]}" --family "$family" --model-path "$model" --selected-keys "$probe_keys" --overlap "$overlap" --output "$out"
    ;;
  qwen-dense-gate|bge-dense-gate)
    out="$B_OUTPUT_ROOT/$stage"
    run_artifact "$out" "${embed[@]}" --family "$family" --gate-from \
      "$B_OUTPUT_ROOT/dense-$family-probe-1" "$B_OUTPUT_ROOT/dense-$family-probe-2" \
      "$B_OUTPUT_ROOT/dense-$family-overlap-half" "$B_OUTPUT_ROOT/dense-$family-overlap-double" \
      --output "$out"
    ;;
  qwen-extract|bge-extract)
    out="$B_OUTPUT_ROOT/dense-$family"
    run_artifact "$out" "${embed[@]}" --family "$family" --model-path "$model" \
      --dense-gate "$B_OUTPUT_ROOT/$family-dense-gate" --output "$out"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
