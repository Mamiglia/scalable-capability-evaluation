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
: "${B_BGE_MODEL:?set B_BGE_MODEL to the staged immutable BGE revision}"
[[ ${B_DESIGN_APPROVED:-} == YES ]] || { echo 'B_DESIGN_APPROVED must be YES'; exit 2; }
[[ ${B_PHASE1_FROZEN:-} == YES ]] || { echo 'B_PHASE1_FROZEN must be YES'; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'GPU stages must run under Slurm'; exit 2; }
[[ $(hostname) != gateway* && $(hostname) != sgw* ]] || { echo 'refusing gateway execution'; exit 2; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$B_CODE_DIR"

embed=("$B_VENV/bin/python" scripts/verdict_dissection/transcript_embeddings.py
       --store "$B_STORE" --split "$B_SPLIT" --freeze "$B_FREEZE")
sae=("$B_VENV/bin/python" scripts/verdict_dissection/transcript_sae.py)

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

case "$stage" in
  qwen-probe-1|qwen-probe-2)
    out="$B_OUTPUT_ROOT/dense-$stage"
    run_artifact "$out" "${embed[@]}" --family qwen --model-path "$B_QWEN_MODEL" --probe --output "$out"
    ;;
  bge-probe-1|bge-probe-2)
    out="$B_OUTPUT_ROOT/dense-$stage"
    keys="$B_OUTPUT_ROOT/dense-qwen-probe-1/rows.parquet"
    artifact_ok "$B_OUTPUT_ROOT/dense-qwen-probe-1" || { echo 'run qwen-probe-1 first'; exit 2; }
    run_artifact "$out" "${embed[@]}" --family bge --model-path "$B_BGE_MODEL" --probe-keys "$keys" --output "$out"
    ;;
  qwen-overlap-half|qwen-overlap-double)
    overlap=512
    [[ $stage == qwen-overlap-double ]] && overlap=2048
    out="$B_OUTPUT_ROOT/dense-$stage"
    keys="$B_OUTPUT_ROOT/dense-qwen-probe-1/rows.parquet"
    artifact_ok "$B_OUTPUT_ROOT/dense-qwen-probe-1" || { echo 'run qwen-probe-1 first'; exit 2; }
    run_artifact "$out" "${embed[@]}" --family qwen --model-path "$B_QWEN_MODEL" --selected-keys "$keys" --overlap "$overlap" --output "$out"
    ;;
  bge-overlap-half|bge-overlap-double)
    overlap=256
    [[ $stage == bge-overlap-double ]] && overlap=1024
    out="$B_OUTPUT_ROOT/dense-$stage"
    keys="$B_OUTPUT_ROOT/dense-qwen-probe-1/rows.parquet"
    artifact_ok "$B_OUTPUT_ROOT/dense-qwen-probe-1" || { echo 'run qwen-probe-1 first'; exit 2; }
    run_artifact "$out" "${embed[@]}" --family bge --model-path "$B_BGE_MODEL" --selected-keys "$keys" --overlap "$overlap" --output "$out"
    ;;
  qwen-dense-gate|bge-dense-gate)
    family=${stage%%-*}
    out="$B_OUTPUT_ROOT/$stage"
    run_artifact "$out" "${embed[@]}" --family "$family" --gate-from \
      "$B_OUTPUT_ROOT/dense-$family-probe-1" "$B_OUTPUT_ROOT/dense-$family-probe-2" \
      "$B_OUTPUT_ROOT/dense-$family-overlap-half" "$B_OUTPUT_ROOT/dense-$family-overlap-double" \
      --output "$out"
    ;;
  combined-dense-gate)
    out="$B_OUTPUT_ROOT/combined-dense-gate"
    run_artifact "$out" "${embed[@]}" --family qwen --combine-gates \
      "$B_OUTPUT_ROOT/qwen-dense-gate" "$B_OUTPUT_ROOT/bge-dense-gate" --output "$out"
    ;;
  qwen-extract)
    out="$B_OUTPUT_ROOT/dense-qwen-development"
    run_artifact "$out" "${embed[@]}" --family qwen --model-path "$B_QWEN_MODEL" --combined-dense-gate "$B_OUTPUT_ROOT/combined-dense-gate" --output "$out"
    ;;
  bge-extract)
    out="$B_OUTPUT_ROOT/dense-bge-development"
    run_artifact "$out" "${embed[@]}" --family bge --model-path "$B_BGE_MODEL" --combined-dense-gate "$B_OUTPUT_ROOT/combined-dense-gate" --output "$out"
    ;;
  qwen-sae)
    dense="$B_OUTPUT_ROOT/dense-qwen-development"
    out="$B_OUTPUT_ROOT/sae-qwen-development"
    artifact_ok "$dense" || { echo 'run qwen-extract first'; exit 2; }
    run_artifact "$out" "${sae[@]}" --dense-dir "$dense" --output "$out"
    ;;
  bge-sae)
    dense="$B_OUTPUT_ROOT/dense-bge-development"
    out="$B_OUTPUT_ROOT/sae-bge-development"
    artifact_ok "$dense" || { echo 'run bge-extract first'; exit 2; }
    run_artifact "$out" "${sae[@]}" --dense-dir "$dense" --output "$out"
    ;;
  development-report)
    : "${B_A1:?set B_A1 to the B-safe A1 transcript-key table}"
    : "${B_A2_LLAMA:?set B_A2_LLAMA}"
    : "${B_A2_QWEN:?set B_A2_QWEN}"
    : "${B_A2_GEMMA:?set B_A2_GEMMA}"
    out="$B_OUTPUT_ROOT/development-report"
    run_artifact "$out" "${sae[@]}" --development-report \
      --dense-dir "$B_OUTPUT_ROOT/dense-qwen-development" \
      --sae-dir "$B_OUTPUT_ROOT/sae-qwen-development" \
      --bge-sae-dir "$B_OUTPUT_ROOT/sae-bge-development" --a1 "$B_A1" \
      --a2 "llama=$B_A2_LLAMA" --a2 "qwen=$B_A2_QWEN" --a2 "gemma=$B_A2_GEMMA" \
      --output "$out"
    ;;
  qwen-confirmation-extract)
    out="$B_OUTPUT_ROOT/dense-qwen-confirmation"
    run_artifact "$out" "${embed[@]}" --family qwen --model-path "$B_QWEN_MODEL" --roles confirmation --confirmation --combined-dense-gate "$B_OUTPUT_ROOT/combined-dense-gate" --output "$out"
    ;;
  bge-confirmation-extract)
    out="$B_OUTPUT_ROOT/dense-bge-confirmation"
    run_artifact "$out" "${embed[@]}" --family bge --model-path "$B_BGE_MODEL" --roles confirmation --confirmation --combined-dense-gate "$B_OUTPUT_ROOT/combined-dense-gate" --output "$out"
    ;;
  qwen-confirmation-apply)
    dense="$B_OUTPUT_ROOT/dense-qwen-confirmation"
    frozen="$B_OUTPUT_ROOT/sae-qwen-development"
    out="$B_OUTPUT_ROOT/sae-qwen-confirmation"
    artifact_ok "$dense" && artifact_ok "$frozen" || { echo 'confirmation dense vectors and frozen Qwen SAE are required'; exit 2; }
    run_artifact "$out" "${sae[@]}" --dense-dir "$dense" --apply-from "$frozen" --report-from "$B_OUTPUT_ROOT/development-report" --output "$out"
    ;;
  bge-confirmation-apply)
    dense="$B_OUTPUT_ROOT/dense-bge-confirmation"
    frozen="$B_OUTPUT_ROOT/sae-bge-development"
    out="$B_OUTPUT_ROOT/sae-bge-confirmation"
    artifact_ok "$dense" && artifact_ok "$frozen" || { echo 'confirmation dense vectors and frozen BGE SAE are required'; exit 2; }
    run_artifact "$out" "${sae[@]}" --dense-dir "$dense" --apply-from "$frozen" --report-from "$B_OUTPUT_ROOT/development-report" --output "$out"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
