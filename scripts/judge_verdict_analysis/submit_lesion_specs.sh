#!/usr/bin/env bash
set -euo pipefail
matrix_root=${1:?usage: submit_lesion_specs.sh MATRIX_ROOT RESULTS_DIR raw|pc l1|ridge}
results_dir=${2:?missing RESULTS_DIR}
feature_space=${3:?missing raw|pc}
penalty=${4:?missing l1|ridge}
repo=${JV_CODE_DIR:?set JV_CODE_DIR}
fold_root="$results_dir/lesions/$feature_space-$penalty/folds"
first=$(find "$fold_root" -name run-specifications.csv -type f | sort | head -1)
[[ -n $first ]] || { echo "no prepared lesion folds" >&2; exit 2; }
per_fold=$(($(wc -l < "$first") - 1))
folds=$(find "$fold_root" -name run-specifications.csv -type f | wc -l | tr -d ' ')
((folds > 0)) || { echo "no prepared lesion folds" >&2; exit 2; }
while IFS= read -r table; do
  [[ $(($(wc -l < "$table") - 1)) == "$per_fold" ]] || {
    echo "prepared folds have different specification counts" >&2
    exit 2
  }
done < <(find "$fold_root" -name run-specifications.csv -type f | sort)
tasks=$((folds * per_fold))
((tasks > 0)) || { echo "no lesion specifications" >&2; exit 2; }
sbatch --parsable --array="0-$((tasks - 1))%128" \
  --export=ALL,JV_CODE_DIR,JV_PYTHON,JV_PACKAGES \
  "$repo/scripts/judge_verdict_analysis/lesion_spec.sbatch" \
  "$matrix_root" "$results_dir" "$feature_space" "$penalty"
