#!/usr/bin/env bash
set -euo pipefail
matrix_root=${1:?usage: submit_lesion_specs.sh MATRIX_ROOT RESULTS_DIR raw|pc l1|ridge [PC_BASIS_ROOT] [PC_COMPONENTS]}
results_dir=${2:?missing RESULTS_DIR}
feature_space=${3:?missing raw|pc}
penalty=${4:?missing l1|ridge}
pc_basis_root=${5:-}
pc_components=${6:-}
repo=${JV_CODE_DIR:?set JV_CODE_DIR}
if [[ $feature_space == pc ]]; then
  [[ $pc_components =~ ^[1-9][0-9]*$ ]] || { echo "PC_COMPONENTS must be explicit and positive" >&2; exit 2; }
elif [[ -n $pc_basis_root || -n $pc_components ]]; then
  echo "PC arguments require feature space pc" >&2
  exit 2
fi
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
  "$matrix_root" "$results_dir" "$feature_space" "$penalty" \
  "$pc_basis_root" "$pc_components"
