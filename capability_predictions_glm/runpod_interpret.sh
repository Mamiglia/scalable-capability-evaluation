#!/usr/bin/env bash
set -euo pipefail

root=/workspace/capability
cd "$root/repo"
export PYTHONPATH="$root/repo:$root/repo/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$root/results/interpret" "$root/logs/interpret"

/workspace/.venv/bin/python -m capability_predictions_glm.interpret \
  --input-root "$root/inputs" --full-results "$root/full-results" --list \
  > "$root/specifications.txt"

run_one() {
  spec=$1
  output="$root/results/interpret/$spec"
  if [[ -f "$output/manifest.json" ]]; then
    echo "already done: $spec"
    return
  fi
  echo "start: $spec"
  /workspace/.venv/bin/python -m capability_predictions_glm.interpret \
    --input-root "$root/inputs" --full-results "$root/full-results" \
    --specification "$spec" --output "$output" \
    > "$root/logs/interpret/$spec.log" 2>&1
  echo "complete: $spec"
}
export -f run_one
export root PYTHONPATH OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS

xargs -P 6 -n 1 bash -c 'run_one "$1"' _ < "$root/specifications.txt"
touch "$root/results/interpret/DONE"
