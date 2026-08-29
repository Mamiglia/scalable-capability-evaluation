#!/usr/bin/env bash
set -euo pipefail

root=/workspace/capability
result_set=${RESULT_SET:-full}
cd "$root/repo"
export PYTHONPATH="$root/repo:$root/repo/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$root/results/$result_set" "$root/logs/$result_set"
pids=()
names=()
for fold in model-fold-{0..3}; do
  for target in Q2 Q3 Q4; do
    name="$fold-$target"
    output="$root/results/$result_set/$name"
    if [[ -f "$output/manifest.json" ]]; then
      echo "already done: $name"
      continue
    fi
    /workspace/.venv/bin/python -m capability_predictions_glm \
      --input-root "$root/inputs" --output-root "$output" \
      --execute --full-only --fold "$fold" --target "$target" \
      > "$root/logs/$result_set/$name.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
  done
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "complete: ${names[$index]}"
  else
    echo "failed: ${names[$index]}" >&2
    failed=1
  fi
done
[[ $failed == 0 ]] || exit 1
touch "$root/results/$result_set/DONE"
