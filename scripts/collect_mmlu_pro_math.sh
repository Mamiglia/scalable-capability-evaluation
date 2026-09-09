#!/usr/bin/env bash
# MMLU-Pro [math], seeded 150-item subsample, for the cohort in
# configs/experiments/mmlu_pro_models.txt. Hosted models go in one invocation; Runpod
# models get one invocation each, MAX_CONCURRENCY at a time, so each spins its own pod.
#   REPEATS=2 bash scripts/collect_mmlu_pro_math.sh          # more repeats
#   bash scripts/collect_mmlu_pro_math.sh --limit 2 --log-dir /tmp/smoke   # validate

set -euo pipefail
cd "$(dirname "$0")/.."

[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache
export PYTHONUNBUFFERED=1

BENCH=configs/bench/mmlu_pro_math.yml
REPEATS="${REPEATS:-1}"
LIST=configs/experiments/mmlu_pro_models.txt

mapfile -t models < <(sed 's/#.*//' "$LIST" | tr -d ' ' | grep -v '^$')
hosted=(); pods=()
for model in "${models[@]}"; do
  if grep -q '^  use: true$' "$model"; then pods+=("$model"); else hosted+=("$model"); fi
done
echo "hosted: ${#hosted[@]}  runpod: ${#pods[@]}  repeats: $REPEATS"

((${#hosted[@]} == 0)) || uv run collect -b "$BENCH" -m "${hosted[@]}" -n "$REPEATS" --log-dir data "$@"

# Bash cannot export an array, so the pass-through flags cross into xargs as a string.
# They are plain flags (--limit 2, --log-dir /tmp/smoke); nothing here contains spaces.
run_model() {
  # shellcheck disable=SC2086
  uv run collect -b "$BENCH" -m "$1" -n "$REPEATS" --log-dir data $EXTRA || true
}
export -f run_model
export BENCH REPEATS
EXTRA="$*"; export EXTRA
((${#pods[@]} == 0)) || printf '%s\n' "${pods[@]}" |
  xargs -n 1 -P "${MAX_CONCURRENCY:-5}" bash -c 'run_model "$@"' _

echo "--- pods still alive (delete any not backing a job) ---"
runpodctl pod list 2>/dev/null || true
