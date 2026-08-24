#!/usr/bin/env bash
# Collect the 26 models used by configs/experiments/pairs/*.yml on gpqa, AIME, APEX
# and HLE. Resumable: a re-run only fills the gaps.
#
#   scripts/collect_pairs.sh                       # everything, 2 repeats
#   BENCHES=hle scripts/collect_pairs.sh           # one bench
#   EPOCHS=1 MODELS_FILE=one.txt scripts/collect_pairs.sh --limit 2   # smoke run

set -euo pipefail
cd "$(dirname "$0")/.."

[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache

read -r -a names <<< "${BENCHES:-gpqa aime24 aime25 aime26 apex_shortlist hle}"
benches=("${names[@]/#/configs/bench/}")
benches=("${benches[@]/%/.yml}")
mapfile -t models < "${MODELS_FILE:-configs/experiments/pairs/models.txt}"

# tolerate a few unrecoverable samples: without this inspect fails the whole task on the
# first sample error, throwing away thousands of good generations for one bad question
uv run collect \
  --fail-on-error "${FAIL_ON_ERROR:-3}" \
  -b "${benches[@]}" \
  -m "${models[@]}" \
  -n "${EPOCHS:-2}" \
  --log-dir "${LOG_DIR:-data}" \
  "$@"
