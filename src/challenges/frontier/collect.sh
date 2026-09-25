#!/usr/bin/env bash
# Collect the frontier subsets (terminal_bench_2_1 then swe_bench_mini) for each model, sequentially:
# the podman host cannot take two container-heavy evals at once.
# No --max-subprocesses 1: these images are pulled, not built, and the flag serialises every sandbox
# exec across the 32 samples.
#
#   setsid -f bash -c 'src/challenges/frontier/collect.sh configs/models/openai/gpt-5.6-sol.yml \
#       configs/models/anthropic/claude-opus-4.8.yml > artifacts/frontier/collect.log 2>&1'
#
# Resumable: a repeat dir with a successful log is skipped by `collect`.
set -uo pipefail
cd "$(dirname "$0")/../../.."
[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/sce-uv-cache
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"

for model in "$@"; do
  echo "=== $(date -Is) terminal_bench_2_1 $model"
  uv run collect --jobs 1 --max-sandboxes "${MAX_SANDBOXES:-32}" --message-limit 200 \
    --max-tokens 32768 --fail-on-error 0.1 -b configs/bench/terminal_bench_2_1_frontier.yml -m "$model" -n 1 --log-dir data
  echo "=== $(date -Is) swe_bench_mini $model"
  uv run --with 'swebench>=3.0.15,<4' collect --jobs 1 --max-sandboxes 8 --message-limit 100 \
    --max-tokens 32768 --fail-on-error 0.2 -b configs/bench/swe-bench-mini-frontier.yml -m "$model" -n 1 --log-dir data
done
echo "=== $(date -Is) all done"
