#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache
export PYTHONUNBUFFERED=1

mapfile -t models < <(grep -rl '^  use: true$' configs/models --include='*.yml' | sort)
benchmarks=(configs/bench/{gpqa,aime24,aime25,aime26,mbpp}.yml)
PARALLEL_JOBS="${MAX_CONCURRENCY:-3}"

run_model() {
  local model="$1"
  local benchmarks=(configs/bench/{gpqa,aime24,aime25,aime26,mbpp}.yml)
  for benchmark in "${benchmarks[@]}"; do
    DOCKER_HOST=unix:///var/run/docker.sock uv run collect -b "$benchmark" -m "$model" -n 1 --max-tokens 4096 --log-dir data || true
    DOCKER_HOST=unix:///var/run/docker.sock uv run collect -b "$benchmark" -m "$model" -n 2 --max-tokens 4096 --log-dir data || true
  done
}

export -f run_model

printf '%s\n' "${models[@]}" | xargs -n 1 -P "$PARALLEL_JOBS" bash -c 'run_model "$@"' _


