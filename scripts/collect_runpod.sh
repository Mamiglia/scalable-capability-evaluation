#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/sce-uv-cache
export PYTHONUNBUFFERED=1

mapfile -t models < <(grep -rl '^  use: true$' configs/models --include='*.yml' | sort)
PARALLEL_JOBS="${MAX_CONCURRENCY:-5}"

run_model() {
  local model="$1"
  local benchmarks=(configs/bench/{gpqa,aime24,aime25,aime26,mbpp}.yml)
  # One invocation for all benches: its parallel jobs share a single pod through the session
  # registry, instead of paying the pod start (image pull, weight download) once per bench.
  DOCKER_HOST=unix:///var/run/docker.sock uv run collect -b "${benchmarks[@]}" -m "$model" -n 2 --log-dir data || true
}

export -f run_model

printf '%s\n' "${models[@]}" | xargs -n 1 -P "$PARALLEL_JOBS" bash -c 'run_model "$@"' _
