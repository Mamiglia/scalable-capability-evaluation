#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache

mapfile -t models < <(
  while IFS= read -r model; do
    grep -q '^  use: true$' "$model" || printf '%s\n' "$model"
  done < <(find configs/models -type f -name '*.yml' | sort)
)

if ((${#models[@]} > 0)); then
  DOCKER_HOST=unix:///var/run/docker.sock uv run collect \
    -b configs/bench/{gpqa,aime24,aime25,aime26,mbpp}.yml \
    -m "${models[@]}" \
    -n 2 \
    --log-dir data
fi
