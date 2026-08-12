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

failed=()
for model in "${models[@]}"; do
  if [[ ${SKIP_API:-false} != true ]]; then
    uv run collect \
      -b configs/bench/{gpqa,aime24,aime25,aime26}.yml \
      -m "$model" \
      -n 2 \
      --log-dir data \
      || failed+=("$model")
  fi

  if [[ ${SKIP_MBPP:-false} != true ]]; then
    # MBPP starts a local sandbox; serialize epochs to avoid Podman socket races.
    if ! DOCKER_HOST=unix:///var/run/docker.sock uv run collect -b configs/bench/mbpp.yml -m "$model" -n 1 --log-dir data; then
      failed+=("$model MBPP epoch 1")
      continue
    fi
    DOCKER_HOST=unix:///var/run/docker.sock uv run collect -b configs/bench/mbpp.yml -m "$model" -n 2 --log-dir data || failed+=("$model MBPP epoch 2")
  fi
done

((${#failed[@]} == 0)) || { printf 'Failed:\n  %s\n' "${failed[@]}"; exit 1; }
