#!/usr/bin/env bash
# Collect Terminal-Bench 2.1 (89 tasks, linux/amd64 images -> needs an x86_64 Docker host).
# Resumable; copy the .eval files back into data/terminal_bench_2_1/<hash>/<model>/<repeat>/.
#
#   scripts/collect_terminal_bench.sh                                   # models.txt, 1 repeat
#   MODELS_FILE=one.txt scripts/collect_terminal_bench.sh --limit 2     # smoke run
#   scripts/collect_terminal_bench.sh -m configs/models/<org>/<model>.yml

set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -m)" == x86_64 ]] || {
  echo "terminal_bench_2_1 needs an x86_64 host" >&2
  exit 1
}

[[ ! -f .env ]] || {
  set -a
  source .env
  set +a
}
export UV_CACHE_DIR=/tmp/small-judge-uv-cache
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"

# skip Runpod-hosted models (no runpodctl / Runpod funds here); open-weight models first, closed last
mapfile -t all < <(grep -L '^  use: true$' $(grep -v '^#' "${MODELS_FILE:-configs/experiments/pairs/models.txt}"))
open=(); closed=()
for m in "${all[@]}"; do [[ $m =~ anthropic/|openai/|gemini|mistral-medium ]] && closed+=("$m") || open+=("$m"); done
models=("${open[@]}" "${closed[@]}")

# The per-model max_tokens in configs/models were tuned for single-turn math (up to 495k) and leave
# <16k of input budget on 128-160k-context models; OpenRouter rejects (400) any agent turn whose
# prompt + max_tokens exceeds the context. 32k output/turn is plenty here (observed ~400/turn).
# fail_on_error as a fraction: 3 flaky provider errors must not abort (and 10x retry) a whole model.
# ponytail: rootless podman's sqlite store locks up under concurrent container creates and the
# 150 sandboxes wrote 54 GB into the 56 GB /dev/shm graphroot in 6 min -> 2 evals x 32 sandboxes (~25 GB peak).
uv run collect \
  --jobs "${JOBS:-2}" \
  --max-sandboxes "${MAX_SANDBOXES:-32}" \
  --message-limit "${MESSAGE_LIMIT:-200}" \
  --max-tokens "${MAX_TOKENS:-32768}" \
  --fail-on-error "${FAIL_ON_ERROR:-0.1}" \
  -b configs/bench/terminal_bench_2_1.yml \
  -m "${models[@]}" \
  -n "${EPOCHS:-1}" \
  --log-dir "${LOG_DIR:-data}" \
  "$@"
