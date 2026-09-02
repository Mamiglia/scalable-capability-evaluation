#!/usr/bin/env bash
# Collect Cybench (39 CTF challenges, "hard" variant, linux/amd64 images -> x86_64 Docker host).
# Resumable. Writes into data/cybench/<hash>/<model>/<repeat>/.
#
#   scripts/collect_cybench.sh                 # full OpenRouter sweep (22 models)
#   scripts/collect_cybench.sh --limit 2       # smoke run
#   MAX_TOKENS_SMALL=... scripts/collect_cybench.sh
#
# Runpod models (gemma4-e2b, qwen3-32b, qwen3.5-4b) and the retired ling-2.6-flash are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -m)" == x86_64 ]] || { echo "cybench needs an x86_64 host" >&2; exit 1; }
[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
export CYBENCH_ACKNOWLEDGE_RISKS=1   # cybench agent gets unrestricted network; refuses to build without this

# Agentic CTF context grows until it overflows the model window (OpenRouter 400s the turn). Lower
# max_tokens buys input headroom; the two groups keep large-context reasoning models from a tight
# CoT cap. Split by OpenRouter context_length: SMALL <=164k, LARGE >=200k (measured 2026-09-02).
small=(
  configs/models/deepseek/deepseek-r1-0528.yml
  configs/models/deepseek/deepseek-v3.2.yml
  configs/models/google/gemma3-12b-it.yml
  configs/models/ibm/granite-4.1-8b.yml
  configs/models/meta/muse-glimmer-30b.yml
  configs/models/mistralai/mistral-medium-3.yml
)
large=(
  configs/models/anthropic/claude-opus-4.7.yml
  configs/models/anthropic/claude-sonnet-4.5.yml
  configs/models/google/gemini-3.5-flash.yml
  configs/models/moonshotai/kimi-k2.6.yml
  configs/models/moonshotai/kimi-k3.yml
  configs/models/nvidia/nemotron-3.5-lightning.yml
  configs/models/nvidia/nemotron-3-ultra-550b-a55b.yml
  configs/models/openai/gpt-4.1.yml
  configs/models/openai/gpt-5.yml
  configs/models/openai/o3.yml
  configs/models/qwen/qwen3.5-9b.yml
  configs/models/qwen/qwen3.8-2.4t-a95b.yml
  configs/models/qwen/qwen3.8-27b.yml
  configs/models/zai/glm-4.6.yml
  configs/models/zai/glm-5.2.yml
  configs/models/zai/glm-5.3.yml
)

run() {  # $1=max-tokens, rest=models
  local mt="$1"; shift
  uv run collect \
    --jobs "${JOBS:-1}" \
    --max-sandboxes "${MAX_SANDBOXES:-16}" \
    --message-limit "${MESSAGE_LIMIT:-100}" \
    --max-tokens "$mt" \
    --fail-on-error "${FAIL_ON_ERROR:-0.2}" \
    -b configs/bench/cybench.yml \
    -m "$@" \
    -n "${EPOCHS:-1}" \
    --log-dir "${LOG_DIR:-data}" \
    "${EXTRA[@]}"
}

EXTRA=("$@")
run "${MAX_TOKENS_LARGE:-32768}" "${large[@]}"
run "${MAX_TOKENS_SMALL:-16384}" "${small[@]}"
