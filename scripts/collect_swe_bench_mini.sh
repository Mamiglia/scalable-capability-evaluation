#!/usr/bin/env bash
# Collect SWE-bench Verified mini (50 instances) for the models missing from the
# 2026-08-22/23 sweep. Resumable. Writes into data/swe_bench_mini/477dc96c/<model>/<repeat>/.
#
#   scripts/collect_swe_bench_mini.sh --limit 2   # smoke first, real runs cost money
#   scripts/collect_swe_bench_mini.sh             # full sweep
#
# Docker sandbox -> x86_64 host with a docker(-compatible) socket. swebench 3.x supplies the
# grader (5.x breaks the import), hence `uv run --with`.
# mistral-small-3.2-24b is Runpod/vLLM-served and is NOT in this sweep: run it separately with
# --runpod once the OpenRouter models are done (the sandbox stays local, the model is remote),
# at --max-sandboxes 4 (its 65536 max-model-len fits only ~4-5 concurrent sequences).
# Roster swaps (2026-09-09): mistral-small-2503 -> mistral-small-3.2-24b;
# llama-4-scout -> minimax-m1 (Llama 4 emits pythonic tool calls no endpoint parses).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -m)" == x86_64 ]] || { echo "swe_bench_mini needs an x86_64 host" >&2; exit 1; }
[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"

# Agentic context grows until it overflows the window (OpenRouter 400s the turn), so max_tokens
# is capped well below the per-model ymls (which are tuned for single-turn math). Groups by the
# pinned endpoint's context_length / max_completion_tokens (checked live 2026-09-08).
tiny=(   # endpoint output cap is 8192
  configs/models/qwen/qwen3-235b-a22b.yml
)
small=(  # context <= 164k
  configs/models/deepseek/deepseek-v3-0324.yml
  configs/models/openai/gpt-oss-120b.yml
  configs/models/qwen/qwen3-235b-a22b-thinking-2507.yml
  configs/models/zai/glm-4.5.yml
)
# DEFERRED 2026-09-10: gpt-oss-20b. Its only bf16 endpoint (deepinfra) 429s -- 0/50 after ~50 min,
# retry 11 at 1800 s backoff, while gpt-oss-120b on the same endpoint crawled at 752 retries.
# No other bf16 endpoint exists on the slug. When picking this up: gpt-oss ships natively with
# MXFP4 MoE weights, so coreweave/fp4 or parasail/fp4 (both OK + tool-capable, probed 09-10) are
# arguably the reference precision rather than a downgrade. Needs allow_fallbacks: true to route
# past a 429ing first entry (see the qwen3.5-397b-a17b note).
#   MAX_TOKENS_SMALL=16384 scripts/collect_swe_bench_mini.sh -m configs/models/openai/gpt-oss-20b.yml
# DEFERRED 2026-09-08: qwen3.5-397b-a17b's pinned parasail/fp8 returns 429 on even a single
# request (both this sweep and the concurrent hle_gold run wedge on it). Run it on its own once
# hle_gold is done, and re-probe the endpoint first -- gmicloud/fp8 (the pin abandoned earlier
# the same day for server_error) and phala both answered fine at the same fp8 precision and caps:
#   MAX_TOKENS_LARGE=32768 scripts/collect_swe_bench_mini.sh -m configs/models/qwen/qwen3.5-397b-a17b.yml
large=(  # context >= 200k
  configs/models/meta-llama/llama-4-maverick.yml
  configs/models/minimax/minimax-m2.yml
  configs/models/moonshotai/kimi-k2-thinking.yml
  configs/models/minimax/minimax-m1.yml
  configs/models/google/gemma4-31b.yml   # repinned parasail/fp8 2026-09-09: 262144 ctx / 235929 out
  configs/models/qwen/qwen3.6-27b.yml
  configs/models/zai/glm-4.7.yml
  configs/models/zai/glm-5.1.yml
)

run() {  # $1=max-tokens, rest=models
  local mt="$1"; shift
  uv run --with 'swebench>=3.0.15,<4' collect \
    --jobs "${JOBS:-1}" \
    --max-sandboxes "${MAX_SANDBOXES:-8}" \
    --message-limit "${MESSAGE_LIMIT:-100}" \
    --max-tokens "$mt" \
    --fail-on-error "${FAIL_ON_ERROR:-0.2}" \
    -b configs/bench/swe-bench-mini.yml \
    -m "$@" \
    -n "${EPOCHS:-2}" \
    --log-dir "${LOG_DIR:-data}" \
    "${EXTRA[@]}"
}

EXTRA=("$@")
run "${MAX_TOKENS_LARGE:-32768}" "${large[@]}"
run "${MAX_TOKENS_SMALL:-16384}" "${small[@]}"
run "${MAX_TOKENS_TINY:-8192}"   "${tiny[@]}"
