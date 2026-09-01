#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
set -a
source /root/.config/canonical-anchor-redo/.env
set +a
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${OPENROUTER_BASELINE_USAGE:?set immediately before the first paid call}"

exec uv run python scripts/frontier_eci/runpod/parallel_topup.py \
  --baseline "$OPENROUTER_BASELINE_USAGE" "$@"
