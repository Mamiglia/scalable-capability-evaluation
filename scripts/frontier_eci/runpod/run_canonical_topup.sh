#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
set -a
source /root/.config/canonical-anchor-redo/.env
set +a
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${OPENROUTER_BASELINE_USAGE:?set immediately before the first paid call}"

uv run python scripts/frontier_eci/run_budgeted.py \
  configs/frontier_eci_canonical_anchors/runs/*/*/*.yaml \
  --baseline "$OPENROUTER_BASELINE_USAGE" \
  --limit 150 \
  --preflight judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/preflight.csv \
  --ledger judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/ledger.jsonl
