#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
# Authoritative main analysis named by MAIN_OUTPUT.json.
exec uv run python scripts/frontier_eci/rank_regression.py "$@"
