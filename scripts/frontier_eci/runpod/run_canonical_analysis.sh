#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
exec uv run python scripts/frontier_eci/canonical_analysis.py "$@"
