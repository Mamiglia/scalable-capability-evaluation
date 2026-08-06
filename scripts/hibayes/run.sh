#!/usr/bin/env bash
set -euo pipefail

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/analysis/hibayes/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

.venv/bin/python scripts/hibayes/prepare_gpqa.py
.venv/bin/hibayes-full --config scripts/hibayes/config.yaml --out analysis/hibayes/run --no-tui
