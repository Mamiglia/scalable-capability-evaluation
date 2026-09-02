#!/usr/bin/env bash
set -euo pipefail

# HF GPQA arm of run.sh: build the full-style frame, then fit the same hibayes
# model. Output directory is required; a plain run must not overwrite a fit.
HIBAYES_VENV=analysis/hibayes/.venv

if [ $# -ne 1 ]; then
  echo "usage: $0 <output-directory>" >&2
  echo "  e.g. $0 analysis/hibayes/run_hf_gpqa" >&2
  exit 2
fi
OUT=$1
if [ -e "$OUT" ]; then
  echo "$OUT already exists; refusing to overwrite a fit. Remove it or pick another." >&2
  exit 2
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/analysis/hibayes/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

.venv/bin/python scripts/hibayes/prepare_hf.py
"$HIBAYES_VENV/bin/hibayes-full" --config scripts/hibayes/config_hf.yaml --out "$OUT" --no-tui
