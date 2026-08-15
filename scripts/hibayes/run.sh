#!/usr/bin/env bash
set -euo pipefail

# hibayes gets its own venv: installing it into .venv downgrades rich and textual,
# which inspect-ai's TUI uses. Create it once with
#   uv venv "$HIBAYES_VENV" --python 3.14
#   uv pip install --python "$HIBAYES_VENV/bin/python" \
#     "hibayes @ git+https://github.com/UKGovernmentBEIS/hibayes"
HIBAYES_VENV=analysis/hibayes/.venv

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/analysis/hibayes/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

.venv/bin/python scripts/hibayes/prepare_from_verdicts.py
"$HIBAYES_VENV/bin/hibayes-full" --config scripts/hibayes/config.yaml --out analysis/hibayes/run --no-tui
