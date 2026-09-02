#!/usr/bin/env bash
set -euo pipefail

# hibayes gets its own venv: installing it into .venv downgrades rich and textual,
# which inspect-ai's TUI uses. Create it once with
#   uv venv "$HIBAYES_VENV" --python 3.14
#   uv pip install --python "$HIBAYES_VENV/bin/python" \
#     "hibayes @ git+https://github.com/UKGovernmentBEIS/hibayes"
HIBAYES_VENV=analysis/hibayes/.venv

# Config and output directory are both required. Neither is defaulted: the two
# arms of the LOO comparison live in sibling directories, and a defaulted output
# would let a plain run silently overwrite the arm it is meant to be compared
# against - after which the comparison would still run, against a stale fit.
if [ $# -ne 2 ]; then
  echo "usage: $0 <config.yaml> <output-directory>" >&2
  echo "  e.g. $0 scripts/hibayes/config.yaml analysis/hibayes/run_style" >&2
  exit 2
fi
CONFIG=$1
OUT=$2
if [ -e "$OUT" ]; then
  echo "$OUT already exists; refusing to overwrite a fit. Remove it or pick another." >&2
  exit 2
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/analysis/hibayes/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

.venv/bin/python scripts/hibayes/style_rates.py
.venv/bin/python scripts/hibayes/prepare_from_verdicts.py
"$HIBAYES_VENV/bin/hibayes-full" --config "$CONFIG" --out "$OUT" --no-tui
