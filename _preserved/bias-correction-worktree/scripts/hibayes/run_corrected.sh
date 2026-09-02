#!/usr/bin/env bash
# Corrected-capability GLM: baseline then style arm, then LOO. Sequential so only
# one model's log-likelihood + PSIS-LOO is in memory at a time.
set -euo pipefail
cd /private/tmp/small-judge-bias-correction
HB=/Users/moriohamada/PycharmProjects/ai_safety/small-judge/analysis/hibayes/.venv/bin
export XDG_CACHE_HOME="$PWD/analysis/hibayes/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"

"$HB/hibayes-full" --config scripts/hibayes/config_corrected_without_style.yaml \
  --out analysis/hibayes/run_corrected_baseline --no-tui
"$HB/hibayes-full" --config scripts/hibayes/config_corrected.yaml \
  --out analysis/hibayes/run_corrected_style --no-tui
"$HB/python" scripts/hibayes/loo_comparison.py \
  analysis/hibayes/run_corrected_style analysis/hibayes/run_corrected_baseline \
  > analysis/hibayes/loo_corrected.txt 2>&1
touch analysis/hibayes/CORRECTED_GLM_DONE
