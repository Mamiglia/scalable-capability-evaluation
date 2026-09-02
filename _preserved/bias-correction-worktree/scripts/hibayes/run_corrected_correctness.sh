#!/usr/bin/env bash
# Corrected-capability correctness model: full + 4 minus arms, then report, then
# the predictor-importance figures (which fit the 4 only-arms themselves).
# Two fits at a time: LOO on 128k rows is ~10GB, 36GB host holds two with headroom.
set -euo pipefail
cd /private/tmp/small-judge-bias-correction
HB=/Users/moriohamada/PycharmProjects/ai_safety/small-judge/analysis/hibayes/.venv/bin
export XDG_CACHE_HOME="$PWD/analysis/hibayes/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONPATH=src:scripts/hibayes

fit() {  # tag; config name is config_correctness[_<tag>]
  local tag=$1 out="analysis/hibayes/run_correctness_$1" cfg
  if [ "$tag" = full ]; then cfg=config_correctness; else cfg=config_correctness_$tag; fi
  [ -f "$out/models/judge_feature_logistic_model_$tag/diagnostics/inference_data_loo_i.csv" ] && { echo "skip $tag (complete)"; return; }
  rm -rf "$out"
  "$HB/hibayes-full" --config "scripts/hibayes/$cfg.yaml" --out "$out" --no-tui \
    > "analysis/hibayes/fit_$tag.log" 2>&1
}

# full first (reference), then minus arms two at a time
fit full
for pair in "minus_capability minus_length" "minus_position minus_style"; do
  set -- $pair
  fit "$1" &
  fit "$2" &
  wait
done

RUNS="analysis/hibayes/run_correctness_full \
  analysis/hibayes/run_correctness_minus_capability \
  analysis/hibayes/run_correctness_minus_length \
  analysis/hibayes/run_correctness_minus_position \
  analysis/hibayes/run_correctness_minus_style"
"$HB/python" scripts/hibayes/correctness_report.py $RUNS \
  --frame analysis/hibayes/gpqa_correctness.csv --out analysis/hibayes --prefix correctness \
  > analysis/hibayes/correctness_report.log 2>&1

# fits the only_* arms and draws the 03 figures into the worktree manuscript dir
"$HB/python" scripts/hibayes/correctness_predictor_figures.py --fit \
  > analysis/hibayes/predictor_figures.log 2>&1

touch analysis/hibayes/CORRECTNESS_DONE
