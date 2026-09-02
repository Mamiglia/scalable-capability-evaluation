#!/usr/bin/env bash
# Start one watchdog chain per judge over every config for that judge under a config
# root laid out as <config_root>/<benchmark>/epoch-<n>/<judge>.yaml (gen_pairs_judge_config.py
# output); the configs run one after another, judges side by side. Logs go to
# run_logs/judge_<config root name>_<judge>.log; the runner output is at <log_root>/runner.log of each config.
#
# Usage: bash scripts/judge/start_judging.sh <config_root> <judge>...
#        e.g. bash scripts/judge/start_judging.sh configs/judge_more_pairs o4-mini gemma-4-31b-it
#        For Llama-3.1-8B-Instruct export RUNPOD_BASE_URL first.
set -u
CONFIG_ROOT="$1"; shift
export PATH=~/.local/bin:$PATH
set -a; . ~/.env; set +a
export RUNPOD_API_KEY=${RUNPOD_API_KEY:-dummy}
mkdir -p run_logs
STALL_MIN=30

for judge in "$@"; do
  if [ "$judge" = "Llama-3.1-8B-Instruct" ] && [ -z "${RUNPOD_BASE_URL:-}" ]; then
    echo "RUNPOD_BASE_URL not set, skipping llama"; continue
  fi
  configs=$(ls "$CONFIG_ROOT"/*/epoch-*/"$judge".yaml | tr "\n" " ")
  tag=$(basename "$CONFIG_ROOT")
  nohup bash -c "
    for cfg in $configs; do
      log_root=\$(grep -m1 '^log_root:' \"\$cfg\" | cut -d' ' -f2)
      bash scripts/judge/run_with_watchdog.sh \"\$cfg\" \"\$log_root\" $STALL_MIN
    done
    echo \"JUDGE DONE $judge \$(date -u)\"" > "run_logs/judge_${tag}_$judge.log" 2>&1 &
  echo "started $judge (pid $!) -> run_logs/judge_${tag}_$judge.log"
done
