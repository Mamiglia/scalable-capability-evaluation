#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mkdir -p scripts/analyses/direct-judge/run_logs scripts/pairwise-newjudges/run_logs
P='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
MA="{\"provider\":{\"only\":[\"baseten/fp8\"],$P}}"
M=openrouter/deepseek/deepseek-v4-flash-0731
for cfg in gpqa otis-aime; do
  uv run python scripts/analyses/direct-judge/run.py --config "scripts/analyses/direct-judge/configs/${cfg}_orig.yaml" --judge-model $M --judge-short deepseek-v4-flash --model-args "$MA" --max-connections 96 > "scripts/analyses/direct-judge/run_logs/deepseek-fast-pt-$cfg.log" 2>&1 &
  uv run python scripts/pairwise-newjudges/run.py --config "scripts/pairwise-newjudges/configs/${cfg}.yaml" --judge-model $M --judge-short deepseek-v4-flash --model-args "$MA" --max-connections 96 > "scripts/pairwise-newjudges/run_logs/deepseek-fast-pw-$cfg.log" 2>&1 &
done
wait; echo "deepseek fast done"
