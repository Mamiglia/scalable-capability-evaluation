#!/bin/bash
# deepseek pointwise ONLY, low concurrency to dodge baseten 429s. Resumable (skips done evals).
# Pairwise was dropped; do not add it back here.
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mkdir -p scripts/analyses/direct-judge/run_logs
CONN=12   # baseten/fp8 429s at 96; keep low
P='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
MA="{\"provider\":{\"only\":[\"baseten/fp8\"],$P}}"
M=openrouter/deepseek/deepseek-v4-flash-0731
for cfg in gpqa otis-aime; do
  uv run python scripts/analyses/direct-judge/run.py --config "scripts/analyses/direct-judge/configs/${cfg}_orig.yaml" --judge-model $M --judge-short deepseek-v4-flash --model-args "$MA" --max-connections $CONN > "scripts/analyses/direct-judge/run_logs/deepseek-pt-lowconn-$cfg.log" 2>&1 &
done
wait; echo "deepseek pointwise lowconn done"
