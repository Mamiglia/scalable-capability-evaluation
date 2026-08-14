#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mkdir -p scripts/analyses/direct-judge/run_logs
P='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
for cfg in gpqa otis-aime; do
  uv run python scripts/analyses/direct-judge/run.py --config "scripts/analyses/direct-judge/configs/${cfg}_orig.yaml" \
    --judge-model openrouter/inclusionai/ling-2.6-flash --judge-short ling-2.6-flash \
    --model-args "{\"provider\":{\"only\":[\"novita\"],$P}}" --max-connections 48 \
    > "scripts/analyses/direct-judge/run_logs/ling-2.6-flash-$cfg.log" 2>&1 &
done
wait; echo "ling pointwise done"
