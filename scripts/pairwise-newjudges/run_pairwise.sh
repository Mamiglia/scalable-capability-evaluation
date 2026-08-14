#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mkdir -p scripts/pairwise-newjudges/run_logs
P='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
run(){ uv run python scripts/pairwise-newjudges/run.py --config "scripts/pairwise-newjudges/configs/$4.yaml" --judge-model "$2" --judge-short "$1" --model-args "{\"provider\":{\"only\":[\"$3\"],$P}}" --max-connections 32 > "scripts/pairwise-newjudges/run_logs/$1-$4.log" 2>&1; }
for cfg in gpqa otis-aime; do
  run deepseek-v4-flash openrouter/deepseek/deepseek-v4-flash-0731 novita/fp8 $cfg &
  run ling-2.6-flash openrouter/inclusionai/ling-2.6-flash novita $cfg &
done
wait; echo "pairwise new judges done"
