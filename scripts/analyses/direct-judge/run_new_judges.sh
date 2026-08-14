#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mkdir -p scripts/analyses/direct-judge/run_logs
P='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
run(){ uv run python scripts/analyses/direct-judge/run.py --config "scripts/analyses/direct-judge/configs/$4_orig.yaml" --judge-model "$2" --judge-short "$1" --model-args "{\"provider\":{\"only\":[\"$3\"],$P}}" --max-connections 48 > "scripts/analyses/direct-judge/run_logs/$1-$4.log" 2>&1; }
for cfg in gpqa otis-aime; do
  run deepseek-v4-flash openrouter/deepseek/deepseek-v4-flash-0731 novita/fp8 $cfg &
  run glm-4.6 openrouter/z-ai/glm-4.6 atlas-cloud/fp8 $cfg &
done
wait; echo "== deepseek + glm pointwise done =="
