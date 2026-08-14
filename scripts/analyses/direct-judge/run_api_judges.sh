#!/bin/bash
# Run the API judge fleet on both direct-judge configs, one judge at a time.
# Resumable: models with a success log are skipped on rerun.
# Usage: bash analyses/direct-judge/run_api_judges.sh            # all six
#        bash analyses/direct-judge/run_api_judges.sh api-glm-4.6 ...  # subset
set -euo pipefail
cd "$(dirname "$0")/../.."

PINS='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
# short|model|provider pin (pins from the pairwise reasoning ablation;
# qwen3-8b repinned alibaba/fp8 -> alibaba, old endpoint tag gone 2026-08-13)
JUDGES=(
  "api-ling-2.6-flash|openrouter/inclusionai/ling-2.6-flash|novita"
  "api-deepseek-v4-flash|openrouter/deepseek/deepseek-v4-flash-0731|novita/fp8"
  "api-qwen3-8b|openrouter/qwen/qwen3-8b|alibaba"
  "api-qwen3-32b|openrouter/qwen/qwen3-32b|siliconflow/fp8"
  "api-qwen3-vl-30b|openrouter/qwen/qwen3-vl-30b-a3b-instruct|siliconflow/fp8"
  "api-glm-4.6|openrouter/z-ai/glm-4.6|atlas-cloud/fp8"
)

for spec in "${JUDGES[@]}"; do
  IFS="|" read -r short model pin <<< "$spec"
  if [ $# -gt 0 ]; then
    case " $* " in *" $short "*) ;; *) continue;; esac
  fi
  for cfg in gpqa otis-aime; do
    echo "== $(date -Is) $short $cfg"
    uv run python analyses/direct-judge/run.py \
      --config "analyses/direct-judge/configs/$cfg.yaml" \
      --judge-model "$model" --judge-short "$short" \
      --model-args "{\"provider\":{\"only\":[\"$pin\"],$PINS}}"
  done
done
echo "== all requested judges done"
