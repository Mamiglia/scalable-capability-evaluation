#!/usr/bin/env bash
# Fit the three GLMs one after another and score each on the confirmation items.
#   nohup caffeinate -i scripts/verdict_dissection/glm/run.sh > run_logs/glm-fits.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.."
uv run python scripts/verdict_dissection/glm/prepare.py judge_results/verdict_dissection/glm_inputs
for name in stronger_model_from_pair capability_from_transcript judge_verdict_from_pair; do
  config=scripts/verdict_dissection/glm/configs/$name.yaml
  out=judge_results/verdict_dissection/glm_fits/$name
  [ -f "$out/models/$name/inference_data.nc" ] || uv run hibayes-full --config "$config" --out "$out" --no-tui
  uv run python scripts/verdict_dissection/glm/evaluate.py "$config" "$out"
done
