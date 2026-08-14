#!/bin/bash
# Parallel direct-scoring judge fleet. Each judge runs concurrently; the two
# benches run sequentially within a judge. Resumable — run.py skips model dirs
# that already hold a success log, so re-launching only does new work.
#
# Usage:
#   bash scripts/analyses/direct-judge/run_fleet_parallel.sh                 # all judges
#   bash scripts/analyses/direct-judge/run_fleet_parallel.sh api-gpt-oss-120b api-gpt-oss-20b
#   MAXCONN=8 bash scripts/analyses/direct-judge/run_fleet_parallel.sh       # gentler on rate limits
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p run_logs

PINS='"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"'
# short | judge-model | provider pin (endpoint-pinned per use-openrouter-safely;
# gpt-oss bf16 matches the judging-variations pairwise serving)
JUDGES=(
  "api-gpt-oss-120b|openrouter/openai/gpt-oss-120b|deepinfra/bf16"
  "api-gpt-oss-20b|openrouter/openai/gpt-oss-20b|deepinfra/bf16"
  "api-deepseek-v4-flash|openrouter/deepseek/deepseek-v4-flash-0731|novita/fp8"
  "api-qwen3-8b|openrouter/qwen/qwen3-8b|alibaba"
  "api-qwen3-32b|openrouter/qwen/qwen3-32b|siliconflow/fp8"
  "api-qwen3-vl-30b|openrouter/qwen/qwen3-vl-30b-a3b-instruct|siliconflow/fp8"
  "api-glm-4.6|openrouter/z-ai/glm-4.6|atlas-cloud/fp8"
  "api-ling-2.6-flash|openrouter/inclusionai/ling-2.6-flash|novita"
)
MAXCONN=${MAXCONN:-16}

run_judge() {
  local short="$1" model="$2" pin="$3"
  for cfg in gpqa otis-aime; do
    echo "[$short] start $cfg $(date +%H:%M:%S)"
    uv run python run.py --config "configs/$cfg.yaml" \
      --judge-model "$model" --judge-short "$short" \
      --model-args "{\"provider\":{\"only\":[\"$pin\"],$PINS}}" \
      --max-connections "$MAXCONN" >"run_logs/$short-$cfg.log" 2>&1
    echo "[$short] done  $cfg (rc=$?) $(date +%H:%M:%S)"
  done
}

pids=()
for spec in "${JUDGES[@]}"; do
  IFS="|" read -r short model pin <<< "$spec"
  if [ $# -gt 0 ]; then case " $* " in *" $short "*) ;; *) continue;; esac; fi
  run_judge "$short" "$model" "$pin" &
  pids+=($!)
done
echo "launched ${#pids[@]} judges in parallel (max-connections $MAXCONN each); per-judge stdout in run_logs/"
wait
echo "== all judges finished =="
