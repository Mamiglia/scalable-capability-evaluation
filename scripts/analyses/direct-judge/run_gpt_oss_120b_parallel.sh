#!/bin/bash
# gpt-oss-120b pointwise, BOTH benches in parallel, higher concurrency. Resumable
# (skips model dirs with a success log). Same serving/framing as the sequential
# runner. Usage: bash scripts/analyses/direct-judge/run_gpt_oss_120b_parallel.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
mkdir -p scripts/analyses/direct-judge/run_logs
PIN='{"provider":{"only":["deepinfra/bf16"],"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"}}'
for cfg in gpqa otis-aime; do
  uv run python scripts/analyses/direct-judge/run.py \
    --config "scripts/analyses/direct-judge/configs/${cfg}_orig.yaml" \
    --judge-model openrouter/openai/gpt-oss-120b --judge-short gpt-oss-120b \
    --model-args "$PIN" --max-connections 64 \
    > "scripts/analyses/direct-judge/run_logs/120b-$cfg.log" 2>&1 &
done
wait
echo "== gpt-oss-120b both benches done =="
