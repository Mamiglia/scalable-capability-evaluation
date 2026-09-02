#!/bin/bash
# gpt-oss-120b pointwise, both benches, via OpenRouter deepinfra/bf16 (matches its
# pairwise serving) using the original-framing configs (matches the other 6 judges).
# Resumable. Usage: bash scripts/analyses/direct-judge/run_gpt_oss_120b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
[ -f .env ] && { set -a; . ./.env; set +a; }   # OPENROUTER_API_KEY
PIN='{"provider":{"only":["deepinfra/bf16"],"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny"}}'
for cfg in gpqa otis-aime; do
  echo "== gpt-oss-120b $cfg $(date +%H:%M:%S)"
  uv run python scripts/analyses/direct-judge/run.py \
    --config "scripts/analyses/direct-judge/configs/${cfg}_orig.yaml" \
    --judge-model openrouter/openai/gpt-oss-120b --judge-short gpt-oss-120b \
    --model-args "$PIN"
done
echo "== gpt-oss-120b done =="
