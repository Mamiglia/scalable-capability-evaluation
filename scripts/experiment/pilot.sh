#!/usr/bin/env bash
# Pilot: 25 pairs x {gpqa, mbpp}, judged with reasoning traces stripped.
# Logs: logs/pilot/<bench>/<judge>/<condition>. Extra flags pass through, e.g. --limit 2.
source "$(dirname "$0")/_sweep.sh"

ROOT=logs/pilot
BENCHES=(gpqa mbpp)
JUDGES=(
  openai/gpt-oss-120b 
  google/gemma4-31b
  microsoft/phi3-medium-128k-instruct
  deepseek/deepseek-v4-flash-0731
  inclusionai/ling-2.6-flash
  meta-llama/llama3.1-8b
  deepseek/deepseek-r1-0528
  openai/o4-mini    
)
FLAGS=(--include_reasoning false)

sweep "$@"
