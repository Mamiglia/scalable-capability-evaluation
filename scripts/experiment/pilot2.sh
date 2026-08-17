#!/usr/bin/env bash
# Pilot 2: same pairs, benchmarks and condition as pilot.sh, six cheaper/older judges.
# Shares logs/pilot as its root — same condition, so the judge dir keeps the runs apart.
source "$(dirname "$0")/_sweep.sh"

ROOT=logs/pilot
BENCHES=(gpqa mbpp)
JUDGES=(
  microsoft/phi3-medium-128k-instruct
  deepseek/deepseek-v4-flash-0731
  inclusionai/ling-2.6-flash
  meta-llama/llama3.1-8b
  deepseek/deepseek-r1-0528
  openai/o4-mini
)
FLAGS=(--include_reasoning false)

sweep "$@"
