#!/usr/bin/env bash
# Reasoning condition: same benchmarks as pilot.sh/pilot2.sh, but the judge sees the
# reasoning traces. Separate log root — inspect cannot tell the two conditions apart on
# its own (same task name, same judge model), so a shared root would look like one run.
#
# Only the pairs where both models actually emit reasoning in the collected transcripts;
# the rest have nothing to include and would just duplicate the pilot runs. This list is
# what the original sweep covered, reproduced verbatim.
source "$(dirname "$0")/_sweep.sh"

ROOT=logs/pilot_reasoning
BENCHES=(gpqa mbpp)
JUDGES=(
  openai/gpt-oss-120b
  google/gemma4-31b
  microsoft/phi3-medium-128k-instruct
  deepseek/deepseek-v4-flash-0731
  inclusionai/ling-2.6-flash
  meta-llama/llama3.1-8b
)
PAIRS=(
  claude-opus-4.7-vs-kimi-k3
  glm-5.2-vs-deepseek-r1-0528
  glm-5.2-vs-glm-4.6
  glm-5.2-vs-gpt-5
  gpt-5-vs-glm-4.6
  gpt-5-vs-o3
  gpt-5-vs-qwen3-32b
  kimi-k2.6-vs-glm-5.2
  kimi-k2.6-vs-kimi-k3
  kimi-k3-vs-deepseek-r1-0528
  kimi-k3-vs-o3
  qwen3.5-4b-vs-qwen3.5-9b
)
FLAGS=(--include_reasoning true)

sweep "$@"
