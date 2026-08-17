#!/usr/bin/env bash
# AIME: the pilot pairs and judges over the three AIME years, reasoning stripped.
# Free-form answers, so the default system prompt applies (no MCQ variant).
source "$(dirname "$0")/_sweep.sh"

ROOT=logs/aime/reason
BENCHES=(aime24 aime25 aime26)
JUDGES=(
  openai/gpt-oss-120b
  google/gemma4-31b
  deepseek/deepseek-v4-flash-0731
  inclusionai/ling-2.6-flash
)

PAIRS=(
  glm-5.2-vs-deepseek-r1-0528
  glm-5.2-vs-glm-4.6
  kimi-k2.6-vs-glm-5.2
  kimi-k2.6-vs-kimi-k3
  kimi-k3-vs-deepseek-r1-0528
  qwen3.5-4b-vs-qwen3.5-9b
)
FLAGS=(--include_reasoning true --fail-on-error 0.3 --retry-attempts 2)

sweep "$@"


# WITHOUT REASONING
ROOT=logs/aime

unset PAIRS
FLAGS=(--include_reasoning false)

sweep "$@"
