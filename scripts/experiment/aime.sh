#!/usr/bin/env bash
# AIME: the pilot pairs and judges over the three AIME years, reasoning stripped.
# Free-form answers, so the default system prompt applies (no MCQ variant).
source "$(dirname "$0")/_sweep.sh"

ROOT=logs/aime
BENCHES=(aime24 aime25 aime26)
JUDGES=(openai/gpt-oss-120b google/gemma4-31b)
FLAGS=(--include_reasoning false)

sweep "$@"
