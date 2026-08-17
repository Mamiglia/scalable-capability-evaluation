#!/usr/bin/env bash
# Pilot: 25 pairs x {gpqa, mbpp}, judged with reasoning traces stripped.
# Logs: logs/pilot/<bench>/<judge>/<pair>. Extra flags pass through, e.g. --limit 2 --jobs 4.
source "$(dirname "$0")/_sweep.sh"

ROOT=logs/pilot
BENCHES=(gpqa mbpp)
JUDGES=(openai/gpt-oss-120b google/gemma4-31b)
FLAGS=(--include_reasoning false)

sweep "$@"
