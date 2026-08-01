#!/usr/bin/env bash

TASKS=(
    easy medium hard
)

JUDGE_MODELS=(
  "openrouter/moonshotai/kimi-k2.6"
  "openrouter/google/gemma-4-31b-it"
  "openrouter/qwen/qwen3.5-9b"
)

MAJOR_MODELS=(
  "openrouter/google/gemma-4-31b-it"
  "openrouter/moonshotai/kimi-k2.6"
  "openrouter/moonshotai/kimi-k2.6"
)

MINOR_MODELS=(
  "openrouter/qwen/qwen3.5-9b"  
  "openrouter/qwen/qwen3.5-9b"
  "openrouter/google/gemma-4-31b-it"
)


for task in "${TASKS[@]}"; do
    for i in "${!JUDGE_MODELS[@]}"; do
        echo "Running benchmark for task: $task, judge: ${JUDGE_MODELS[$i]}"
        
        uv run judge \
            --benchmark "configs/bench/core-bench/$task.yml" \
            --judge ${JUDGE_MODELS[$i]} \
            --major ${MAJOR_MODELS[$i]} \
            --minor ${MINOR_MODELS[$i]} "$@"
        
    done
done