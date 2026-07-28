#!/bin/env/sh 

JUDGE_MODEL=openrouter/inclusionai/ling-2.6-flash
BENCHMARK=gpqa_diamond
MINOR_MODEL=gemma_2b
MAJOR_MODEL=gemma_31b

uv run inspect eval src/task.py \
    -T log_minor_path="$PWD/data/${MINOR_MODEL}/${BENCHMARK}/100000000001.eval" \
    -T log_major_path="$PWD/data/${MAJOR_MODEL}/${BENCHMARK}/100000000001.eval" \
    --model $JUDGE_MODEL \
    --log-dir logs/${BENCHMARK}/$JUDGE_MODEL/${MAJOR_MODEL}_vs_${MINOR_MODEL} \
    --limit 100000
