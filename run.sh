#!/bin/env/sh 

JUDGE_MODEL=openrouter/inclusionai/ling-2.6-flash
BENCHMARK=gpqa_diamond
MINOR_MODEL=gemma_2b
MAJOR_MODEL=gemma_31b

usage() {
    printf 'Usage: %s [--judge MODEL] [--benchmark NAME] [--minor MODEL] [--major MODEL]\n' "$0"
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --judge|-j)
            shift
            [ $# -gt 0 ] || usage
            JUDGE_MODEL="$1"
            ;;
        --benchmark|-b)
            shift
            [ $# -gt 0 ] || usage
            BENCHMARK="$1"
            ;;
        --minor|-m)
            shift
            [ $# -gt 0 ] || usage
            MINOR_MODEL="$1"
            ;;
        --major|-M)
            shift
            [ $# -gt 0 ] || usage
            MAJOR_MODEL="$1"
            ;;
        --help|-h)
            usage
            ;;
        *)
            printf 'Unknown option: %s\n' "$1"
            usage
            ;;
    esac
    shift
done

uv run inspect eval src/task.py \
    -T log_minor_path="$PWD/data/${MINOR_MODEL}/${BENCHMARK}/100000000001.eval" \
    -T log_major_path="$PWD/data/${MAJOR_MODEL}/${BENCHMARK}/100000000001.eval" \
    --model $JUDGE_MODEL \
    --log-dir logs/${BENCHMARK}/$JUDGE_MODEL/${MAJOR_MODEL}_vs_${MINOR_MODEL} \
    --limit 100000
