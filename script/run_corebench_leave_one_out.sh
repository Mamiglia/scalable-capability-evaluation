#!/usr/bin/env bash
# Script to run all CoreBench paired judge experiments using leave-one-out strategy
# Models: Qwen (qwen3.5-9b), Gemma (gemma-4-31b-it), Kimi (kimi-k2.6)
# Levels: easy, medium, hard

set -e

EXTRA_ARGS=("$@")

LEVELS=("easy" "medium" "hard")
JUDGES=("qwen" "gemma" "kimi")

echo "============================================================"
echo "Running CoreBench Leave-One-Out Judge Experiments"
echo "Levels: ${LEVELS[*]}"
echo "Judges: ${JUDGES[*]}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra args: ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

# Ensure dataset soft-links registry is fresh
uv run python -c "from registry.build import build_registry; build_registry()"

for level in "${LEVELS[@]}"; do
    for judge in "${JUDGES[@]}"; do
        config="configs/experiments/core-bench/judge-${judge}-${level}.yml"
        echo ""
        echo "---> Running experiment: Judge=${judge}, Level=${level} (${config})"
        uv run judge --config_path "${config}" "${EXTRA_ARGS[@]}"
    done
done

echo ""
echo "All leave-one-out CoreBench judge experiments finished!"
