#!/usr/bin/env bash
TASKS=(easy medium hard)

JUDGE_MODELS=(
  "configs/models/kimi-k2.6.yml"
  "configs/models/gemma4-31b.yml"
  "configs/models/qwen3.5-9b.yml"
)
MAJOR_MODELS=(
  "configs/models/gemma4-31b.yml"
  "configs/models/kimi-k2.6.yml"
  "configs/models/kimi-k2.6.yml"
)
MINOR_MODELS=(
  "configs/models/qwen3.5-9b.yml"
  "configs/models/qwen3.5-9b.yml"
  "configs/models/gemma4-31b.yml"
)

MAX_PARALLEL=16    # tune to your OpenRouter rate limit / CPU
LOGDIR="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"

run_one() {
    local task="$1" judge="$2" major="$3" minor="$4"
    local judge_stem="$(basename "$judge" .yml)"
    local major_stem="$(basename "$major" .yml)"
    local minor_stem="$(basename "$minor" .yml)"
    local tag="${task}__${judge_stem}"
    local folder="${judge_stem}:${major_stem}|${minor_stem}"
    uv run judge \
        --benchmark "configs/bench/core-bench/$task.yml" \
        --judge "$judge" \
        --major "$major" \
        --minor "$minor" \
        --log_dir "logs/judge_run2/$task/$folder" \
        --display plain \
        > "$LOGDIR/$tag.log" 2>&1
}

pids=()
for task in "${TASKS[@]}"; do
    for i in "${!JUDGE_MODELS[@]}"; do
        # throttle: wait until we're under MAX_PARALLEL
        while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
            wait -n
        done
        echo "Launching task: $task, judge: ${JUDGE_MODELS[$i]}, major: ${MAJOR_MODELS[$i]}, minor: ${MINOR_MODELS[$i]}"
        run_one "$task" "${JUDGE_MODELS[$i]}" "${MAJOR_MODELS[$i]}" "${MINOR_MODELS[$i]}" &
        pids+=($!)
    done
done

wait
echo "All runs finished. Logs in $LOGDIR"