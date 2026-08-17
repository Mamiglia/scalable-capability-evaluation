#!/usr/bin/env bash
# Shared driver for the judging sweeps. Sourced, not run.
#
# Runs pair configs from configs/experiments/pairs/ over BENCHES x JUDGES; the pair configs
# pin only major/minor, everything else is a CLI override. Callers set ROOT, BENCHES, JUDGES
# and FLAGS (the condition), optionally PAIRS (bare config stems; default is all of them),
# then call `sweep "$@"`.
#
# Logs land in $ROOT/<bench>/<judge>/<pair> and eval_set resumes off them, so re-running
# a script only fills gaps. ROOT must differ per condition — inspect sees the same task
# name and judge model whether or not reasoning is included, so a shared root would look
# like one run to it.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

[[ ! -f .env ]] || { set -a; source .env; set +a; }
export UV_CACHE_DIR=/tmp/small-judge-uv-cache

# Benchmarks not listed here get the default (free-form) system prompt.
declare -A PROMPT=(
  [gpqa]=assets/system_prompt/mcq.txt
)

sweep() {
  local -a pairs
  if [[ -n "${PAIRS+x}" ]]; then
    pairs=("${PAIRS[@]/#/configs/experiments/pairs/}")
    pairs=("${pairs[@]/%/.yml}")
  else
    pairs=(configs/experiments/pairs/*.yml)
  fi

  for bench in "${BENCHES[@]}"; do
    for judge in "${JUDGES[@]}"; do
      uv run judge \
        --cfg "${pairs[@]}" \
        -j "configs/models/$judge.yml" \
        -b "configs/bench/$bench.yml" \
        --system_prompt "${PROMPT[$bench]:-assets/system_prompt/default.txt}" \
        --log_dir "$ROOT" \
        "${FLAGS[@]}" "$@"
    done
  done
}
