#!/usr/bin/env bash
# Judge the coding_frontier 2026-I targets one after another with one judge config, in the given order.
# Each pass judges only one target (the others go into `exclude_models`, which is not part of the
# condition hash, so every pass lands in the same run dir); reference-reference pairs come with the
# first pass and are served from inspect's generate cache when the coding run already judged them.
#
#   setsid -f bash -c 'src/challenges/frontier/judge.sh configs/judges/kimi-k2-thinking.yml \
#       claude-fable-5 gpt-5.6-sol claude-opus-4.8 gemini-3.5-flash claude-opus-4.7 union-alpha \
#       > artifacts/frontier/judge.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../../.."
[[ ! -f .env ]] || { set -a; source .env; set +a; }
judge=$1; shift
targets=("$@")
mkdir -p artifacts/frontier
for target in "${targets[@]}"; do
  others=$(printf '%s\n' "${targets[@]}" | grep -vx "$target" | sed 's/^/  - /')
  cfg=artifacts/frontier/judge-$(basename "$judge" .yml)-$target.yml
  { cat "$judge"; printf 'exclude_models:\n%s\n' "$others"; } > "$cfg"
  echo "=== $(date -Is) $target ($cfg)"
  uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I --judge "$cfg"
done
echo "=== $(date -Is) all done"
