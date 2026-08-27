#!/usr/bin/env bash
# Run the whole pipeline on one domain: build its artefacts (only if missing — the fit moves
# when data/ changes, so rebuild deliberately with `uv run domain` after collecting), then
# judge the default pairs on every source quartile (Q1..Q3; ground truth Q4) — each run ranks
# itself and scores the ranking against Q1..Q4. Resumable; extra flags pass through to
# `judge` (--limit 2, ...).
#
#   bash scripts/experiment/domain.sh math [--judge configs/models/openai/gpt-oss-120b.yml] [--limit 2]
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ ! -f .env ]] || { set -a; source .env; set +a; }
domain=$1; shift
[[ -f "data/meta/domains/$domain/items.csv" ]] || uv run domain "configs/domain/$domain.yml"
for q in 1 2 3; do
  uv run judge --cfg "configs/experiments/$domain.yml" --q_src "$q" --q_dst 4 "$@"
done
