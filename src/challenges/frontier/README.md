# Frontier models on the coding subset

Closed frontier models are collected on the coding domain only on the items a protocol needs: per bench, the
2026-I train set (the EASY items every protocol judges) and the 2026-I HARD set (what θ* is fit on), taken from
the frozen `data/meta/coding/items.csv`. The subsets are the task arguments of
`configs/bench/terminal_bench_2_1_frontier.yml` and `configs/bench/swe-bench-mini-frontier.yml`.

Anthropic and OpenAI refusals (content-filter stops) are left unscored in the released logs
(`sample.metadata.refusal`), so refused items drop out of θ* and judging instead of counting as failures.

```bash
PYTHONPATH=src .venv/bin/python src/challenges/frontier/items.py --check   # the bench ymls match the frozen item lists

# collection: terminal then swe per model (configs/collect/frontier_models.txt)
src/challenges/frontier/collect.sh configs/models/anthropic/claude-opus-4.8.yml ...

# data/meta/coding_frontier/ from data/meta/coding/: the frontier models become extra 2026-I targets,
# θ* from reference.freeze.local_fit on their scored HARD responses (item parameters pinned to the global fit)
PYTHONPATH=src .venv/bin/python src/challenges/frontier/integrate.py --write

# judging: one target at a time with the reasoning-hidden judge (frontier transcripts carry no raw reasoning)
src/challenges/frontier/judge.sh configs/judges/hidden/kimi-k2-thinking.yml <target> ...
uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I \
  --direct-judge configs/judges/direct/kimi-k2-thinking.yml --protocol ridge embedding

# figure: prediction scores per frontier target and protocol, open-weight 2026-I targets as the reference group
PYTHONPATH=src .venv/bin/python src/challenges/frontier/plot_scores.py   # -> analysis/frontier/prediction_scores.*
```

Always pass `--semester 2026-I`: the 2025-II columns of `coding_frontier` are coding's, unchanged. The GPT-5.x
targets come from Epoch AI's public SWE-bench logs, imported into `data/swe_bench_mini/6d369262/` (SWE-bench only,
Epoch's scaffold).
