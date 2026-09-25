# Pairwise judging on transcripts with missing thirds

Each rendered transcript (reasoning shown) is split into three equal character spans and the judge sees one of the
7 non-empty subsets; `{1,2,3}` is the reasoning-shown baseline. Every dropped run of thirds is shown as `\n...\n`
and the user prompt tells the judge that parts may be redacted (`judge/dataset.py THIRDS_NOTE`). Domains science,
math, coding, cyber x semesters 2025-II, 2026-I; the top reference judge only, on exactly the baseline's items.

A condition is the baseline's `configs/judges/reasoning/<stem>.yml` plus `evaluate --truncate thirds-<kept>`
(`JudgeConfig.truncate`, `configs/judge.py TRUNCATE_MODES`). `condition_hash` appends `truncate`, so each kept set
has its own dir next to the baseline under `logs/judge/<domain>/<semester>/<judge>/`.

| domain | semester | judge | YAML stem (`runs.py RUNS`) |
| --- | --- | --- | --- |
| science | 2025-II | minimax-m1 | `minimax-m1` |
| science | 2026-I | kimi-k2-thinking | `kimi-k2-thinking` |
| math | 2025-II | deepseek-r1-0528 | `deepseek-r1-0528` |
| math | 2026-I | kimi-k2-thinking | `kimi-k2-thinking` |
| coding | 2025-II | minimax-m1 | `minimax-m1` |
| coding | 2026-I | minimax-m2 | `minimax-m2-full` |
| cyber | 2025-II | deepseek-r1-0528 | `deepseek-r1-0528` |
| cyber | 2026-I | deepseek-v3.2 | `deepseek-v3.2-full` |

Pairs whose kept third leaves under `MIN_RESPONSE_CHARS` are dropped by `build_dataset`, so a thirds run judges a
subset of the baseline's samples.

```bash
uv run evaluate configs/domain/<domain>.yml --semester <semester> --judge configs/judges/reasoning/<stem>.yml --truncate thirds-<kept>
PYTHONPATH=src .venv/bin/python src/challenges/thirds/audit.py <domain> <semester> [--inputs]
```

`audit.py` checks every log succeeded with the right metadata and every sample id exists in the baseline dir;
`--inputs` rebuilds every prompt from `data/` and compares byte for byte. The figures are drawn by
`scripts/analyses/position_contributions.py` and `scripts/analyses/other_protocol_thirds.py`.
