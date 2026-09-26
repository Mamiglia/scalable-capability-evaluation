# Pairwise judging on transcripts with missing thirds

How does the pairwise judge degrade when parts of every transcript are missing? Each rendered transcript
(reasoning shown) is split into three equal character spans and the judge sees one of the 7 non-empty subsets;
`{1,2,3}` is the reasoning-on baseline already collected (conditions `24d6cc32` / `66c170cd`), the other 6 are new.
Every dropped run of thirds is shown as `\n...\n` and the user prompt tells the judge that parts may be redacted
(`judge/dataset.py THIRDS_NOTE`; system prompts are unchanged, since MCQ benches override them per bench).
Domains science, math, coding, cyber x semesters 2025-II, 2026-I; the top reference judge only
(`data/meta/<domain>/fit.yml top[0]`), on exactly the baseline's items (science/math 2026-I: the 40-item subset).

## Conditions

A condition is the baseline's `configs/judges/reasoning/<stem>.yml` plus `evaluate --truncate thirds-<kept>`
(`JudgeConfig.truncate`, validated by `configs/judge.py TRUNCATE_MODES`). `condition_hash` appends `truncate`,
so each kept-set has its own dir next to the baseline under `logs/judge/<domain>/<semester>/<judge>/`:

| kept | full-set hash (tpq 0) | 40-item hash (tpq 10) |
| --- | --- | --- |
| 1 | 66f9615b | 8311027f |
| 2 | 574946b9 | f2b0de83 |
| 3 | e4d05a32 | 04e3f7ec |
| 12 | b8f67d26 | e5f74d2a |
| 13 | 9d38c63d | 4e2e8bc9 |
| 23 | 97b40582 | 03c21282 |

| domain | semester | judge | baseline | YAML stem (`runs.py RUNS`) |
| --- | --- | --- | --- | --- |
| science | 2025-II | minimax-m1 | 24d6cc32 | `minimax-m1` |
| science | 2026-I | kimi-k2-thinking | 66c170cd | `kimi-k2-thinking` |
| math | 2025-II | deepseek-r1-0528 | 24d6cc32 | `deepseek-r1-0528` |
| math | 2026-I | kimi-k2-thinking | 66c170cd | `kimi-k2-thinking` |
| coding | 2025-II | minimax-m1 | 24d6cc32 | `minimax-m1` |
| coding | 2026-I | minimax-m2 | 24d6cc32 | `minimax-m2-full` |
| cyber | 2025-II | deepseek-r1-0528 | 24d6cc32 | `deepseek-r1-0528` |
| cyber | 2026-I | deepseek-v3.2 | 24d6cc32 | `deepseek-v3.2-full` |

Pairs whose kept third leaves under `MIN_RESPONSE_CHARS` are dropped by `build_dataset`, so a thirds run judges a
subset of the baseline's samples; `compare.py` scores every condition on the rows judged in all of them.

## Scripts

```bash
.claude/runs/launch_thirds.sh <domain> <semester> <stem> <kept...>      # detach with setsid; wave 1 = 12 13 23, wave 2 = 1 2 3
PYTHONPATH=src .venv/bin/python src/challenges/thirds/audit.py <domain> <semester> [--inputs]
PYTHONPATH=src .venv/bin/python src/challenges/thirds/compare.py --recompute   # -> analysis/judge/thirds.{csv,png,svg}
```

`audit.py` checks every log succeeded with the right metadata, every sample id exists in the baseline dir and counts
unscored verdicts; `--inputs` rebuilds every prompt from `data/` and compares byte for byte. Push finished runs with
`.claude/runs/push_judge_logs.sh judge "<message>"`.

## Cost

From the baselines' token totals: input scales with the kept fraction, output does not. Baseline runs sum to
~$278 input + ~$151 output; wave 1 (three 2/3 conditions) ≈ $1,000, wave 2 (three 1/3 conditions) ≈ $730.
