# Direct judging under transcript truncation (challenge 3)

Does capability prediction survive deterministic prefix truncation of agentic transcripts? Subject
rollouts, the EASY/HARD split, θ* and the calibration cohort stay fixed; only judgments and forecasts
are recomputed with `JudgeConfig.max_length` (characters kept per transcript, applied to calibration
and target transcripts alike). Cyber and coding only.

## Conditions

`<domain>/<judge>/<semester>/<head|tail>-<max_length>`: 4,000 characters, head (first 4,000) and tail
(last 4,000), best full-transcript judge per domain and semester. Frozen inputs, logs and results live
under `artifacts/direct-judging-truncation/`.

| Domain | Semester | Judge | Retained inputs | Cut at 4,000 |
| --- | --- | --- | ---: | ---: |
| cyber | 2025-II | Qwen3-235B-A22B | 985 | 664 |
| cyber | 2026-I | Kimi-K2-Thinking | 1,231 | 723 |
| coding | 2025-II | Qwen3-235B-A22B | 779 | 404 |
| coding | 2026-I | DeepSeek-V3.2 | 1,154 | 578 |

Retained sets, exclusion masks (context preflight of the full-transcript campaign) and reference
cohorts are exactly the published full-transcript campaigns'. Inputs shorter than the limit render
identically and reuse the full-transcript rating; every cut transcript is verified to be a prefix
(head) or suffix (tail) of the full one. Published under `direct_truncated-<head|tail>-4000/` in the
judging repo. Pairwise truncation is under `judge_truncated-<head|tail|ends>-2000/` (see below). `artifacts/.../superseded-char-limits/` keeps an earlier head-only
2,000 / 8,000 freeze that predates this design.

## Baselines

- cyber: `artifacts/direct-judging-cyber/` (`LASR-G5/benchmarks@29a4ee20`), 102583f8 logs.
- coding: the published latest refit = SWE-bench mini / Terminal-Bench 2.1 campaign
  (`artifacts/direct-judging-coding/`, staged at `e3d86809`, filtered to the latest train set) plus
  the AgentBench OS add-on (`artifacts/direct-judging-coding-agentbench/`, `c199498b`), calibrated on
  the add-on's tables. `staging/coding/` is a symlink tree joining both staged transcript sets
  (`data/` holds LFS pointers). One retained SWE/Terminal source changed between the two revisions
  (qwen3-30b-a3b on Terminal-Bench, inline `<think>` normalisation); its full rating was judged on
  the earlier text, so the truncated prefix is taken from that same earlier text.

## Scripts

Run as `PYTHONPATH=src .venv/bin/python src/challenges/truncation/<script>.py <condition>`.
`prepare.py` freezes inputs (no API calls) → `run.py` collects (10 retries baked in; `--max-connections`
override, SiliconFlow served the baseline at 64) → `audit.py` verifies inputs, prefixes, pins, served
providers and reuse provenance against the baseline logs → `recollect.py` / `consolidate.py` retry
unusable judgments → `prepare_publication.py` stages the condition under `publication/direct/...`
for the `logs/` repo → `compare_truncation.py` writes the matched full-vs-truncated table to
`comparison/`. `sweep_status.py` reports progress; `self_check.py` exercises `run.py` without API calls.
Only logs with status success count anywhere (eval_set leaves error logs beside successful retries).

## Pairwise truncation

Twelve conditions collected 2026-09-15/16 with the stock pairwise path, no separate harness:
`uv run evaluate configs/domain/<domain>.yml --protocol chars --judge configs/judges/truncated/<judge>-<mode>-2000.yml --semester <semester>`,
2,000 characters per transcript (two transcripts per prompt, so the same ~4k judge budget as direct),
head, tail and ends. The frozen `<semester>_train` column fixes the items, so the truncated runs
share every pair and item with the reasoning-on full-transcript baselines (`logs/judge/<domain>/...
/24d6cc32`, full train set). Judges: MiniMax-M1 (coding 2025-II) and Kimi-K2-Thinking (coding 2026-I),
the best pairwise judges by baseline MSE; Qwen3-235B-A22B (cyber 2025-II) and Kimi-K2-Thinking
(cyber 2026-I), matching the direct truncation judges. Logs and per-mode READMEs with the MSE table
are in the judging repo under `judge_truncated-<mode>-2000/`.
