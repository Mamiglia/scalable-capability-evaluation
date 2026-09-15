# Direct judging under transcript truncation (challenge 3)

Does capability prediction survive deterministic prefix truncation of agentic transcripts? Subject
rollouts, the EASY/HARD split, θ* and the calibration cohort stay fixed; only judgments and forecasts
are recomputed with `JudgeConfig.max_length` (characters kept per transcript, applied to calibration
and target transcripts alike). Cyber and coding only.

## Conditions

`<domain>/<judge>/<semester>/<max_length>`; two limits and one judge per semester:

| Domain | Judge | Semester | Limits (chars) | Retained inputs | Truncated at 2,000 / 8,000 |
| --- | --- | --- | --- | ---: | ---: |
| cyber | Qwen3-235B-A22B | 2025-II | 2,000; 8,000 | 985 | 878 / 416 |
| cyber | DeepSeek-V3.2 | 2026-I | 2,000; 8,000 | 1,155 | 985 / 328 |
| coding | Qwen3-235B-A22B | 2025-II | 2,000; 8,000 | 779 | 590 / 269 |
| coding | DeepSeek-V3.2 | 2026-I | 2,000; 8,000 | 1,154 | 915 / 331 |

2,000 characters cuts most transcripts (question plus the opening moves survive); 8,000 cuts the
long tail (roughly a third to a half of inputs). Retained sets, exclusion masks (context preflight
of the full-transcript campaign) and reference cohorts are exactly the published full-transcript
campaigns'. Inputs shorter than the limit render identically and reuse the full-transcript rating;
every truncated input is verified to be a prefix of the full one.

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

`prepare.py` freezes inputs (no API calls) → `run.py` collects (10 retries baked in; `--max-connections`
override, SiliconFlow served the baseline at 64) → `audit.py` verifies inputs, prefixes, pins, served
providers and reuse provenance against the baseline logs → `recollect.py` / `consolidate.py` retry
unusable judgments → `prepare_publication.py` stages the condition under `publication/direct/...`
for the `logs/` repo → `compare_truncation.py` writes the matched full-vs-truncated table to
`comparison/`. `sweep_status.py` reports progress; `self_check.py` exercises `run.py` without API calls.
Only logs with status success count anywhere (eval_set leaves error logs beside successful retries).
