---
name: judge-logs
description: Handling the logs/ repo of judge runs - layout, where a judging campaign's verdicts land, resuming, moving stale/invalid verdicts, and committing/pushing to the generality-labs/judging HF repo. Use whenever the user works with logs/, judge eval-sets, verdict logs, or rankings - launching or resuming a judging run, finding which dir a run wrote to, cleaning up after data/ transcripts changed, or syncing logs/.
---

# Handling logs/ (judge runs)

**Read `logs/CONTRACT.md` first** — it is the binding rulebook (layout, fixed areas,
quarantine, sync). This skill is the operational companion. `logs/` holds *derived* data:
every verdict is recomputable from `data/` + `configs/judges/` (judge at temperature 0,
generate cache on), which is why the repo is gitignored in the parent and pinned nowhere.

## Where things are

- Current runs: `logs/judge/<domain>/<semester>/<judge shortname>/<condition hash>/`, one `.eval` per `<bench, reference, target>` (`bench/reference|target` is the
  task name). One dir per `<domain, semester, judge, condition>`, written by `JudgeBT`.
- To find a run's dir, don't guess the hash — `JudgeConfig.condition_hash` (`src/configs/judge.py`)
  covers system_prompt, include_reasoning, include_reverse, max_length, temperature; or just
  `ls -t logs/judge/<domain>/<semester>/<judge>/`.
- `logs/rank/`, `logs/pilot*`, `logs/aime` = older campaigns; `scripts/judge/import_judge_logs.py <domain.yml>`
  copies the (reference, target) pairs among them into `logs/judge/` as `(imported)` tasks.
- `pilot*/`, `aime/`, `apex/` are frozen pre-08-26 experiments; `smoke/` is scratch;
  `legacy/` = superseded, `invalid/` = must-not-use (see its README).

## Running / resuming

```bash
uv run evaluate configs/domain/<domain>.yml --judge configs/judges/<judge>.yml [--semester 2026-I]  # judges what is missing, then scores
```

Re-running the same `evaluate --judge` resumes the same `eval_set` dir; a refrozen roster adds
pairs to it. Never hand-create condition-hash dirs, never mix conditions in one dir.
Re-judging an unchanged pair is free (generate cache).

## When data/ changes under the verdicts

Canonicalization, rescoring, or normalization in `data/` makes verdicts on the old
transcripts **stale**: move them to `logs/legacy/<mirrored path>` (never delete), re-judge,
and say so in the commit message. Unusable verdicts (serving defects) go to `logs/invalid/`
*plus a line in `invalid/README.md` explaining why*.

## Syncing

`logs/` is its own git repo → `https://huggingface.co/datasets/generality-labs/judging`, git-LFS on
`*.eval`. Git is the only transport (no `hf upload`). No parent pin exists, so an unpushed
commit lives only on this machine: commit + push after every judging campaign and every
legacy/invalid move. Commit messages name the campaign or the move and its reason.

## Hygiene

- Junk to flag: nested `inspect_evals/...` dirs (wrong `--log_dir`), stray test dirs.
- Findings cite the experiment config + condition, never a log path.
- Before bulk-reading logs use the `reading-logs` skill; never unzip a `.eval`.
