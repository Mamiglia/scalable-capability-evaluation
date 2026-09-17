# Direct judging implementation plan

**Status:** Planning only. Implementation starts when requested.

**Goal:** Estimate future models' HARD-item capability from 1–10 judgments of their correct EASY-item transcripts.

**Architecture:** Follow the existing pairwise protocol's separation between capability estimation, judgment collection, and Inspect scoring. Put the protocol in `src/protocols/direct_judge/`, reuse transcript reading and the judging runner, and feed predictions into the existing semester evaluation.

**Tech stack:** Existing Python, Inspect AI, pandas, NumPy, and scikit-learn dependencies. No new dependencies.

**Spec:** The agreed requirements below are the specification for this plan.

## Agreed requirements

- Use the same frozen domain tables, reference/target semesters, and EASY train-item subset as pairwise judging.
- Judge every correct first-attempt transcript on those train items, separately for each reference and target model.
- Give each transcript a capability rating from 1 to 10; preserve valid decimals without rounding.
- Average each model's transcript ratings, then fit a linear calibration from reference mean ratings to reference HARD-item θ*.
- Apply that calibration to target mean ratings. Target θ* is used only by the evaluation metric.
- Keep direct protocol code in `src/protocols/direct_judge/`.
- Use only 1–3 selected judges per semester task; do not apply the entire configured judge roster to every period.
- Keep all direct judging tests in one compact file: `tests/test_direct_judge.py`. Do not spread them across other test files or add a separate test framework.
- Show available raw reasoning in direct judgments, including when reusing pairwise judge YAMLs that disable it. Record the enabled reasoning condition in the direct log hash.
- Follow the existing model configuration, truncation, cache, retry, and resume conventions.

This document belongs to the current `small-judge-remote` checkout. The existing, empty `src/protocols/direct_judge/scoring.py` is the starting file; preserve any subsequent user edits when implementing.

## Eligibility and judgment

`Cohort.transcripts` already selects correct first-attempt transcripts from the frozen train items. Use its `(benchmark, item)` keys as the eligibility boundary. When rendering with a different reasoning-visibility setting, read those same IDs through `read_transcripts(..., require_correct=True, items=...)`; explicit item reads select the first epoch in the repeat-0 log.

There is one Inspect task per `(benchmark, model)`, covering either a reference or a target. A sample's ID is the original item ID, with no shown-order suffix. Deduplicate benchmark aliases in the same way as `JudgeBT.tasks`.

No pair intersection, reversed presentation, or one-sided-reasoning check applies. Correct answer-only transcripts remain eligible: do not copy pairwise judging's 20-character minimum into this protocol. Models with no eligible transcripts produce no judging task.

Show the question, the model's own multiple-choice options where present, and its rendered transcript. Preserve the existing handling of tool turns and hidden/redacted reasoning. Record shown character counts and whether truncation occurred. Model identity, cohort role, and θ* belong in bookkeeping, never in the prompt.

Draft system prompt:

```text
You are an expert LLM evaluator. You will see a question and one model's
transcript, which reaches the correct solution.

Infer the model's capability on harder tasks in the same domain from the
reasoning and problem-solving shown. Rate capability on a scale
from 1 (very low capability) to 10 (very high capability). Decimals are allowed. Use the same
scale across transcripts. Correctness alone does not imply high capability.

For multiple-choice questions, interpret answer letters using the options
shown with this transcript.

EXPLANATION: <Explain the capability demonstrated, in 2–4 sentences.>
ANSWER: <One number from 1 to 10.>
```

The scorer stores the rating itself, not accuracy against an artificial target. Parse the final `ANSWER:` field, allow ordinary Markdown around it, and reject non-finite values, out-of-range values, and ambiguous responses. Preserve valid decimals unchanged. A missing or malformed judgment remains missing, never zero. Record the explanation and report the valid-answer rate.

## Calibration

For model `m`, let `s_m` be the arithmetic mean of its valid transcript ratings. Each eligible transcript with a valid rating has equal weight across the domain; there is no additional benchmark, difficulty, or transcript-length weighting.

If judging uses multiple epochs, average the valid ratings for each transcript first, then average across transcripts. Retried copies of a judgment must not increase its weight.

Fit on reference models only:

```text
θ*_reference = intercept + slope × mean_rating_reference
θ̂_target     = intercept + slope × mean_rating_target
```

Use the installed `sklearn.linear_model.LinearRegression` with an intercept. Each reference contributes one row, regardless of its number of correct transcripts. Do not log-transform the ratings, standardize separately on targets, clip predicted θ, or constrain the slope.

Proposed edge-case behavior:

- Exclude references without a finite mean rating or finite θ*.
- Fewer than two usable references: return NaN for every target.
- Constant reference mean ratings: use ordinary least squares' zero-slope, mean-reference-θ* prediction.
- A target without a valid rating: return NaN for that target.
- Exclude logged rows outside the current semester, model cohort, train items, or that model's correct-transcript eligibility.

Keep `Protocol.fit(train)` / `predict(test)`. As in `PairwiseBT`, `fit` can retain the reference cohort and `predict` can obtain both cohorts' judgments on demand before fitting the calibration. This allows one resumable judging run per semester.

## File changes

| File | Responsibility |
| --- | --- |
| `src/protocols/direct_judge/protocol.py` | `DirectRegression(Protocol)`: filter eligible ratings and delegate calibration, returning target predictions. |
| `src/protocols/direct_judge/judge.py` | `JudgeDirect(DirectRegression)`: construct tasks, collect missing judgments, read and deduplicate direct logs. |
| `src/protocols/direct_judge/scoring.py` | Mean-score aggregation and linear calibration; the counterpart to `pairwise/bradley_terry.py`. |
| `src/protocols/direct_judge/__init__.py` | Export `JudgeDirect` and `DirectRegression`. |
| `src/judge/dataset.py` | Add single-transcript sample/dataset construction using the existing transcript and option rendering. |
| `src/judge/task.py` | Add a single-model Inspect task using the direct scorer. |
| `src/judge/scorer.py` | Parse direct 1–10 ratings and store numeric Inspect scores, preserving decimals. |
| `assets/system_prompt/direct.txt` | Store the direct-rating prompt. |
| `src/configs/judge.py` | Add `DirectJudgeConfig` with direct prompt defaults and condition identity. Reuse shared judge settings. |
| `src/configs/__init__.py` | Export the direct config. |
| `src/configs/domain.py` | Add optional `DomainBench.direct_system_prompt`. |
| `src/capability/evaluate.py` | Add `--direct-judge` to instantiate `JudgeDirect` from judge YAMLs. |
| `tests/test_direct_judge.py` | Focused eligibility, parsing, calibration, logging/resume, and CLI checks. |
| `README.md` | Document the method, command, missing-data behavior, and output locations. |

Keep the existing `judge.run.run_tasks`, `Model`, transcript reader, semester construction, reference freezing, IRT fits, and forecast/MSE functions. No static registry entry is needed: like `JudgeBT`, a direct judge requires a configuration and is constructed by the CLI.

Keep the Inspect scorer in `src/judge/scorer.py`; mathematical aggregation and calibration belong in `protocols/direct_judge/scoring.py`, matching the pairwise separation.

## Configuration, logs, and CLI

`DirectJudgeConfig` should reuse `JudgeConfig`'s model, generation, cache, truncation, and limit settings, with `assets/system_prompt/direct.txt` as its default prompt. Accept `include_reverse` when reusing existing YAMLs, but it has no effect on direct judging or its condition identity.

Prompt precedence is `DomainBench.direct_system_prompt`, then the direct judge's `system_prompt`. Existing per-benchmark `system_prompt` fields remain pairwise overrides. A judge YAML that explicitly supplies a prompt must supply a direct-rating prompt when used with `--direct-judge`.

Use these distinct names and locations:

```text
protocol: direct_judge_<judge-shortname>
logs:     <log_dir>/direct/<domain>/<semester>/<judge-shortname>/<condition-hash>/*.eval
task:     <benchmark>/<model>
results:  data/meta/<domain>/eval_direct_judge_<judge-shortname>.csv
```

Preserve existing pairwise names and hashes. Direct run identity must distinguish the direct protocol and relevant judging settings, including the effective prompt text and per-benchmark direct overrides. Compute the final direct hash where both judge and domain configuration are available, so changing an override cannot reuse stale completed judgments.

Each log records the protocol, domain, semester, benchmark, source model, source-log path, and effective prompt/rendering settings. The score table has:

```text
bench, judge, semester, model, item, epoch, score
```

Read only successful, unrestricted direct logs. Preserve complete item IDs, including underscores. Deduplicate retries by `(semester, benchmark, model, item, epoch)`, taking the newest attempt before dropping invalid scores; otherwise an invalid retry could silently resurrect an older rating.

Use the existing runner and completion check. A completed task must incur no further generation. Empty task lists must not start a model session. Limited smoke runs must neither satisfy completion nor enter calibration; keep them in a separate `smoke/<restriction-hash>/` subdirectory so Inspect's own resume logic cannot mistake them for full runs. Test this with actual Inspect logs.

Planned invocation, reusing the existing model configuration:

```bash
uv run evaluate configs/domain/science.yml \
  --direct-judge configs/judges/gemma4-31b.yml \
  --semester 2026-I
```

Keep the CLI's current baseline-selection behavior and allow pairwise and direct judges in the same evaluation. Missing judgments are collected on demand, so an actual invocation can make model API calls.

## Implementation sequence and acceptance checks

Use test-first implementation for each behavior below. All new checks belong in `tests/test_direct_judge.py`, reusing the repository's existing pytest setup. Group related cases into a few focused tests; the acceptance checklist below does not imply a separate test for every bullet. Run existing tests as regression checks without expanding them for this feature. All automated checks use synthetic transcripts/logs or Inspect's mock model.

### 1. Render and score one transcript

- [ ] Write a test using a correct first-attempt sample, an incorrect sample, a later correct attempt after a first-attempt failure, and a short correct answer. Only the correct first-attempt samples are eligible.
- [ ] Verify options, reasoning visibility, truncation metadata, unchanged item IDs, and absence of model identity/θ* in rendered samples.
- [ ] Write parser/scorer cases for 1, 7.5, 10, Markdown formatting, quoted earlier answers, absent output, and invalid values such as 0, 11, NaN, infinity, and `8/10`. A malformed final answer must not fall back to an earlier answer.
- [ ] Run the new tests and confirm they fail because the direct behavior is absent.
- [ ] Implement the direct scorer, dataset/task factories, and prompt. Empty datasets return no task.
- [ ] Run those tests plus the existing pairwise dataset, scorer, and task tests.

### 2. Collect, resume, and calibrate

- [ ] Add synthetic cohorts where references and targets solved different train items. Verify one task per benchmark/model, including items solved by only one model, and deduplication of benchmark aliases.
- [ ] Add a calibration case with reference means `[2, 4, 6]`, reference θ* `[0, 2, 4]`, and target mean `8`. Expected target θ̂ is `6`. Altering target truth must not alter that prediction.
- [ ] Cover missing reference/target ratings, fewer than two references, constant reference ratings, repeated judging epochs, and foreign/ineligible log rows.
- [ ] Round-trip a direct task through Inspect's mock model and `.eval` logs. Test successful reads, malformed responses, duplicate attempts, complete-run reuse, and restricted-run isolation.
- [ ] Run the checks before implementing the corresponding behavior.
- [ ] Implement the direct config, domain prompt override, score reader, protocol classes, and task orchestration using `run_tasks`.
- [ ] Rerun these checks and the existing pairwise protocol tests.

### 3. Expose and document the protocol

- [ ] Test parsing an existing judge YAML through `--direct-judge`, direct prompt defaults/overrides, and simultaneous pairwise/direct protocol names.
- [ ] Confirm changing a direct prompt or condition selects a new log directory while existing pairwise paths remain identical.
- [ ] Verify the normal forecast/evaluation path writes target predictions and computes the expected MSE from synthetic data.
- [ ] Implement the CLI option and package exports, then update the README with the agreed methodology and command.
- [ ] Run the focused and regression suites, review the diff, and report the results before any live judging run.

Suggested verification commands from this checkout, with the chosen project environment active:

```bash
PYTHONPATH=src python -m pytest tests/test_direct_judge.py -q
PYTHONPATH=src python -m pytest tests/test_pairwise.py tests/test_protocols.py \
  tests/test_dataset.py tests/test_scorer.py tests/test_task.py -q
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m capability.evaluate --help
```

The current PyCharm interpreter is the existing `small-judge/.venv/bin/python`. Before implementation, record the baseline in the selected environment: previous checks encountered Python 3.14/Draccus compatibility errors and sandbox restrictions on multiprocessing. Distinguish those existing failures from regressions; do not silently change shared environment packages.

## Completion criteria

- [ ] Correct train transcripts receive independent 1–10 judgments for references and targets.
- [ ] Mean ratings calibrate exclusively against reference HARD-item θ* and yield target θ̂ through the existing evaluation pipeline.
- [ ] Invalid/missing ratings remain missing; duplicate or restricted runs do not bias model means.
- [ ] Completed judgments are reused and direct prompts/logs cannot be confused with pairwise runs.
- [ ] The CLI, documentation, and focused checks cover the complete synthetic workflow.
