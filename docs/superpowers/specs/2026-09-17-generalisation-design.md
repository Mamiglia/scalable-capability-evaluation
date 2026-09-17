# When does the capability signal generalise (research question 3)

**Status:** design, agreed in chat 2026-09-17, awaiting review. Nothing here launches judging or
collection. Every analysis reads the existing forecasts, judge runs and transcripts, except the
closed-source targets, which wait on the closed-frontier coding campaign the user is running.

**Question.** Protocols are calibrated on earlier open-weight models in one domain. Does that
calibration still hold when the targets move away from the references: later in time, further above
the references' capability, across the switch to reasoning models, to closed-source models, and to
another domain?

## Conventions shared by every analysis

The four paper protocols, with the variants figure one uses: Deterministic features (`glm_regex`),
Embedding (`embedding`, `full`), Absolute scoring (`direct`, `raw`), Pairwise judge (`pairwise`,
`linear`). Judge protocols use the best judge on references for each task
(`analysis.stats.one_per_target`). The baseline is the saturated-item prediction (`easy_items`, the
references' mean θ\*). Length is left out.

The prediction score is `signal_glm.score`, 100 · (1 − error / baseline error). Squared errors are
pooled before a score is taken, never scored one target at a time: 6 of the 112 existing
(target, domain) forecasts have a baseline squared error under 0.05, where a single target's score
runs to minus thousands. Per-task numbers pool over the task's targets; per-model numbers pool over
the model's forecasts, as figure one pools each model over its four domains.

Every figure writes the csv it plots. Figures are drafts in the existing conventions (paper palette,
domain marker shapes: science circle, math square, coding triangle, cyber diamond) for the user to
restyle.

Code: the reshaping of existing forecasts (time, distance, reasoning targets) in
`src/analysis/challenges/generalisation.py`; the new cross-domain forecasts in
`src/analysis/challenges/cross_domain.py`; one cell runner `scripts/analyses/generalisation.py`.
Outputs under `analysis/results/generalisation/` and `analysis/results/figures/generalisation/`.

## Prediction score over time

All tasks in `analysis/results/forecasts.csv`: science and math have five target semesters
(2024-I to 2026-I), coding and cyber two (2025-II, 2026-I). For each domain, task and protocol, the
prediction score from the task's 8 targets' pooled squared errors, and the existing timeline test
(paired t-test of the 8 targets' squared errors against the saturated-item prediction's).

Figure: one panel per domain, x = target semester, y = prediction score, one line per protocol.
`plots.timeline_figure` already draws this with metric "pct" (MSE as a percentage of the baseline's,
i.e. 100 minus the score); add a score metric to it rather than a new figure function.

For the text: before 2025-II the candidate judges are older models (GPT-3.5, GPT-4, Mixtral in
2024-I; GPT-4 Turbo, GPT-4o, Qwen2-72B in 2024-II), mostly with reasoning hidden, so a trend over
semesters also reflects judge strength.

## Error against distance from the references

Every target in every task (112 target-domain forecasts per protocol). Two distances per target,
read from the frozen tables: θ\* minus the strongest reference's θ\*, and θ\* minus the mean
reference θ\*. The y variable is the signed error θ̂ − θ\*, so a negative value is an under-forecast.

Figure: one row per distance, one panel per protocol, protocol colour, domain marker shapes. The
saturated-item prediction's signed error is exactly minus the distance from the mean reference, so
the mean-reference row draws the line y = −x for it.

Summary per protocol and distance: the OLS slope of signed error on distance with a 95% bootstrap
interval over target-domain rows. A slope of 0 means the error does not grow as targets move away;
−1 means the protocol does no better than the saturated-item prediction. Targets within a task share
references, so the interval is optimistic; it is reported, not tested.

## Reasoning-model targets in 2025-I

Math and science 2025-I. The references (the 2024-II roster) are all non-reasoning models. Expected
split of the targets: reasoning = deepseek-r1-0528, minimax-m1, qwen3-235b-a22b, qwen3-30b-a3b,
qwen3-32b; non-reasoning = deepseek-v3-0324, gemma-3-27b-it, mistral-small-3.2-24b-instruct. The
Qwen3 models can run with thinking on or off, so the split is confirmed from the transcripts: a
target counts as reasoning when its correct train transcripts carry a reasoning channel.

Per domain, protocol and group: each target's signed and squared error, and the group's prediction
score from pooled squared errors. The groups have 5 and 3 models, so no test; every model is drawn.

Figure: `plots.task_figure` with `reasoning=` already draws the task (reasoning targets as open
diamonds); beside it, per protocol, the two groups' signed errors as per-model points with the group
mean. The draft caption's "before and after reasoning models" comparison is the 2024-II and 2025-I
points of the time figure.

## Transcripts from one domain, calibrated on another domain's θ\*

Can transcripts from one domain forecast capability in another, when the references' capability in
that other domain is known? Semesters 2025-II and 2026-I, the only ones with all four domains:
16 targets, 16 ordered pairs of (transcript domain, scored domain).

For each semester and pair, build the task of the transcript domain with `Task.from_semester`, then
replace every θ\* in it with the scored domain's θ\* for the same semester: the references' and the
historical models' θ\* (what protocols fit to) and the targets' truth (what is scored). Historical
models without a θ\* in the scored domain are dropped. Items, transcripts, judge runs and their
matching stay the transcript domain's. `calibration.forecasts` then runs unchanged:

- Pairwise judge: verdicts on transcript-domain items, references pinned at their scored-domain θ\*
  in the linear conversion.
- Absolute scoring: OLS of the references' scored-domain θ\* on their mean transcript-domain rating.
- Embedding and Deterministic features: regressions from transcript-domain features to
  scored-domain θ\*, fitted on references and history.
- The best judge is picked by reference leave-one-out error recomputed against the scored-domain
  θ\*, so no target label enters the choice.
- The saturated-item prediction is the references' mean scored-domain θ\*, identical to that
  domain's own baseline. Every cell in a column shares its baseline, and the diagonal reproduces the
  per-domain forecasts in `forecasts.csv`. The runner asserts this match.

Output: `cross_domain_forecasts.csv`, the `forecasts.csv` columns with `domain` as the transcript
domain plus `scored_domain`.

Figures:

1. One 4 × 4 heatmap per protocol. Rows are the transcript domain, columns the scored domain, each
   cell the prediction score from its 16 targets' pooled squared errors. The diagonal is figure one's
   data split by domain.
2. Protocol comparison for math or science transcripts scored on coding or cyber θ\*. For each model,
   squared errors pooled over those four cells give one prediction score; one box per protocol with
   per-model circles and the one-sample t-test against 0 over the 16 models, as in figure one
   (`model_box_figure`). Beside each protocol, the same models' score from their own coding and cyber
   transcripts (the coding and cyber diagonal cells), so the loss from changing domain is visible.
   The csv also holds the scores pooled over all 12 off-diagonal cells.

Runtime needs the T7 volume (`SMALL_JUDGE_DATA`, `SMALL_JUDGE_EMBEDDINGS`) and the pulled judge logs
that `scripts/analyses/results.py` already reads.

Dropped: scoring a transcript-domain forecast directly against scored-domain θ\* with no
recalibration. θ scales differ between domains (2025-II reference mean θ\*: coding −1.15, science
0.22), so that error would mostly measure the offset between scales.

## Closed-source targets

Waits on the closed-frontier coding campaign
(`docs/superpowers/specs/2026-09-16-closed-frontier-coding-design.md`). When its data lands: coding
2026-I, the 8 open and 6 closed targets, unchanged references. For the protocols judged on closed
targets (Pairwise judge with Kimi-K2-Thinking, reasoning shown on 100 items and hidden on 40;
Pairwise judge with GPT-5.2 on 40 items; Absolute scoring), the prediction score pooled over the
open targets and over the closed targets separately, filling the draft's open/closed table. With 6
closed models, no test; per-model points. Embedding and Deterministic features for closed targets
need those transcripts embedded and featurised, which the campaign does not include; they are left
out of this table unless that run is added.

## Removed from the draft

The within-domain table: figure one and the signal GLM already report each protocol's score per
domain, and the table's numbers predate the current feature set.
