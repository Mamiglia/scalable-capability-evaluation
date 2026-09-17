# SDD ledger — plan: docs/superpowers/plans/2026-09-15-results-analysis.md
Spec: docs/superpowers/specs/2026-09-15-results-analysis-design.md (read).
Workspace ruling: work in place on branch `methodology` (a feature branch, not main); a worktree
would strand the untracked draft files and the user's other uncommitted work. Cost if wrong: none
beyond ordinary commits on the user's branch.

## Pre-flight conflict scan
| Pair / task | Produces vs consumes | Finding |
| --- | --- | --- |
| T1 -> T4 | tables.load_responses(domain) columns bench,item,model,repeat,correct | T4 uses exactly those; ok |
| T2 -> T3 | JudgeRun(protocol, judge, condition, harness, table, missing) | T3 mutates table, missing; ok |
| T2 -> T6 | JudgeRun.table, .judge, .condition, .missing | ok |
| T4/T5 -> T6 | AccuracyCalibration(dom, "train"/"hard"), ReferenceMean(), EmbeddingRegression() | names match |
| T6 -> T7 | COLUMNS includes judge_theta, missing | T7 judge_summaries uses judge_theta; test rows carry it; ok |
| T7 -> T8 | stats.FLOOR, ORACLE, JUDGED, core_condition | T8 imports those names; ok |
| T7 -> T9 | stats.scores, stats.summary; judge labels "most capable judge"/"mean over judges" | T9 filters on those; ok |
| T8/T9 | figures read scores.csv columns domain, semester, protocol, judge, condition, mse | ok |
| T1 self | deletes src/analysis/run.py + pyproject entry | consistent with T9 runner in scripts/ |
| T4 self | test monkeypatches tables.load to return (None, items) with s_set; oracle merges on bench,item | consistent |
| T7 self | toy_forecasts direct rows: hidden-condition rows only for "strong"; core_condition test expects direct hidden-only set when shown rows removed | consistent |
| T8 self | series() applies core_condition to per-judge scores rows lacking judge_theta; core_condition uses only domain, semester, protocol, judge, condition | ok |
| T9 | runner uses exec-replace for the smoke; fine |
Scan clean.

Note: plan file lost Tasks 7 (steps 4-5), 8, 9 and the self-review during a controller edit; restored verbatim from the original before any brief for those tasks was dispatched. Briefs 1-9 extracted.
Dispatch: Tasks 1-3 batched to one implementer (sonnet), BASE 8221667.
Incident: the Tasks 1-3 implementer ran an unscoped `git -C logs lfs pull` (37 GB + 19 GB cache) and a
cleanup in data/ that left 9,314 tracked files deleted; the laptop crashed. Agent stopped. Recovered:
logs/.git/lfs/objects dropped, legacy/ and legacy_2/ reverted to pointers (logs now 8.7 GB), data/
restored to a clean tree; frozen transcript logs are mostly pointer stubs again and need a scoped pull.
Tasks 1 and 2 committed (8523988, 10177b9); Task 3 file matches its brief, check not yet run, uncommitted.
Ruling: transcript logs live on the T7 volume (/Volumes/T7/small-judge-remote/data, a pointer clone of
LASR-G5/benchmarks at c199498b with only the frozen reference/target logs pulled); src/reference/tables.py
gains a SMALL_JUDGE_DATA env override for DATA, META stays at data/meta — because the user asked for pulls
to go to T7, not the laptop disk. Cost if wrong: one env var and a two-line change to revert.
Task 3: complete (matching.py verbatim from brief; check on science 2026-I from T7 passed; committed by the
controller after the implementer was stopped — review folded into the final whole-branch review).
Ruling: merged origin/calibrated-bt (teammate's JudgeCalibrated + JudgeLinear) into methodology, 257d212 —
the user's new pairwise conversion lives there; cost if wrong: one merge commit to revert.
User decisions 2026-09-15 late: conditions are (1) intercept baseline, (2) pairwise via JudgeLinear or
JudgeCalibrated (toggle, pick by reference LOO error), (3) direct via OLS on mean rating, raw or per-item
z-scored on reference stats (toggle), (4) glm_regex ridge, (5) embedding ridge from HF LASR-G5/transcript-embeddings
(.npz per log, full/answer). Judge selection: lowest MSE on reference models (reference pairs are judged), plus
mean over judges. Spec and plan Tasks 4-9 to be rewritten; embeddings pulled to T7.
Plan rewritten 2026-09-15 late: Tasks 4-10 (baselines, embeddings from HF npz on T7, judges.py conversions,
calibration with variants, stats with paired metrics, plots = forecast dot plot / bars / semester tables,
runner). Spec revised to match (conditions, metrics, outputs). Data on T7: transcripts (455 logs) and
embeddings (1,820 npz) pulled and verified.
Ruling: the z metric is the paired-bootstrap z of (condition MSE − intercept MSE) over resampled targets —
the user said "might need to think this through"; cost if wrong: one function (paired_metrics) to change.
Task 4: complete (cc35bba). Task 5: complete (9b526b4). Reviews folded into the whole-branch review.
Task 6: implementer BLOCKED — plan defect: the toy LOO test used 4 references, leaving the linear conversion 3 pairs per fold.
Ruling: the toy uses 8 evenly spaced references (checked: max LOO error 0.30 linear, 0.21 calibrated, tolerance 0.8);
plan and brief patched; implementer resumed. Cost if wrong: one test line.
Controller wrote src/analysis/stats.py and plots.py from the plan text (untracked) to preview the user's plots on the
old science smoke forecasts; fixed there: err_condition/err_baseline column names, chronological semesters, x-label
rotation, sorted table index — plan and briefs 8/9 patched to match. Tasks 8-9 implementers verify and commit them.
Tasks 6-7: implementer DONE (4ea0893, c3e762c). Review package for Tasks 4-7 (4092231..c3e762c) dispatched.
Science end-to-end run OK with the on-disk stats.py/plots.py/results.py (Task 10 step 2 satisfied early): outputs in
analysis/results/. Observations for the user: glm_regex 2025-I MSE 38 (LOO ridge picks a tiny alpha; pre-existing
protocol behaviour), direct science has 95 unrated transcripts, best-judge-on-references picks a poor direct judge.
Tasks 4-7: complete (commits 4092231..c3e762c, review clean). ⚠️ items resolved by the controller: forecast() columns
and JudgeRun fields are exercised by the successful science end-to-end run.
Tasks 4-7: minor (deferred): mid-file `from analysis import judges` import in tests (plan-mandated); no per-method
docstring on EmbeddingRegression.__init__ / Intercept.__call__; prior computed needlessly for the linear LOO.
Dispatch: Tasks 8-10 to one implementer (sonnet), BASE c3e762c; full four-domain run already in progress (controller).
Four-domain run complete (analysis/results/). Pooled MSE: intercept 1.42, oracle 0.44, embedding 1.08-1.12,
glm_regex 44.8 (math 2025-I = 308, unstable LOO ridge), direct best-judge 1.49 (1,579 unrated transcripts across
domains: coding ~880, cyber ~500, math ~220, science 95), pairwise linear 0.69 / calibrated 0.42.
Task 8: implementer BLOCKED on a plan defect (summary test indexed a non-unique (protocol, judge)); Ruling: the
test restricts to condition in {"-", "core"} first; plan, brief, and the on-disk test patched by the controller.
Ruling: runner defaults PAIRWISE_CONVERSION="calibrated", DIRECT_NORMALISATION="raw", chosen by mean reference LOO
error (calibrated 0.34 < linear 0.45; raw 0.43 < item_z 0.59), not target MSE; plan/brief 10 patched. Cost if wrong:
two constants.
Tasks 8-10: complete (43e0d31, 891cd6a, 2ef3a93); reviews folded into the whole-branch review.
User 2026-09-15 late: bar plot must not stretch for the regex outlier (zigzag break), and axes touch (no spine
offset) on every figure — recorded in ~/.claude/skills/nature-figures/SKILL.md. Task 11 dispatched (sonnet), BASE 2ef3a93.
Preview pipeline re-run on the four-domain table (scripts/analyses/preview_plots.py -> analysis/preview/).
Task 11: fix round 1/5 — implementer BLOCKED (legend overlapped rotated tick labels in the two-panel branch;
tight_layout(rect) incompatible with the gridspec). Ruling: layout="constrained" + fig.legend(loc="outside lower
center"), no tight_layout/bbox_to_anchor; np.ptp instead of ndarray.ptp (NumPy 2.x). Plan/brief patched. Cost if
wrong: a few layout lines.
Task 13 (baseline level in forecast plots) added to the plan; dispatch after Task 12.
Task 14 (judge ensemble conditions) added to the plan; order after 11: 12 -> 13 -> 14.
Task 11: complete (259efcd, fix round 1 addressed; review folded into the whole-branch review). Implementer reported
plots.py overwritten mid-task by an unrelated rewrite from outside this session (a parallel session is active:
untracked items_curve.py, configs/judges/truncated/); it rebuilt from HEAD. User warned.
Dispatch: Tasks 12-14 (rename, baseline in forecast plots, ensembles) to one implementer (sonnet), BASE 259efcd.
Task 15 (point styling, 95% CIs, per-semester bars) added; dispatch after 12-14.
Tasks 12-14: complete (3848985, 4b81f89, 9197fd0); reviews folded into the whole-branch review.
Direct-judge diagnosis for the user: calibration reproduces protocols.direct_judge.scoring.calibrate exactly; the
signal is compressed (reference ratings sd 0.07-0.19 on 1-10) so OLS slopes of 2-8 theta/point amplify noise, and
best-judge-on-references anti-selects (lowest LOO = most compressed judge). Ensembles/mean over judges preferred.
Dispatch: Task 15 (sonnet), BASE 9197fd0.
Task 15: complete (9c133da). Full pipeline re-run started; whole-branch review package 8221667..HEAD.
Final pipeline run complete: pooled MSE easy-item 1.42, oracle 0.44, embedding 1.08-1.12, direct best 1.49 / pooled judges 1.01, pairwise best 0.42 / pooled judges 0.40 (best condition). glm_regex 44.8 (math 2025-I 308).
Whole-branch review (opus): Needs fixes. Critical: direct_runs not restricted to the top-3 roster (math 2024-II gets a
4th judge that wins best-on-references). Important: pairwise "NaN without evidence" docstrings false (prior mean is
returned); scores blanks lo/hi for every core row instead of MEAN only. Minors listed in the review; deferred-minor
triage: all three "leave".
Ruling: pooled z resamples targets as independent (spec-fixed definition) — a caveat for the results text, not a code
change. Ruling: judge_loo_mse uses the other references' prior, not the frontier prior — documented, not changed
(judge ordering unaffected).
Fix wave (one dispatch, sonnet), FIX_BASE 9c133da: Critical + 2 Important + minors 1-8 + controller's own: the
upper panel of the broken bar plot must draw only the off-axis points (and bars whose mean lies there).
Fix wave: dd1d4b0, 49d0070, 18e315a, f41db0e; scoped re-review 9c133da..HEAD dispatched; final pipeline re-run started.
Final pipeline run after the fix wave complete.
Whole-branch review + fix wave: clean (re-review: all 12 addressed). Handed over; workspace deleted.
