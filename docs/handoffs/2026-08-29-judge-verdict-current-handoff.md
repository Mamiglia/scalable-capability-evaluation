# Handoff — judge-verdict interpretation

Date: 2026-08-29
Repo: `/Users/moriohamada/PycharmProjects/ai_safety/small-judge`
Branch: `judge-dissect` only; never push to `main`.

## Latest update — 2026-08-29 13:54 BST

This section supersedes the stale figure and RunPod status below.

- Figure cleanup is complete in local commit `80d1a53` (`Polish judge
  contribution figures`). The push to `origin/judge-dissect` was blocked by the
  environment's remote-trust safeguard, so the commit is local only.
- Fourteen pooled/overview/source-block/secondary PNGs and six per-judge PNGs
  were rendered from the canonical 117-feature matrix and manually inspected.
  Their SVG partners retain editable text. Non-symmetric axes now use endpoint
  ticks only, all judge figures share the same ten feature rows, and the
  misleading `bare_answer` display label is now `minimal non-answer prose`.
- SWC lesion-cell job `3418867` completed successfully. Validated local outputs
  are under `judge_results/verdict_interpretability/readable-eda-20260828/lesion-cells/`:
  6,550 summary rows and 1,238,736 item-cell delta rows with manifest hashes
  verified. All six judge figures are populated under `figures/models/by-judge/`.
- The readable frequentist fit is complete: 16/16 folds, 113,951 strict
  new-model-by-new-item OOF predictions, accuracy `0.7256891`, log loss
  `0.5420933`, final ridge iterations at most 59, and 93–116 selected features
  per fold. The 113,951 count is correct for the configured six-judge strict
  outer-fold test roles; 120,200 is the larger processed verdict population.
  All 112 fit artifacts and four worker logs are downloaded locally.
- RunPod pod `jizl5l8gv3hol6` is stopped (`runtimeStatus=stopped`), reducing
  account spend from about $0.45/hour to $0.011/hour. Permanent deletion was
  blocked pending explicit approval because it destroys ephemeral storage.
- New heavy work should default to SWC/HPC. Expanded SWC fits remain active;
  latest counts were readable+surprisal 6/16, readable+SAE 5/16, and
  readable+both 5/16. Next: wait for all 48 outputs, validate/download them,
  render the five-variant comparison, then review/stage PC-L1 on SWC.

## Immediate priority

The current figures are not acceptable. They are visually cluttered, have gone
through too many incremental revisions, and still do not cleanly reproduce the
capability-GLM presentation. Do not treat the current PNG/SVG outputs as final.

The first next task is a manual visual cleanup that directly follows
`capability_predictions_glm/contributions.py`:

- use `Singular`, `Unique`, and `Coefficient` columns;
- produce a compact overall feature summary;
- produce one three-column figure per source block;
- produce a pooled benchmark grid with benchmark rows;
- produce one figure per judge with that judge's available benchmarks as rows;
- use identical selected feature rows within comparable panels;
- keep the scatter secondary only, with tight independent axes and readable,
  outward-pointing labels;
- organize outputs under:
  - `figures/models/overview/`
  - `figures/models/by-feature-group/`
  - `figures/models/by-benchmark/`
  - `figures/models/by-judge/`
  - `figures/models/secondary/`
- render and inspect every actual PNG, not only fixture plots;
- retain compact Nature style, 8 pt text, editable SVG text, no clipped labels,
  no overlapping labels, external legends, and meaningful endpoint ticks.

The current `by-judge/` directory is not populated because the saved lesion
predictions still need a judge-by-benchmark aggregation pass.

## Fixed analysis decisions

- Primary matrix:
  `judge_results/capability_predictions_glm/no-common-support/design-matrix/`
- Population: 17,116 transcripts and 120,200 verdict rows.
- Features: 117 readable regressors: 96 non-annotator and 21 annotation-derived.
- Domain-specific regressors remain (`common_support=False`).
- Planning regressors retained: `plan_items`, `plan_unplanned_share`,
  `plan_position`.
- Correlation grouping: within-benchmark Pearson, mean available absolute
  correlation, threshold 0.8, non-transitive all-pairs, shortest-name
  representative.
- Joint partially pooled hierarchy:
  population + judge + benchmark + judge-by-benchmark slopes, with
  item-within-benchmark intercepts only.
- Evaluation: 4 held-out model folds x 4 held-out item folds = 16 outer folds.
- HiBayes is cancelled.
- Uncertainty: model-fold/item-fold variation and model/item-clustered intervals.

## Current scientific results

### Readable elastic-net screen + hierarchical ridge

Last exact interim summary: 13/16 folds, 96,120 OOF verdicts.

- Accuracy: 0.72844
- Conservative two-way fold interval: 0.70779–0.74676
- Log loss: 0.53797
- Row-matched conservative exact-rejudge floor: 0.78118
- Gap to conservative floor: 5.27 percentage points

The current screen retains roughly 100/117 features in completed folds. Several
unselected grid candidates hit SAGA's 3,000-iteration limit. Final ridge fits
converged. Candidate-level convergence instrumentation exists locally for future
runs, but the already-running RunPod code predates it.

### PCA-20 ridge

All 16 folds complete.

- Accuracy: 0.72085
- Conservative two-way fold interval: 0.70473–0.73610
- Log loss: 0.54365
- Row-matched conservative floor: 0.78130
- 20 components explain about 61–68% of variance by fold.

PCA retains most but not all readable predictive power.

Largest mean-absolute PC coefficients were PC2, PC3, PC5, PC6, and PC4.
Representative high-loading variables:

- PC2: `sink_share`, `depth_over_width`, `depth_max`, `redundant_steps`,
  `tool_calls`, `motif_feed_forward`;
- PC3: `symbol_links`, `chained_equalities`, `components_per_node`, `rigour`,
  `fan_out_share`, `unjustified_claims`;
- PC5: `display_equations`, `rigour`, `equation_explained`, `depth`,
  `verifies_result`, `clarity`;
- PC6: `rigour`, `degree_cv`, `depth`, `verifies_result`, `plan_items`,
  `unjustified_claims`.

### Readable lesioning

Complete: 16 folds, 262 labelled specifications, 260 unique refits per fold,
1,000 item-cluster bootstrap draws.

Aggregate artifacts are local at:

`judge_results/verdict_interpretability/readable-eda-20260828/lesions/`

Main results:

- annotation family: singular +0.10025 log loss, unique +0.06024;
- rubric block: singular +0.10140, unique +0.04005;
- strongest unique features:
  - `self_corrections`: +0.00451
  - `think_aloud_per_100_words`: +0.00338
  - `hedging_count`: +0.00336
  - `depth`: +0.00291
  - `efficiency`: +0.00187
- strongest singular features:
  - `rigour`: +0.05396
  - `clarity`: +0.04522
  - `efficiency`: +0.04339
  - `depth`: +0.04168
  - `bare_answer`: +0.03313
- planning family: singular +0.00274, unique −0.00158.

Important terminology: stored `bare_answer` does not mean the whole transcript
was answer-only. It marks fewer than 40 non-answer characters in assistant prose,
excluding tool calls/results. Eligibility uses at least 100 characters in the
full rendered transcript before the trailing answer. The user-facing label needs
review because the current name is misleading.

### Expanded features

Three exact matrices were built without regrouping:

- readable + 10 Llama surprisal features: 127 total;
- readable + 32 shared transcript-SAE features: 149 total;
- readable + both: 159 total.

Early four-item-0-fold comparison:

- readable mean: 0.7204;
- readable + surprisal: 0.7216;
- readable + SAE: 0.7165;
- readable + both had only one completed fold at the last comparison and was
  effectively unchanged.

Do not conclude until all folds complete.

## Live execution

### RunPod readable fits

- Pod: `jizl5l8gv3hol6`
- Secure A40, 9 vCPU, 50 GB RAM.
- Four model-fold workers originally launched; one worker had completed its four
  item folds at the last process check.
- Last known completion count: 13/16 folds.
- Remote results:
  `/workspace/small-judge/judge_results/verdict_interpretability/readable-eda-20260828/fits/`
- Logs:
  `/workspace/small-judge/run_logs/judge_fit_parallel-*.log`
- Do not stop until all 16 folds are saved. Then download results and stop/delete
  the pod to stop billing.

### SWC lesioning

- Fold-0 calibration: job `3418785`
- Folds 1–15: job array `3418786`
- Summary: job `3418791`
- All completed successfully.
- Ceph results:
  `/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/results/judge-verdict-lesions-readable117-20260829-4016c5f/`

### SWC PCA-20 ridge

- Fold array: `3418792`
- Aggregate: `3418797`
- Complete.
- Ceph results:
  `/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/results/judge-verdict-pca20-readable117-20260829-b366453/`

### SWC expanded fits

- readable + SAE: `3418836`
- readable + both: `3418837`
- readable + surprisal: `3418838`
- Last known status: 12 workers running; 9 fold outputs complete at an earlier
  checkpoint.
- Ceph results:
  `/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/results/judge-verdict-expanded-fits-20260829-4fd025a/`

## Eligibility audit

The simple transcript rule was already applied upstream:

```text
len(strip_answer(full rendered transcript)) >= 100
```

Evidence from the historical build:

- 335 canonical scored-correct short transcripts were removed;
- transcript rows: 17,451 -> 17,116;
- verdict rows: 120,998 -> 120,200, removing 798;
- 2,110 unparseable judge answers were excluded before verdict construction;
- current `pick_left` has no nulls and only 0/1.

Therefore the current verdict rows already satisfy the user's intended filter.
The real gaps are:

- `judge_verdict_analysis` does not verify the eligibility contract in the
  processed manifest;
- old short-but-score-correct rows still entered leave-item-out model-accuracy
  calculations, contaminating `model_accuracy`, `accuracy_difference`, and
  `left_stronger` targets. This affects capability/stronger-model analyses, not
  the current `pick_left` verdict target.

Any rebuilt source matrix should recompute:

```text
correct = score_correct AND has_reasoning(transcript, min_chars=100)
```

before capability targets, scaling, grouping, PCA, or folds, and record the
eligibility predicate/counts/hash in the manifest.

## Figure status

Current plot code is in:

`scripts/judge_verdict_analysis/result_plots.py`

Relevant commits include `63c1f1e`, `4589f6b`, and `4bedab8`.

Some revised figures have been rendered under the requested subfolders, but the
user explicitly considers the current figure set poor. Treat all current figures
as drafts requiring visual cleanup. In particular:

- exact capability-GLM layout is binding;
- Singular/Unique/Coefficient must all be shown;
- benchmark and judge splits are primary;
- per-judge figures are still missing;
- scatter labels/limits and semantic labels require final user-facing review;
- do not hand off figures based only on tests—open every actual PNG.

## Code and verification

- Branch is pushed regularly to `origin/judge-dissect` only.
- Never push to `main`.
- Last full package verification before the latest concurrent plot/cell-summary
  changes: 178 tests passed; later agents reported 234 passing with their own
  additions.
- Relevant commits:
  - `bd46b7d`: parallel/resumable frequentist fitting
  - `d6746e0`: convergence diagnostics
  - `90a3f3d`: expanded matrix construction/schema validation
  - `4016c5f`: hardened lesion checkpoints
  - `b366453`: PCA-ridge pipeline hardening
  - `375d3d7`: PC-L1 implementation
  - `48496c0`: expanded matrix provenance validation
  - `d61ba52`: benchmark and judge-by-benchmark lesion summary implementation
  - `4fd025a`: old/new sklearn elastic-net compatibility
  - `4bedab8`: latest reviewed figure hardening before final visual cleanup

Preserve unrelated dirty-worktree files. Use path-scoped commits.

## Next tasks, in order

1. **Redo and visually clean all figures.**
   - Use the capability-GLM layouts exactly.
   - Populate subfolders.
   - Generate benchmark and per-judge benchmark-row figures with
     Singular/Unique/Coefficient columns.
   - Open every PNG and iterate until useful.

2. **Run the saved-prediction judge/cell aggregation.**
   - Code: `scripts/judge_verdict_analysis/lesion_cells.py`
   - Commit: `d61ba52`
   - It reads existing lesion checkpoints; no refitting.
   - Independently review, stage, submit, download `cell-summary.csv`, and render
     `by-judge/` figures.

3. **Complete and download the remaining readable/expanded fits.**
   - Monitor RunPod and Slurm.
   - Validate convergence and artifacts.
   - Stop/delete RunPod after download.
   - Render the five-variant ceiling comparison.

4. **Run PC-L1.**
   - Code commit: `375d3d7`.
   - Pure L1 hierarchical logistic regression on fold-local 20-PC scores.
   - Review/stage/submit Slurm fold array and aggregate.

5. **Run PC singular/drop-one lesioning.**
   - Tune the full PC-ridge penalty once per fold.
   - Freeze it for 20 singular and 20 drop-one refits.
   - Align/load component definitions carefully across fold-local bases.

6. **Interpret PCs.**
   - Combine selection/lesion results with loadings.
   - Account for fold-local component sign/permutation instability.
   - Do not present PC names as stable constructs without alignment.

7. **Harden eligibility provenance and capability targets.**
   - Add fail-closed matrix manifest checks for canonical/both-correct/100-char
     eligibility.
   - Recompute corrected capability/stronger-model targets before rerunning those
     analyses.

8. **Only after the above:** consider a smaller predeclared readable model,
   grouped base-feature regularization, and the planned controlled length
   interactions.

## Standing safety rules

- No heavy local fits or large local Parquet processing.
- Run full fitting/lesions on SWC or RunPod.
- No HiBayes unless explicitly re-authorized.
- Never delete/overwrite Ceph data; use new immutable paths and retained partials.
- No Python/install/build jobs on the SWC gateway.
- Never reset/stash unrelated dirty-worktree changes.
- Package-relative imports only; no imports from other `scripts/` analysis
  folders.
