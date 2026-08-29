# Judge-verdict interpretation — overnight execution plan

Date: 2026-08-29
Repo: `/Users/moriohamada/PycharmProjects/ai_safety/small-judge`
Branch: `judge-dissect`

## Objective

Explain how partially pooled LLM judges choose between two correct transcripts,
using the smallest readable feature set that retains useful held-out predictive
power. Report singular signal, unique signal, redundancy, coefficient stability,
and the gap to the empirical judge-reliability ceiling.

## Fixed data and model decisions

- Primary readable matrix:
  `judge_results/capability_predictions_glm/no-common-support/design-matrix/`
- Rows: 17,116 transcripts and 120,200 pair rows.
- Features: 117 total: 96 non-annotator and 21 annotation-derived.
- Domain-specific features are retained (`common_support=False`).
- Planning predictors are `plan_items`, `plan_unplanned_share`, and
  `plan_position`.
- Correlation grouping remains Pearson `|r| >= 0.8`, non-transitive all-pairs,
  shortest-name representative selection.
- The verdict model is joint and partially pooled, never one independent model
  per judge. It contains population, judge, benchmark, judge-by-benchmark slopes,
  and item-within-benchmark intercepts. Items have no slopes.
- Evaluation is 4 held-out model folds x 4 held-out item folds = 16 outer folds.
- HiBayes is cancelled. Frequentist fold variation and model/item-clustered
  intervals are the uncertainty treatment.

## Current RunPod fit

- Pod: `jizl5l8gv3hol6`, secure A40, 9 vCPU, 50 GB RAM, `$0.44/hour` plus storage.
- Four model-fold processes run concurrently; each evaluates two tuning settings
  concurrently. Do not stop or alter these processes.
- Input is the 117-feature matrix above. No lesioning or HiBayes runs on the pod.
- Results root on pod:
  `/workspace/small-judge/judge_results/verdict_interpretability/readable-eda-20260828/fits/`
- Logs:
  `/workspace/small-judge/run_logs/judge_fit_parallel-*.log`
- Monitor approximately every 20 minutes.
- Every individual solver is bounded by `glm_max_iter=3000`; there is no
  wall-clock timeout. A fold contains a finite 64 fits.
- Several SAGA grid candidates have reached the iteration cap. Local code now
  records candidate-level convergence, but the already-running pod code predates
  that diagnostic. Do not accept a fold without checking whether the chosen
  setting is plausibly unaffected.

First completed fold:

- Fold: `model-fold-0__item-0`
- Accuracy: `0.7181247780`
- Log loss: `0.5475400652`
- Elastic-net choice: `C=0.01`, L1 ratio `0.25`
- Ridge choice: `C=0.01`
- Final ridge iterations: 48
- Selected features: 100 of 117

## Ceiling comparison

Ceiling source:
`judge_results/verdict_dissection/exact_rejudge_20260828/summary.csv`

- Use `wilson_95_low` as the conservative ceiling.
- Weighted conservative ceiling across the 21 available judge-by-benchmark
  cells: `0.7803589165`.
- Weighted raw rejudge agreement: `0.8595238095`.
- The first fold is about 6.2 percentage points below the conservative ceiling.
- Do not draw a final conclusion until all 16 out-of-fold predictions are
  aggregated by judge and benchmark.

## Decision after readable predictions

Lesioning happens regardless of predictive outcome.

In parallel, compare the readable model with progressively less interpretable
existing features:

1. readable + 10 historical Llama surprisal features: 127 total;
2. readable + 32 shared transcript-SAE features: 149 total;
3. readable + both: 159 total.

Use the exact feature lists in
`scripts/judge_verdict_analysis/config.py`. Do not add all scorer variants,
benchmark-specific transcript SAEs, or clause SAEs to the primary comparison.
Do not regenerate features or download Hugging Face data.

The canonical source design matrices live at
`/Volumes/T7/judge-dissection/design-matrices`. Expanded matrices must preserve
the exact 117 baseline and append explicit columns; they must not rerun
correlation grouping. Check for an existing canonical matrix on Ceph before any
large laptop-to-cluster transfer.

## Readable lesioning contract

Run on SWC Slurm, never on the laptop or RunPod.

For each of 16 outer folds:

1. Tune ridge `C` once on the complete 117-feature training set.
2. Save full-model and intercept-only references.
3. Freeze the selected ridge `C` for every interpretation refit.
4. Never rerun elastic net or retune a lesion/singular model.

Required labelled specifications:

- remove both annotation blocks together (`rubric` + `intermediate-errors`),
  leaving 96 features;
- leave each of 117 features out individually;
- fit each of 117 features alone;
- leave out and fit alone each of 9 source blocks;
- leave out and fit alone each of 5 higher-level families:
  - annotations;
  - hand-engineered text;
  - clause;
  - clause structure;
  - planning.

Planning exactly duplicates the `clause-plan` block. Keep both labels but reuse
the same two fitted artifacts.

Totals:

- 262 labelled specifications;
- 260 unique feature sets per fold;
- 16 fold tasks plus one aggregation task;
- Slurm array `0-15%4`, with fold 0 inspected as a retained calibration run
  before releasing the other folds when practical.

Every specification must save:

- held-out predictions;
- item-level log-loss and accuracy inputs/deltas;
- coefficients;
- specification/reference metadata;
- frozen penalty and convergence diagnostics.

Unique contribution is the held-out loss from removing a feature/group relative
to the full model. Singular contribution is the held-out gain of the alone model
relative to the intercept-only model. High singular and low unique contribution
means redundancy.

## Pooled 20-component sensitivity fit

Run as a separate Slurm sensitivity model; do not replace readable fitting or
feature-level lesioning.

- Fit one pooled-across-benchmarks 20-component basis within each outer training
  fold only.
- Use an uncentred TruncatedSVD/PCA-equivalent transform on pair differences so
  reversing left/right exactly negates component scores.
- Transform train and held-out rows with the same fold-local basis.
- Fit the same joint hierarchy with ridge tuning only; no elastic-net screen.
- Save predictions, metrics, loadings, explained variance, ridge tuning, and
  convergence diagnostics.
- Do not launch a component-count sweep unless requested.

## Plot contract

All figures use compact Nature style, 8 pt text, shared comparable axes,
300-DPI PNG, editable-text SVG, and no clipped labels.

Prediction figures:

- held-out accuracy and log loss overall and by judge x benchmark;
- conservative ceiling reference/band from exact rejudge;
- readable versus +surprisal, +SAE, +both, and 20-component sensitivity models;
- model/item-clustered 95% intervals and visible 16-fold variation.

Interpretation figures:

- singular-versus-unique feature scatter;
- separate compact ranked singular and unique feature panels;
- grouped block and family singular/unique comparison;
- per-block feature panels;
- annotation-removal comparison;
- coefficient sign/magnitude and feature-selection stability across folds;
- common judge/benchmark scales and labels only for informative points.

## Slurm and Ceph safety

- Stage to a new immutable path containing the code hash and matrix-manifest
  hash. Never reuse, delete, or overwrite existing Ceph data.
- Gateway work is limited to `rsync`/`scp`, `sbatch`, `squeue`, `sacct`, `ls`,
  `tail`, `stat`, `grep`, and other trivial read-only checks.
- No Python, environment creation, installation, compilation, or analyses on the
  gateway.
- Compute jobs require `SLURM_JOB_ID` and must reject `gateway*`/`sgw*` hosts.
- Use deterministic resumable checkpoints, retained partials, `--requeue`, and
  `--open-mode=append`.
- Stage only required code and matrix artifacts. Do not stage HF mirrors or raw
  rollout logs.

## Code status

Relevant commits:

- `bd46b7d`: parallel and resumable frequentist tuning.
- `d6746e0`: candidate/final convergence diagnostics.
- `500e3ce`, `90a3f3d`: exact expanded-matrix construction and schema fixes.
- `17c1ac0`: initial resumable Slurm lesion grid; currently under fix review for
  coefficient/metric artifacts, checkpoint locking, and completion validation.

Shared integration changes are in progress for:

- exact expanded feature sets and lesion families in `config.py`;
- checksum-pinned old-matrix provenance;
- Slurm compute-node guard;
- lazy HiBayes imports for CPU-only jobs;
- committing required package files that were previously untracked.

Before staging or submission, run the full package tests with cache roots under
`/tmp`, review all task diffs, and ensure the Slurm scripts pass shell syntax
checks.
