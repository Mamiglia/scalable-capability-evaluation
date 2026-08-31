# Pairwise Capability Hierarchical Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit and freeze one learned capability benchmark-deviation ratio per outer fold, rerun the capability analysis on SWC, and promote judge-capability comparison in the regenerated report.

**Architecture:** Extend the existing `SparseTask` seam with task-owned pooling parameters and a task-owned scale estimator. Keep the judge estimator unchanged; add the no-intercept population-plus-benchmark scale model to the existing capability wrapper. Reuse the lesion and report pipelines, adding only the missing frozen ratio and a compact joined comparison table.

**Tech Stack:** Python, pandas, NumPy/SciPy, scikit-learn, JAX/NumPyro, pytest, Slurm on SWC.

**Spec:** `docs/superpowers/specs/2026-08-31-pairwise-capability-hierarchical-refit-design.md`

## Global Constraints

- Run every test, synthetic fit, smoke fit, full fit, ablation, and report job through Slurm on SWC; do not execute Python tests or fitting locally.
- Preserve exact pair-reversal antisymmetry and the fixed probability-0.5 null.
- Estimate scales from each outer-training fold only, then freeze the derived ratio for `C` tuning, the full fit, and every lesion in that fold.
- Do not change the judge hierarchy, judge estimator, or judge numerical behavior.
- Write only new immutable Ceph paths; never overwrite or delete existing Ceph data artifacts.
- Preserve unrelated dirty and untracked work in the laptop checkout.

---

### Task 1: Task-owned pooling and capability scale fit

**Files:**
- Modify: `scripts/judge_verdict_analysis/fit_frequentist.py`
- Modify: `capability_predictions_glm/pairwise_wrapper.py`
- Test: `capability_predictions_glm/test_pairwise_wrapper.py`
- Test: `scripts/judge_verdict_analysis/test_pooling.py`

**Interfaces:**
- Consumes: existing `estimate_pooling_ratios(train, features, penalty, config)` for the judge task.
- Produces: `SparseTask.pooling_parameters`, `SparseTask.estimate_pooling`, and `_estimate_capability_pooling_ratio(train, features, penalty, config)` returning `(ratios, diagnostics)`.

- [ ] **Step 1: Write failing wrapper tests**

Add tests asserting that L1 uses multiplier `1 / r`, L2 uses `1 / sqrt(r)`, the capability task declares only `benchmark_deviation_penalty`, the synthetic scale fit returns a finite positive non-config ratio, and a completed fold records that ratio in diagnostics and provenance.

```python
@pytest.mark.parametrize(("penalty", "expected"), (("l1", 0.25), ("ridge", 0.5)))
def test_capability_benchmark_ratio_uses_penalty_specific_scaling(tmp_path, penalty, expected):
    frame = pd.DataFrame({
        "bench": ["b"], "item": ["i"], "model_left": ["l"],
        "model_right": ["r"], "signal": [2.0], "left_stronger": [1],
    })
    config = replace(_config(tmp_path, penalty=penalty), benchmark_deviation_penalty=4.0)
    design, columns = _capability_design(frame, ["signal"], ("b",), config)
    assert columns.loc[columns.block == "benchmark_slope", "multiplier"].item() == expected
    assert design[0, 1] == pytest.approx(frame.signal.item() * expected)
```

- [ ] **Step 2: Commit the failing tests and stage that commit into a new SWC code directory**

```bash
git add capability_predictions_glm/test_pairwise_wrapper.py scripts/judge_verdict_analysis/test_pooling.py
git commit -m "Test capability hierarchy scale fitting"
rsync -a --ignore-existing .worktrees/pairwise-capability-hierarchical-refit/ \
  swc:/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/capability-hierarchy-red-20260831/repo/
```

- [ ] **Step 3: Submit the focused red test job**

```bash
ssh swc "sbatch --partition=cpu --cpus-per-task=2 --mem=16G --time=00:30:00 --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/capability-hierarchy-red-20260831/test-%j.out --wrap='/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_wrapper.py scripts/judge_verdict_analysis/test_pooling.py -q'"
```

Expected: FAIL because the task pooling metadata and capability estimator do not yet exist and L1 still uses square-root scaling.

- [ ] **Step 4: Implement the minimal shared seam**

Extend `SparseTask` with:

```python
pooling_parameters: tuple[str, ...] = ()
estimate_pooling: Callable[[pd.DataFrame, list[str], str, AnalysisConfig],
                           tuple[dict[str, float], pd.DataFrame]] | None = None
```

Set `JUDGE_TASK` to the existing four `POOLING_PARAMETERS` and existing estimator. In `tune_sparse`, use `task.estimate_pooling` rather than identity comparison with `JUDGE_TASK`. In `fit_sparse`, validate/apply exactly `task.pooling_parameters` from `frozen_hyperparameters` and include them in diagnostics.

- [ ] **Step 5: Implement the capability scale model in the existing wrapper**

Add a no-intercept NumPyro model with population shape `(feature_count,)` and benchmark shape `(benchmark_count, feature_count)`. Estimate both HalfNormal scales from outer-training rows and derive:

```python
power = 1 if penalty == "l1" else 2
ratio = (population_scale / benchmark_scale) ** power
```

Set `CAPABILITY_TASK.pooling_parameters` to `("benchmark_deviation_penalty",)` and its estimator to `_estimate_capability_pooling_ratio`. Replace the manual square-root multiplier with `penalty_multiplier` and reject non-converged selected tuning/final fits before writing artifacts.

- [ ] **Step 6: Stage the green implementation into a second new SWC code directory and rerun focused tests**

```bash
git add scripts/judge_verdict_analysis/fit_frequentist.py capability_predictions_glm/pairwise_wrapper.py
git commit -m "Fit capability benchmark hierarchy scale"
rsync -a --ignore-existing .worktrees/pairwise-capability-hierarchical-refit/ \
  swc:/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/capability-hierarchy-green-20260831/repo/
ssh swc "sbatch --partition=cpu --cpus-per-task=2 --mem=16G --time=00:30:00 --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/capability-hierarchy-green-20260831/test-%j.out --wrap='/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_wrapper.py scripts/judge_verdict_analysis/test_pooling.py -q'"
```

Expected: PASS, including the unchanged judge pooling tests.

### Task 2: Freeze the fitted ratio for lesions

**Files:**
- Modify: `capability_predictions_glm/pairwise_wrapper_lesions.py`
- Test: `capability_predictions_glm/test_pairwise_wrapper_lesions.py`

**Interfaces:**
- Consumes: full-fit `diagnostics.json` and `tuning.csv` containing `benchmark_deviation_penalty`.
- Produces: `frozen-hyperparameters.json` containing `C`, `l1_ratio`, and `benchmark_deviation_penalty` for every capability lesion refit.

- [ ] **Step 1: Write the failing lesion test**

Update the preparation and specification tests to require the benchmark ratio in the frozen JSON and every lesion diagnostic, and to assert that no tuner is called.

```python
assert set(frozen) == {"C", "l1_ratio", "benchmark_deviation_penalty"}
assert diagnostics["benchmark_deviation_penalty"] == frozen["benchmark_deviation_penalty"]
```

- [ ] **Step 2: Run the focused lesion test on SWC and confirm failure**

```bash
/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_wrapper_lesions.py -q
```

Expected: FAIL because lesion preparation currently copies only `C` and `l1_ratio`.

- [ ] **Step 3: Copy and validate the fitted ratio**

Load `benchmark_deviation_penalty` from full diagnostics, require it to be finite and positive, require the selected converged `C` tuning record to carry the same value, and include it in `frozen-hyperparameters.json`. Let the generalized `fit_sparse` apply it during each shared lesion refit.

- [ ] **Step 4: Rerun the focused lesion test on SWC**

```bash
/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_wrapper_lesions.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add capability_predictions_glm/pairwise_wrapper_lesions.py capability_predictions_glm/test_pairwise_wrapper_lesions.py
git commit -m "Freeze capability hierarchy scale for lesions"
```

### Task 3: Promote judge-capability comparison

**Files:**
- Modify: `capability_predictions_glm/pairwise_report.py`
- Test: `capability_predictions_glm/test_pairwise_report.py`

**Interfaces:**
- Consumes: existing `stronger_model_accuracy_summary`, `judge_coefficient_statistics`, `judge_cosine_summary`, and `judge_eci` tables.
- Produces: `judge_capability_overview` with variant, judge, matched accuracies, accuracy difference, coefficient agreement, cosine, pair count, and ECI.

- [ ] **Step 1: Write a failing overview-table test**

```python
overview = judge_capability_overview(accuracy, statistics, cosines, eci)
assert set(overview) >= {
    "variant", "judge", "capability_accuracy", "judge_accuracy",
    "accuracy_difference", "pearson", "spearman", "cosine", "eci",
}
assert overview.accuracy_difference.item() == pytest.approx(0.08)
```

- [ ] **Step 2: Run the report test on SWC and confirm failure**

```bash
/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_report.py -q
```

Expected: FAIL because `judge_capability_overview` does not exist.

- [ ] **Step 3: Add the compact joined table and promote it in the summary**

Build accuracy comparisons for all four variants, join only overall coefficient statistics/cosines, attach ECI by judge, and compute `accuracy_difference = capability_accuracy - judge_accuracy`. Return/save the table with the other comparison artifacts and render it directly under `## Judge-capability prediction and representation comparison` in `scientific-summary.md`, followed by the existing descriptive caveat.

- [ ] **Step 4: Rerun the report test on SWC**

```bash
/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add capability_predictions_glm/pairwise_report.py capability_predictions_glm/test_pairwise_report.py
git commit -m "Promote judge capability comparison"
```

### Task 4: SWC verification, smoke fit, and immutable submission

**Files:**
- Modify: `docs/handoffs/2026-08-31-hierarchical-fit-invalidation-handoff.md`

**Interfaces:**
- Consumes: committed implementation, pinned matrix/response/PCA hashes, existing Slurm wrappers, and canonical judge artifact handoff.
- Produces: new immutable capability full-fit and lesion roots plus recorded Slurm job IDs.

- [ ] **Step 1: Run the complete relevant suite on SWC**

```bash
/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/glm-ablations-20260826/venv/bin/python -m pytest capability_predictions_glm/test_pairwise_wrapper.py capability_predictions_glm/test_pairwise_wrapper_lesions.py capability_predictions_glm/test_pairwise_report.py scripts/judge_verdict_analysis/test_pooling.py scripts/judge_verdict_analysis/test_pipeline.py -q
```

Expected: all tests pass; judge regression coverage remains green.

- [ ] **Step 2: Submit one raw-L2 outer-fold smoke fit to a new immutable root**

Invoke `python -m capability_predictions_glm.pairwise_wrapper` with fold index zero and the exact pinned matrix, response, and output arguments from the authoritative handoff. Use a CPU Slurm allocation, append-only log, and code/input identifiers in the output path.

- [ ] **Step 3: Audit the smoke artifacts**

Require a finite positive non-default `benchmark_deviation_penalty`, a variance-component tuning row, one selected converged `C` row carrying the same ratio, a converged final fit, exact reversal probabilities, and hash-valid manifest artifacts.

- [ ] **Step 4: Submit the 64 full-fit fold tasks**

Submit 16 folds for each of raw L1, raw L2, PC L1, and PC L2 into the new immutable root, with at most four concurrent tasks per variant and `afterok` aggregate dependencies.

- [ ] **Step 5: Submit dependency-safe lesion stages**

After each full aggregate succeeds, submit 16 preparation shards, verify the exact per-fold specification inventory, then submit specification, finalize, and summary stages through the existing `pairwise_wrapper_lesions.sbatch` entry point.

- [ ] **Step 6: Record jobs and commit the handoff update**

```bash
git add docs/handoffs/2026-08-31-hierarchical-fit-invalidation-handoff.md
git commit -m "Record hierarchical capability refit jobs"
```

Record staged code commit, immutable roots, all parent/dependent job IDs, expected inventories, and the rule that comparisons wait for hash-valid canonical judge aggregates.
