# Canonical-Anchor Frontier Redo Implementation Plan

> **SUPERSEDED FOR MAIN INTERPRETATION BY THE LATEST USER REQUIREMENT.** This plan records the
> historical anchor/ECI workflow only. The sole authoritative result is the deliberately
> in-sample [rank–accuracy result](../../results/2026-08-30-canonical-anchor-frontier-redo.md),
> whose machine-readable authority and complete discarded-provenance rules are in
> [MAIN_OUTPUT.json](../../../judge_results/frontier/eci-semiannual-canonical-anchors/MAIN_OUTPUT.json).
> Do not interpret the forecast, anchor, or ECI steps below as main results.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the noncanonical quarterly-frontier forecasts using temporally eligible canonical-pool anchors, broad low-item pair coverage, explicit-intercept calibration, and matched ECI baselines for GPQA and SWE-mini.

**Architecture:** A coverage pass merges frozen canonical outcomes, frontier outcomes, and every compatible existing verdict, then freezes only the missing pair/item top-ups. A separate analysis entry point caps dense pair influence, fits Bradley-Terry abilities and binomial-logit calibration without target truth, validates the intercept score equation, and reuses the established release-time plotting grammar. Heavy log processing and all paid judging run on RunPod; local work is limited to code, small tests, control-plane operations, and final figure inspection.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, PyYAML, Inspect AI, matplotlib, pytest, RunPod/runpodctl, OpenRouter.

**Spec:** `docs/superpowers/specs/2026-08-30-canonical-anchor-frontier-redo-design.md`

## Global Constraints

- This is a side analysis only; never modify canonical difficulty, canonical verdicts, or canonical pool files.
- Keep the eight-model quarterly frontier identical across GPQA and SWE-mini; exclude `high` and `xhigh` answer variants and retain medium/default variants.
- Judges are GPT-4o 2024-11-20, Gemini 2.5 Pro, GPT-5.2, o4-mini, and DeepSeek V4 Flash 0731.
- Judges see only frozen Q1 items solved correctly by both answer models, with reasoning hidden and both presentation orders.
- An edge or anchor is eligible only when its model releases are no later than `max(release(target), release(judge))`.
- The canonical pool is the calibration set; historical models are backfill only when fewer than four eligible connected canonical anchors remain.
- New unordered pairs use at most eight deterministic jointly-correct Q1 item IDs and both presentation orders.
- Primary Bradley-Terry fits cap every unordered pair at eight item IDs; uncapped fits are sensitivity outputs.
- Headline calibration is unregularized binomial logistic regression with a free intercept; target hard outcomes never enter its fit.
- Heavy source-log work, full analysis, and paid calls run on RunPod, not the laptop.
- Incremental API spend stops before the estimated total project spend exceeds $200 unless the user gives new authorization.
- OpenRouter research calls must use hard endpoint pins where available, disabled fallbacks, denied data collection, compatible parameter enforcement, endpoint snapshots, and per-response served-provider logs.
- Primary figures use answer-model release time on x, hard-quartile accuracy on y, and a dashed vertical judge-release line; save PDF and 300 dpi PNG in Nature-manuscript style.

---

### Task 1: Freeze the canonical coverage and top-up manifest

**Files:**
- Create: `configs/frontier_eci_canonical_anchors/experiment.yaml`
- Create: `scripts/frontier_eci/canonical_coverage.py`
- Create: `tests/frontier_eci/test_canonical_coverage.py`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/coverage/blocks.csv`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/coverage/pair_items.csv`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/coverage/summary.json`

**Interfaces:**
- Consumes: `configs/frontier_eci/experiment.yaml`, `configs/frontier_eci/item_quartiles.csv`, frozen `identity.parquet`, frontier `rollouts.parquet`, and compatible verdict parquet files.
- Produces: `prediction_cutoff(target: str, judge: str, releases: dict[str, str]) -> pd.Timestamp`; `eligible_models(models: list[str], target: str, judge: str, releases: dict[str, str]) -> list[str]`; `canonical_anchor_models(models: list[str], target: str, judge: str, releases: dict[str, str]) -> list[str]`; `stable_items(items: list[str], limit: int = 8) -> list[str]`; `choose_pairs(existing: pd.DataFrame, candidates: pd.DataFrame, desired_direct_anchors: int) -> pd.DataFrame`; `coverage_blocks(config: dict, rollouts: pd.DataFrame, verdicts: pd.DataFrame, quartiles: pd.DataFrame) -> pd.DataFrame`; `pair_items(blocks: pd.DataFrame, rollouts: pd.DataFrame, verdicts: pd.DataFrame, limit: int) -> pd.DataFrame` with columns `bench,judge,target,opponent,item,reason`.

- [ ] **Step 1: Write eligibility and deterministic-cap tests**

```python
def test_anchor_is_eligible_when_older_than_target_or_judge():
    releases = {"target": "2024-01-01", "judge": "2026-01-01",
                "older": "2025-01-01", "future": "2027-01-01"}
    assert prediction_cutoff("target", "judge", releases) == pd.Timestamp("2026-01-01")
    assert eligible_models(["older", "future"], "target", "judge", releases) == ["older"]


def test_item_cap_is_stable_and_never_exceeds_eight():
    items = [f"item-{index}" for index in range(20)]
    assert stable_items(items) == stable_items(list(reversed(items)))
    assert len(stable_items(items)) == 8
```

- [ ] **Step 2: Write graph-coverage and pair-breadth tests**

```python
def test_missing_direct_anchor_pairs_are_preferred_to_more_items():
    existing = pd.DataFrame({"target": ["t"] * 16, "opponent": ["a"] * 16,
                             "item": [f"i{x // 2}" for x in range(16)]})
    candidates = pd.DataFrame({"target": ["t", "t"], "opponent": ["a", "b"],
                               "hard_accuracy": [0.2, 0.8]})
    selected = choose_pairs(existing, candidates, desired_direct_anchors=2)
    assert selected[["target", "opponent"]].drop_duplicates().values.tolist() == [["t", "b"]]


def test_target_truth_is_not_an_anchor():
    releases = {"a": "2023-01-01", "b": "2023-02-01",
                "frontier": "2024-01-01", "judge": "2024-01-01"}
    anchors = canonical_anchor_models(["a", "b", "frontier"], "frontier", "judge", releases)
    assert anchors == ["a", "b"]
```

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/frontier_eci/test_canonical_coverage.py -q`

Expected: FAIL because `canonical_coverage.py` and its functions do not exist.

- [ ] **Step 4: Implement the minimal coverage pass**

The YAML fixes the existing source paths and policy:

```yaml
base_experiment: configs/frontier_eci/experiment.yaml
quartiles: configs/frontier_eci/item_quartiles.csv
canonical_identity: judge_results/frontier/eci-semiannual/canonical-source/identity.parquet
frontier_rollouts: judge_results/frontier/eci-semiannual/new-tables/rollouts.parquet
verdicts:
  - judge_results/frontier/eci-semiannual/hf/verdicts.parquet
  - judge_results/frontier/eci-semiannual/new-tables/verdicts.parquet
pair_item_cap: 8
minimum_anchors: 4
desired_direct_anchors: 6
historical_backfill: []
results: judge_results/frontier/eci-semiannual-canonical-anchors
```

Implement temporal filtering with the existing release-normalization logic from `analyze.py`. Use a stable SHA-256 ordering of `bench|target|opponent|item` before taking eight item IDs. Build connected components with a small adjacency-set breadth-first search; add no graph dependency. For every block, export eligible anchors, connected anchors, direct anchors, ability/accuracy range proxies, and exclusion reasons. Recommend new direct target–canonical-anchor pairs until six direct anchors exist or the eligible set is exhausted; flag fewer than four as `needs_historical_backfill`.

- [ ] **Step 5: Run focused and existing side-analysis tests**

Run: `uv run pytest tests/frontier_eci/test_canonical_coverage.py tests/frontier_eci/test_analysis.py -q`

Expected: PASS.

- [ ] **Step 6: Generate and inspect the lightweight coverage audit locally**

Run:

```bash
uv run python scripts/frontier_eci/canonical_coverage.py \
  --config configs/frontier_eci_canonical_anchors/experiment.yaml
```

Expected: `blocks.csv`, `pair_items.csv`, and `summary.json` are written; the command prints the number of identifiable, top-up, and backfill-required blocks without opening raw `.eval` logs.

- [ ] **Step 7: Commit the coverage implementation and frozen manifest**

```bash
git add configs/frontier_eci_canonical_anchors/experiment.yaml scripts/frontier_eci/canonical_coverage.py tests/frontier_eci/test_canonical_coverage.py judge_results/frontier/eci-semiannual-canonical-anchors/coverage
git commit -m "Audit canonical frontier anchor coverage"
git push origin judge-dissect
```

---

### Task 2: Add balanced Bradley-Terry and explicit-intercept calibration

**Files:**
- Create: `scripts/frontier_eci/canonical_analysis.py`
- Create: `tests/frontier_eci/test_canonical_analysis.py`
- Read/reuse: `scripts/frontier_eci/analyze.py`
- Read/reuse: `src/ranking/bradley_terry.py`

**Interfaces:**
- Consumes: the frozen coverage config, canonical/frontier outcomes, and combined compatible verdicts.
- Produces: `cap_pair_items(verdicts: pd.DataFrame, limit: int = 8) -> pd.DataFrame`; `fit_binomial_logit(ability: pd.Series, outcomes: pd.DataFrame) -> Fit`; `predict_from_ability(ability: pd.Series, hard_outcomes: pd.DataFrame, target: str, anchors: list[str]) -> Prediction`; `predict_block(verdicts: pd.DataFrame, hard_outcomes: pd.DataFrame, target: str, anchors: list[str], resamples: int, seed: int) -> tuple[dict, pd.DataFrame]`, where `Fit` contains `intercept, slope, converged, gradient_intercept, gradient_slope`.

- [ ] **Step 1: Write pair-cap tests**

```python
def counterbalanced_rows(items):
    rows = []
    for item in items:
        rows.extend([
            {"bench": "gpqa", "judge": "j", "item": str(item),
             "model_left": "a", "model_right": "b", "pick_left": 1},
            {"bench": "gpqa", "judge": "j", "item": str(item),
             "model_left": "b", "model_right": "a", "pick_left": 0},
        ])
    return pd.DataFrame(rows)


def test_pair_cap_keeps_both_orders_for_the_same_eight_items():
    verdicts = counterbalanced_rows(range(12))
    capped = cap_pair_items(verdicts, limit=8)
    assert capped.item.nunique() == 8
    assert capped.groupby("item").size().eq(2).all()


def test_pair_cap_is_invariant_to_input_order():
    verdicts = counterbalanced_rows(range(12))
    pd.testing.assert_frame_equal(cap_pair_items(verdicts),
                                  cap_pair_items(verdicts.sample(frac=1, random_state=7)))
```

- [ ] **Step 2: Write free-intercept and leakage tests**

```python
def test_binomial_logit_has_balanced_weighted_anchor_residuals():
    ability = pd.Series({"a": -1.5, "b": -0.5, "c": 0.5, "d": 1.5})
    outcomes = pd.DataFrame({"model": ability.index,
                             "successes": [2, 6, 14, 18], "total": [20] * 4})
    fit = fit_binomial_logit(ability, outcomes)
    assert fit.converged
    assert abs(fit.gradient_intercept) < 1e-6
    assert abs(fit.gradient_slope) < 1e-6
    assert (fit.anchor_table.residual > 0).any()
    assert (fit.anchor_table.residual < 0).any()


def test_target_hard_outcomes_never_change_prediction():
    ability = pd.Series({"a": -1.5, "b": -0.5, "c": 0.5, "d": 1.5,
                         "frontier": 2.0})
    outcomes = pd.DataFrame({"model": ["a", "b", "c", "d", "frontier"],
                             "successes": [2, 6, 14, 18, 0], "total": [20] * 5})
    first = predict_from_ability(ability, outcomes, "frontier", ["a", "b", "c", "d"])
    outcomes.loc[outcomes.model.eq("frontier"), "successes"] = 20
    second = predict_from_ability(ability, outcomes, "frontier", ["a", "b", "c", "d"])
    assert first.predicted == second.predicted
    assert first.intercept == second.intercept
    assert first.slope == second.slope
```

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/frontier_eci/test_canonical_analysis.py -q`

Expected: FAIL because the new analysis functions do not exist.

- [ ] **Step 4: Implement the capped ranking and binomial likelihood**

Reuse `ranking.rank` after deterministic per-pair item capping. Fit the two-parameter model by minimizing:

```python
def loss(beta):
    eta = beta[0] + beta[1] * x
    return np.sum(total * np.logaddexp(0, eta) - successes * eta)
```

Compute the analytic gradient, reject fits that do not converge, have fewer than four anchors, have a coefficient magnitude above 50, or fail `abs(gradient) < 1e-6`. Export anchor fitted probabilities and residuals. Keep the target completely outside the outcomes passed to the optimizer.

- [ ] **Step 5: Implement two-source item bootstrap**

Resample Q1 item IDs, refit capped Bradley-Terry abilities, independently resample hard-quartile item IDs, refit calibration, and retain successful target predictions. Export the 2.5%, 50%, and 97.5% quantiles and successful-refit count. A block with zero successful refits raises an explicit error; a block with fewer than 80% successful refits is marked `bootstrap_unstable`.

- [ ] **Step 6: Run the analysis tests**

Run: `uv run pytest tests/frontier_eci/test_canonical_analysis.py tests/frontier_eci/test_analysis.py tests/test_ranking.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the calibration implementation**

```bash
git add scripts/frontier_eci/canonical_analysis.py tests/frontier_eci/test_canonical_analysis.py
git commit -m "Fit canonical-anchor frontier calibration"
git push origin judge-dissect
```

---

### Task 3: Generate resumable low-item judging configs

**Files:**
- Create: `scripts/frontier_eci/write_topup_configs.py`
- Create: `tests/frontier_eci/test_write_topup_configs.py`
- Create: `configs/frontier_eci_canonical_anchors/historical_backfill_swe.yaml`
- Modify: `scripts/frontier_eci/run_budgeted.py`
- Modify: `tests/frontier_eci/test_run_budgeted.py`
- Generate: `configs/frontier_eci_canonical_anchors/runs/<bench>/<judge>/<pair>.yaml`

**Interfaces:**
- Consumes: `coverage/pair_items.csv` and existing `configs/frontier_eci/runs/<bench>/<judge>.yaml` transcript maps.
- Produces: one ordinary `run_judging.py` config per judge/benchmark/unordered pair, each containing one comparison and its exact `only_items`; `run_budgeted.py` consumes those configs unchanged.

- [ ] **Step 1: Write config-emission tests**

```python
def test_each_config_has_one_pair_and_frozen_items(tmp_path):
    outputs = write_configs(manifest_frame(), base_configs(), tmp_path)
    config = yaml.safe_load(outputs[0].read_text())
    assert len(config["comparisons"]) == 1
    assert config["only_items"] == ["i1", "i2", "i3"]
    assert config["include_reasoning"] is False
    assert config["phase1_items"] == 8
    assert config["cache"] is False
```

- [ ] **Step 2: Write cumulative-budget tests**

```python
def test_historical_spend_reduces_incremental_allowance():
    assert remaining_budget(total_limit=200, historical_spend=133.4) == 66.6


def test_pair_is_not_started_without_worst_case_reserve():
    assert not can_afford(spent=63.0, reserve=4.0, limit=66.6)
```

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/frontier_eci/test_write_topup_configs.py tests/frontier_eci/test_run_budgeted.py -q`

Expected: FAIL because config emission and cumulative-budget accounting are absent.

- [ ] **Step 4: Implement config emission and budget accounting**

Emit one-pair configs with the approved prompts, hidden reasoning, no cache, exact item allowlist, and output root `logs/frontier-eci-canonical-anchors/<bench>`. Preserve judge endpoint settings in `configs/models`; do not duplicate secrets or pins into generated YAML. Extend the ledger header with `historical_spend=133.4`, `total_limit=200`, and computed incremental allowance `66.6`.

The SWE backfill config is fixed to the four protocol-compatible historical Epoch artifacts needed by GPT-4o-era blocks:

```yaml
out_dir: external_data/frontier-eci-canonical-anchors/epoch-swe
models:
  H2sQhavY5BPYxyYHva8mgS: claude-3.5-sonnet-0620
  GUbGX7GABSbhxMXiYC5NFr: claude-3.5-sonnet-1022
  kvVyU6crkxAnG3DAUciMGc: o3-mini-0131-medium
  ne2Anciuxh2k3VXGYG4Nqy: gemini-2.0-flash-001
```

The GPQA backfill reuses the already imported `gpt-4-1106`, `claude-3-opus`, `claude-3.5-sonnet-0620`, and `gpt-4o-0806` logs named in `configs/frontier_eci/early_anchor_gpqa.yaml`; it does not download or regenerate GPQA rollouts.

- [ ] **Step 5: Generate configs and dry-run locally without API calls**

Run:

```bash
uv run python scripts/frontier_eci/write_topup_configs.py \
  --config configs/frontier_eci_canonical_anchors/experiment.yaml
for pair_config in configs/frontier_eci_canonical_anchors/runs/gpqa/gpt-5.2/*.yaml
do
  uv run python scripts/judge/run_judging.py --config "$pair_config" --dry-run
done
```

Expected: every listed phase contains at most eight item IDs, hence at most sixteen counterbalanced samples.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/frontier_eci/test_write_topup_configs.py tests/frontier_eci/test_run_budgeted.py tests/frontier_eci/test_judges.py -q`

Expected: PASS.

- [ ] **Step 7: Commit config generation**

```bash
git add scripts/frontier_eci/write_topup_configs.py scripts/frontier_eci/run_budgeted.py tests/frontier_eci/test_write_topup_configs.py tests/frontier_eci/test_run_budgeted.py configs/frontier_eci_canonical_anchors/historical_backfill_swe.yaml configs/frontier_eci_canonical_anchors/runs
git commit -m "Generate low-item canonical anchor topups"
git push origin judge-dissect
```

---

### Task 4: Stage and run the top-up on RunPod

**Files:**
- Create: `scripts/frontier_eci/runpod/run_canonical_topup.sh`
- Create: `scripts/frontier_eci/runpod/run_canonical_analysis.sh`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/ledger.jsonl`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/provenance/endpoints-<timestamp>.json`

**Interfaces:**
- Consumes: committed code/configs, the new personal OpenRouter key from the laptop `.env`, the RunPod API key resolved from `.env`, and staged transcript/result inputs.
- Produces: resumable Inspect logs, RunPod job logs, budget ledger, endpoint snapshot, and per-response provider provenance.

- [ ] **Step 1: Load the RunPod and OpenRouter safety instructions before the first infrastructure or API call**

Read the RunPod router, matching pod golden path, runpodctl, and OpenRouter safety skills completely. Follow their current auth, pod lifecycle, transfer, secret-handling, endpoint-pin, and provenance requirements. Treat live `runpodctl --help` output as authoritative for CLI syntax.

- [ ] **Step 2: Audit the five judge model YAMLs and snapshot OpenRouter endpoints**

Run the OpenRouter safety scanner and inspect each critical route. Confirm `allow_fallbacks: false`, `data_collection: deny`, the intended provider/endpoint, adequate context/output limits, and compatible required parameters. Save the endpoint responses under the new provenance root before judging.

- [ ] **Step 3: Provision or reuse a cost-guarded RunPod and stage required data**

Run `runpodctl user` before the first billable action and record the account identity without exposing either key. Reuse the prior stopped project pod only if it belongs to this account and retains sufficient disk; otherwise create the cheapest available CPU pod with at least 8 vCPU, 32 GB RAM, and 80 GB disk, or the cheapest GPU pod meeting those host-memory/disk requirements when CPU inventory is unavailable. Record pod ID, image, resources, storage, and hourly price. Clone `canonical-anchor-redo`, then transfer only the frozen identity/quartile tables, source transcript logs referenced by generated configs, existing verdict parquet files, and new configs; do not copy unrelated T7 collections. Transfer the new OpenRouter key via the authenticated pod channel into a mode-600 `.env`; never print it, include it in commands captured by the ledger, or commit it.

Download the four public historical SWE artifacts on the pod into the isolated source directory:

```bash
curl -fL --retry 3 -o external_data/frontier-eci-canonical-anchors/epoch-swe/H2sQhavY5BPYxyYHva8mgS.eval https://epoch-benchmarks-production-public.s3.us-east-2.amazonaws.com/inspect_ai_logs/H2sQhavY5BPYxyYHva8mgS.eval
curl -fL --retry 3 -o external_data/frontier-eci-canonical-anchors/epoch-swe/GUbGX7GABSbhxMXiYC5NFr.eval https://epoch-benchmarks-production-public.s3.us-east-2.amazonaws.com/inspect_ai_logs/GUbGX7GABSbhxMXiYC5NFr.eval
curl -fL --retry 3 -o external_data/frontier-eci-canonical-anchors/epoch-swe/kvVyU6crkxAnG3DAUciMGc.eval https://epoch-benchmarks-production-public.s3.us-east-2.amazonaws.com/inspect_ai_logs/kvVyU6crkxAnG3DAUciMGc.eval
curl -fL --retry 3 -o external_data/frontier-eci-canonical-anchors/epoch-swe/ne2Anciuxh2k3VXGYG4Nqy.eval https://epoch-benchmarks-production-public.s3.us-east-2.amazonaws.com/inspect_ai_logs/ne2Anciuxh2k3VXGYG4Nqy.eval
```

Trim them to the frozen 46 SWE-mini items and 100-message cap with the existing importer:

```bash
uv run python scripts/judge/trim_epoch_swe_logs.py \
  --config configs/frontier_eci_canonical_anchors/historical_backfill_swe.yaml \
  --items-from 'data/swe_bench_mini/477dc96c/claude-opus-4.7/0/*.eval' \
  --target data/swe_bench_mini/frontier-eci-canonical-anchors \
  --message-limit 100
```

Regenerate the coverage audit and one-pair configs on the pod after the backfill paths exist; verify that backfill enters only blocks still below four canonical anchors. Run a real `run_judging.py --dry-run` against staged `.eval` sources before any paid call.

- [ ] **Step 4: Submit the resumable paid top-up**

The RunPod script activates the project environment and runs every generated one-pair config through:

```bash
uv run python scripts/frontier_eci/run_budgeted.py \
  configs/frontier_eci_canonical_anchors/runs/*/*/*.yaml \
  --baseline "$OPENROUTER_BASELINE_USAGE" \
  --limit 66.6 \
  --preflight judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/preflight.csv \
  --ledger judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/ledger.jsonl
```

Set the task-specific baseline variable from the key-usage response immediately before submission; never reuse `$HOME`, `$home`, or `$CODEX_HOME`.

- [ ] **Step 5: Monitor every 20 minutes while the job is active**

At each check, record RunPod state, completed/failed pair count, OpenRouter incremental usage, projected final spend, pod cost, and unexpected-provider count. Resume interrupted work from completed logs; do not rerun successful pair directories. Stop the pod whenever the workload is not actively progressing.

- [ ] **Step 6: Audit completed API calls before analysis**

Run:

```bash
uv run python scripts/frontier_eci/audit_api.py \
  logs/frontier-eci-canonical-anchors \
  --out judge_results/frontier/eci-semiannual-canonical-anchors/provenance
```

Expected: every paid response has provider provenance, every provider matches the intended route, and cumulative estimated project spend is at most $200.

- [ ] **Step 7: Commit and push run manifests, not secrets or bulky logs**

```bash
git add scripts/frontier_eci/runpod judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod judge_results/frontier/eci-semiannual-canonical-anchors/provenance
git commit -m "Record canonical anchor topup run"
git push origin judge-dissect
```

---

### Task 5: Produce judge forecasts, matched ECI baselines, and release-time figures

**Files:**
- Create: `scripts/frontier_eci/canonical_capability_baseline.py`
- Create: `tests/frontier_eci/test_canonical_capability_baseline.py`
- Modify: `scripts/frontier_eci/canonical_analysis.py`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/analysis/`
- Generate: `judge_results/frontier/eci-semiannual-canonical-anchors/capability_baseline/`

**Interfaces:**
- Consumes: audited new verdict logs plus every compatible existing canonical/frontier verdict and the frozen coverage tables.
- Produces: `fit_eci_blocks(judge_fits: pd.DataFrame, eci: pd.Series, hard_outcomes: pd.DataFrame, targets: dict[str, str]) -> pd.DataFrame`; `predictions.csv`, `anchor_fits.csv`, `anchor_residuals.csv`, `metrics.csv`, `coverage.csv`, ECI analogues, and per-judge/per-benchmark/per-quartile PDF/PNG figures.

- [ ] **Step 1: Write matched-anchor ECI tests**

```python
def test_eci_baseline_uses_exactly_the_judge_blocks_anchor_ids():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "e": 130.0, "target": 140.0})
    outcomes = pd.DataFrame({"model": ["a", "c", "d", "e", "target"],
                             "successes": [2, 7, 13, 18, 0], "total": [20] * 5})
    result = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    assert result.iloc[0].anchor_ids == "a|c|d|e"


def test_eci_baseline_excludes_target_truth():
    judge_fit = pd.DataFrame({"block": ["b"], "anchor_ids": ["a|c|d|e"]})
    eci = pd.Series({"a": 100.0, "c": 110.0, "d": 120.0, "e": 130.0, "target": 140.0})
    outcomes = pd.DataFrame({"model": ["a", "c", "d", "e", "target"],
                             "successes": [2, 7, 13, 18, 0], "total": [20] * 5})
    first = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    outcomes.loc[outcomes.model.eq("target"), "successes"] = 20
    second = fit_eci_blocks(judge_fit, eci, outcomes, targets={"b": "target"})
    assert first.iloc[0].predicted == second.iloc[0].predicted
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/frontier_eci/test_canonical_capability_baseline.py -q`

Expected: FAIL because the matched baseline does not exist.

- [ ] **Step 3: Implement matched ECI fitting and complete analysis outputs**

Use the exact anchor identity list exported by each judge block; fail visibly if an ECI value is unavailable rather than silently changing the set. Reuse `fit_binomial_logit` with ECI as x. Export judge and baseline equations as plain CSV columns (`intercept`, `slope`), not LaTeX text in figures.

- [ ] **Step 4: Run the full analysis on RunPod**

Run `scripts/frontier_eci/runpod/run_canonical_analysis.sh` with 1,000 bootstrap resamples. Expected outputs include all five judges, both benchmarks, all three hard quartiles, an explicit reason for every missing prediction, and separate capped/uncapped sensitivity metrics.

- [ ] **Step 5: Generate the release-time figures**

Reuse `plot_frontier_style` so each plot has release time on x, actual hard accuracy on y, blue predictions, black actuals, residual connectors, and the judge-release vertical line. Write judge figures under `analysis/figures/frontier_style/` and ECI figures under `capability_baseline/figures/`.

- [ ] **Step 6: Run analysis tests**

Run:

```bash
uv run pytest tests/frontier_eci/test_canonical_analysis.py \
  tests/frontier_eci/test_canonical_capability_baseline.py \
  tests/frontier_eci/test_analysis.py \
  tests/frontier_eci/test_capability_baseline.py -q
```

Expected: PASS.

- [ ] **Step 7: Pull only final tables and figures to the laptop and inspect representative PNGs**

Inspect GPQA and SWE Q2/Q3/Q4 plots for GPT-4o, Gemini 2.5 Pro, GPT-5.2, o4-mini, and DeepSeek V4 Flash at original resolution. Verify readable text, consistent axes, correct release lines, no clipped labels, and no unexplained missing points. Regenerate on RunPod after any plotting fix, recover compact outputs, then stop the pod.

- [ ] **Step 8: Commit analysis code and compact result artifacts**

```bash
git add scripts/frontier_eci/canonical_analysis.py scripts/frontier_eci/canonical_capability_baseline.py tests/frontier_eci/test_canonical_analysis.py tests/frontier_eci/test_canonical_capability_baseline.py judge_results/frontier/eci-semiannual-canonical-anchors/analysis judge_results/frontier/eci-semiannual-canonical-anchors/capability_baseline
git commit -m "Plot canonical-anchor frontier forecasts"
git push origin judge-dissect
```

---

### Task 6: Verify, document, and hand off the noncanonical result

**Files:**
- Create: `docs/results/2026-08-30-canonical-anchor-frontier-redo.md`
- Modify: `docs/results/2026-08-28-frontier-eci-side-analysis.md`
- Verify: all files under `judge_results/frontier/eci-semiannual-canonical-anchors/`

**Interfaces:**
- Consumes: final tables, figures, tests, provenance, and cost ledger.
- Produces: a concise side-analysis report naming the valid figures, discarded predecessors, fit equations, anchor coverage, errors, limitations, and exact spend.

- [ ] **Step 1: Run the complete relevant test set**

Run: `uv run pytest tests/frontier_eci tests/test_ranking.py tests/test_frontier_forecast.py -q`

Expected: PASS with no skipped new tests.

- [ ] **Step 2: Run invariant checks over final tables**

Assert programmatically that every reported fit has at least four anchors, no target in `anchor_ids`, `abs(gradient_intercept) < 1e-6`, both residual signs unless all residuals are numerical zero, correct temporal eligibility, both presentation orders, and no more than eight item IDs per pair in the capped fit.

- [ ] **Step 3: Write the results report**

State that the work is noncanonical. Report capped and uncapped results, judge-versus-ECI comparison, per-model residuals, inside/outside-anchor-range metrics, missing-cell reasons, intercept/slope tables, provider provenance, and exact incremental and cumulative spend. Mark all earlier `eci-semiannual/analysis*` plots as discarded.

- [ ] **Step 4: Verify repository cleanliness only for task-owned paths**

Run: `git status --short` and inspect the diff for the config, scripts, tests, spec/plan, result report, and compact result directories. Leave unrelated user changes untouched.

- [ ] **Step 5: Commit and push the final write-up**

```bash
git add docs/results/2026-08-30-canonical-anchor-frontier-redo.md docs/results/2026-08-28-frontier-eci-side-analysis.md
git commit -m "Document canonical-anchor frontier redo"
git push origin judge-dissect
```

- [ ] **Step 6: Hand off exact figure and table paths**

Report the valid GPQA/SWE judge-plot roots, ECI-baseline roots, calibration residual tables, coverage audit, provenance summary, cumulative cost, test command/output, and final commit hash. Explicitly name any scientifically unidentified cell rather than describing the run as complete without qualification.
