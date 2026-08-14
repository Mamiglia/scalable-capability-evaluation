# Monte Carlo Cost Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validated posterior-predictive simulator that estimates pairwise-judging results, token usage, and cost for automatic `J`/`M`/`N` designs and explicit `(judge, M1, M2)` comparisons.

**Architecture:** Extend the in-progress fixed-floor IRT analysis so it emits model ability and posterior draws, then prepare one versioned judgment table and two role-specific token tables. Fit the approved hierarchical models directly in NumPyro, simulate and validate in Python, and export exact precomputed scenario summaries to a static Plotly dashboard; the browser never reimplements the statistical model.

**Tech Stack:** Python 3.13+, Pandas, NumPy, NumPyro/JAX, SciPy, scikit-learn, Inspect eval logs, pytest, stdlib `argparse`/JSON/CSV, existing Bradley–Terry helpers, static HTML/CSS/JavaScript, local Plotly.

**Spec:** `docs/superpowers/specs/2026-08-14-monte-carlo-cost-simulator-design.md`

## Global Constraints

- Preserve the user's current uncommitted changes in `src/irt/data.py`, `src/irt/fitting.py`, and `scripts/irt_proxy/`; integrate with them instead of replacing them.
- Automatic mode uses one shared set of `M` anchors for all `J` judges. Every anchor is newer than every selected judge and, by default, has higher `theta`; each anchor's `N` distinct comparators are restricted only to being older than that anchor.
- Manual `(judge, M1, M2)` rows retain duplicates and run even when they violate automatic sampling rules; violations become warnings.
- Use fixed-floor GPQA IRT `theta` for capability. Never insert raw accuracy or ECI directly into a `theta` difference.
- A model without `theta` or compatible GPQA correct/total counts, or without a release date, is ineligible for automatic sampling. Missing metadata must not be silently imputed.
- Observed judges/models use identity posteriors. Unseen identities receive one population residual draw per Monte Carlo trial, reused across their rows in that trial.
- Enable capability/date-conditioned unseen-judge effects only when leave-one-judge-out log loss beats the pooled-judge baseline beyond cross-validation uncertainty.
- `attempted_items` is a fixed pre-filter item budget. Both-correct filtering makes usable-item and actual judgment-call counts random; do not sample until a usable target is reached.
- Report 5th/median/95th posterior-predictive percentiles for every forecasted choice, ranking, usable-count, token, and cost quantity.
- Treat `reasoning_tokens` as a subset of billed `output_tokens`. Derive ordinary output as `max(output_tokens - reasoning_tokens, 0)` and price total output once.
- Do not add a live service, database, job queue, neural text scorer, model-discovery API, or pricing API.
- `scripts/hibayes/model.py` is the formula reference only. The `hibayes` package is absent from `pyproject.toml` and the current environment; port the hierarchy to direct NumPyro and do not add an unversioned dependency.
- The static dashboard consumes Python-generated scenario summaries. Changing durable model metadata or exported pool definitions requires rerunning `mc-cost export-dashboard`, keeping one authoritative simulation engine.
- Full model fitting is offline. Unit tests use traces, synthetic posterior arrays, or tiny MCMC settings; they do not run production 1,000/1,000 fits.

---

## File Map

### Create

- `src/mc_cost/__init__.py` — public package version and exported types.
- `src/mc_cost/catalogue.py` — canonical model catalogue, metadata validation, and accuracy-to-`theta` inversion.
- `src/mc_cost/data.py` — judgment feature preparation and role-specific token-row extraction.
- `src/mc_cost/model.py` — judgment NumPyro model, posterior prediction, and artifact persistence.
- `src/mc_cost/tokens.py` — candidate/judge token models and token prediction.
- `src/mc_cost/design.py` — automatic and manual comparison selection.
- `src/mc_cost/simulation.py` — posterior-predictive trials, cost accounting, quantiles, and dashboard export rows.
- `src/mc_cost/validation.py` — held-out validation, model selection, support diagnostics, and ranking recovery.
- `src/mc_cost/cli.py` — `catalogue`, `prepare`, `fit`, `validate`, `simulate`, and `export-dashboard` commands.
- `configs/mc_cost/default.json` — reproducible paths, fit sizes, pools, sweep grid, prices, and seeds.
- `tests/test_mc_cost_catalogue.py`
- `tests/test_mc_cost_data.py`
- `tests/test_mc_cost_model.py`
- `tests/test_mc_cost_tokens.py`
- `tests/test_mc_cost_design.py`
- `tests/test_mc_cost_simulation.py`
- `tests/test_mc_cost_validation.py`
- `tests/test_mc_cost_cli.py`
- `tests/test_mc_cost_dashboard.py`
- `interactive_viewer/mc_cost.html`
- `interactive_viewer/mc_cost.js`
- `interactive_viewer/mc_cost.css`

### Modify

- `scripts/irt_proxy/item_banks.py` — retain model `theta` summaries and exact fixed-3PL posterior arrays.
- `scripts/irt_proxy/run.py` — write `fit_capabilities.csv` and `irt_posterior.npz`.
- `scripts/irt_proxy/test_item_banks.py`
- `scripts/irt_proxy/test_run.py`
- `pyproject.toml` — add the `mc-cost` entry point only; add no dependency.
- `interactive_viewer/index.html` — add a link to the simulator page.
- `docs/plans/2026-08-14-MC-cost-simulation.md` — replace the unfinished draft ending with status and run instructions.

### Generated, not committed

- `analysis/mc_cost/catalogue.csv`
- `analysis/mc_cost/capability_draws.npz`
- `analysis/mc_cost/judgments.parquet`
- `analysis/mc_cost/token_candidate.parquet`
- `analysis/mc_cost/token_judge.parquet`
- `analysis/mc_cost/model/*.npz`
- `analysis/mc_cost/model/*.json`
- `analysis/mc_cost/validation/*.csv`
- `analysis/mc_cost/validation/decision.json`
- `analysis/mc_cost/runs/<run-id>/{trials.parquet,summary.csv,warnings.csv,config.json}`
- `interactive_viewer/data/mc_cost.json`

---

### Task 1: Expose Fixed-3PL Model Ability and Posterior Draws

**Files:**
- Modify: `scripts/irt_proxy/item_banks.py`
- Modify: `scripts/irt_proxy/run.py`
- Test: `scripts/irt_proxy/test_item_banks.py`
- Test: `scripts/irt_proxy/test_run.py`

**Interfaces:**
- Consumes: `irt.fitting.fit(...) -> dict[str, np.ndarray]` and encoded `data.models`/`data.items`.
- Produces: `fit_parameters(...) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]`, `fit_capabilities.csv`, and `irt_posterior.npz` with `fixed_3pl__gpqa__theta/a/b/floor` plus model/item labels.

- [ ] **Step 1: Write failing capability/posterior tests**

Add to `scripts/irt_proxy/test_item_banks.py`:

```python
def test_fit_parameters_returns_model_theta_and_gpqa_posterior(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": model, "benchmark": benchmark, "item": item, "epoch": 1,
         "answer": "A", "correct": model == "m1", "source_priority": 1}
        for model in ("m1", "m2")
        for benchmark, item in (
            ("gpqa", "g1"), ("gpqa", "g2"),
            ("aime2025", "a25"), ("aime2026", "a26"),
        )
    ])

    def fake_fit(data, **_kwargs):
        return {
            "theta": np.array([[1.0, -1.0], [1.2, -0.8]]),
            "a": np.ones((2, len(data.items))),
            "b": np.zeros((2, len(data.items))),
        }

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    difficulties, capabilities, posteriors = item_banks.fit_parameters(
        responses, floors={"gpqa": 0.25, "aime2025": 0.001, "aime2026": 0.001},
        warmup=2, samples=2, seed=7,
    )

    gpqa = capabilities.query("irt_model == 'fixed_3pl' and fit == 'gpqa'")
    assert dict(zip(gpqa["model"], gpqa["theta"])) == {"m1": 1.1, "m2": -0.9}
    assert set(difficulties["fit"]) == {"gpqa", "aime", "joint"}
    assert posteriors["fixed_3pl__gpqa"]["models"] == ["m1", "m2"]
```

Add to `scripts/irt_proxy/test_run.py`:

```python
def test_write_irt_posterior_round_trips_without_pickle(tmp_path):
    import numpy as np
    from run import write_irt_posterior

    path = tmp_path / "irt_posterior.npz"
    write_irt_posterior(path, {
        "fixed_3pl__gpqa": {
            "theta": np.array([[1.0, -1.0]]),
            "a": np.array([[1.2]]),
            "b": np.array([[0.3]]),
            "floor": np.array([0.25]),
            "models": ["m1", "m2"],
            "items": [("gpqa", "i1")],
        }
    })

    stored = np.load(path, allow_pickle=False)
    assert stored["fixed_3pl__gpqa__models"].tolist() == ["m1", "m2"]
    assert stored["fixed_3pl__gpqa__theta"].shape == (1, 2)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
uv run pytest scripts/irt_proxy/test_item_banks.py::test_fit_parameters_returns_model_theta_and_gpqa_posterior scripts/irt_proxy/test_run.py::test_write_irt_posterior_round_trips_without_pickle -v
```

Expected: FAIL because `fit_parameters` and `write_irt_posterior` do not exist.

- [ ] **Step 3: Implement one-pass parameter retention**

In `scripts/irt_proxy/item_banks.py`, factor the current fitting loop into:

```python
def fit_parameters(responses, *, floors, warmup=1000, samples=1000, seed=0):
    cells = irt_cells(responses)
    difficulty_rows, capability_rows, posteriors = [], [], {}
    for irt_model in ("fixed_3pl", "2pl"):
        for fit_name, benchmarks in FIT_BENCHMARKS.items():
            data = encode(cells[cells["benchmark"].isin(benchmarks)])
            item_floors = np.array([floors[benchmark] for benchmark, _ in data.items])
            kwargs = {"lower_asymptote": item_floors} if irt_model == "fixed_3pl" else {}
            posterior = fit(data, model=irt_model, warmup=warmup,
                            samples=samples, seed=seed, **kwargs)

            b = summary(data.items, posterior["b"])
            difficulty = pd.DataFrame(b.pop("name").tolist(), columns=["benchmark", "item"])
            difficulty.insert(0, "fit", fit_name)
            difficulty.insert(0, "irt_model", irt_model)
            difficulty["lower_asymptote"] = item_floors if irt_model == "fixed_3pl" else 0.0
            difficulty["difficulty"] = b["mean"]
            difficulty["difficulty_p5"] = b["p5"]
            difficulty["difficulty_p95"] = b["p95"]
            difficulty["discrimination"] = np.asarray(posterior["a"]).mean(axis=0)
            difficulty_rows.append(difficulty)

            theta = summary(data.models, posterior["theta"])
            capability_rows.append(pd.DataFrame({
                "irt_model": irt_model, "fit": fit_name,
                "model": theta["name"], "theta": theta["mean"],
                "theta_p5": theta["p5"], "theta_p95": theta["p95"],
            }))
            posteriors[f"{irt_model}__{fit_name}"] = {
                "theta": np.asarray(posterior["theta"]),
                "a": np.asarray(posterior["a"]),
                "b": np.asarray(posterior["b"]),
                "floor": item_floors if irt_model == "fixed_3pl" else np.zeros_like(item_floors),
                "models": list(data.models),
                "items": list(data.items),
            }
    return (pd.concat(difficulty_rows, ignore_index=True),
            pd.concat(capability_rows, ignore_index=True), posteriors)


def fit_difficulties(responses, **kwargs):
    return fit_parameters(responses, **kwargs)[0]
```

In `scripts/irt_proxy/run.py`, add `fit_capabilities.csv` to `EXPECTED_CSV` and exempt it from the component-column rule. Change `build_tables(...)` to return `(tables, posteriors)`: it calls `fit_parameters(...)` once, puts the capability frame in `tables["fit_capabilities.csv"]`, and uses the returned difficulty frame for the existing split analysis. Change `main` to pass only `tables` to `write_outputs`/`render_figures` and then call `write_irt_posterior(output_directory / "irt_posterior.npz", posteriors)`. Update existing `build_tables` tests to unpack the tuple. Write posterior labels using Unicode arrays rather than object arrays:

```python
def write_irt_posterior(path: Path, fits: dict[str, dict]) -> None:
    arrays = {}
    for prefix, values in fits.items():
        arrays[f"{prefix}__theta"] = values["theta"]
        arrays[f"{prefix}__a"] = values["a"]
        arrays[f"{prefix}__b"] = values["b"]
        arrays[f"{prefix}__floor"] = values["floor"]
        arrays[f"{prefix}__models"] = np.asarray(values["models"], dtype=str)
        arrays[f"{prefix}__items"] = np.asarray(
            [f"{benchmark}:{item}" for benchmark, item in values["items"]], dtype=str,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
```

- [ ] **Step 4: Run all IRT-proxy tests**

Run:

```bash
uv run pytest scripts/irt_proxy/test_item_banks.py scripts/irt_proxy/test_run.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/irt_proxy/item_banks.py scripts/irt_proxy/run.py scripts/irt_proxy/test_item_banks.py scripts/irt_proxy/test_run.py
git commit -m "feat: export IRT capability posteriors"
```

---

### Task 2: Build and Validate the Editable Model Catalogue

**Files:**
- Create: `src/mc_cost/__init__.py`
- Create: `src/mc_cost/catalogue.py`
- Create: `configs/mc_cost/default.json`
- Test: `tests/test_mc_cost_catalogue.py`

**Interfaces:**
- Consumes: `docs/canonical-models.md`, `model-sampler/model_pool.json`, `fit_capabilities.csv`, and `irt_posterior.npz`.
- Produces: `build_catalogue(...) -> pd.DataFrame`, `load_catalogue(path) -> pd.DataFrame`, `validate_catalogue(frame) -> None`, `theta_draws_from_accuracy(correct, total, irt_draws, rng) -> np.ndarray`, and `build_capability_draws(catalogue, irt_draws, rng) -> tuple[list[str], np.ndarray]` for `capability_draws.npz`.

- [ ] **Step 1: Write failing catalogue tests**

Create `tests/test_mc_cost_catalogue.py`:

```python
import json
import numpy as np
import pandas as pd
import pytest

from mc_cost.catalogue import (
    build_capability_draws, build_catalogue, theta_draws_from_accuracy, validate_catalogue,
)


def test_catalogue_joins_normalized_names_and_keeps_missing_metadata_visible(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- model-a\n- model-missing\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([{
        "name": "Model A", "release": "2025-01-02", "price_in": 1.0,
        "price_out": 4.0, "reasoning": False,
    }]))
    capabilities = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "model-a",
        "theta": 1.2, "theta_p5": 1.0, "theta_p95": 1.4,
    }])

    got = build_catalogue(canonical, pool, capabilities)

    assert got.loc[got.model_id == "model-a", "release"].item() == "2025-01-02"
    assert got.loc[got.model_id == "model-a", "theta"].item() == 1.2
    assert pd.isna(got.loc[got.model_id == "model-missing", "release"]).item()


def test_accuracy_to_theta_inverts_each_fixed_3pl_draw():
    draws = {"a": np.ones((3, 2)), "b": np.zeros((3, 2)), "floor": np.full(2, 0.25)}
    got = theta_draws_from_accuracy(75, 100, draws, np.random.default_rng(3))
    assert got.shape == (3,)
    assert np.all(np.isfinite(got))


def test_capability_draws_reuse_irt_and_do_not_impute_bare_point_estimate():
    catalogue = pd.DataFrame([
        {"model_id": "fitted", "theta": 1.0, "theta_source": "fixed_3pl"},
        {"model_id": "counted", "gpqa_correct": 7, "gpqa_total": 10,
         "theta_source": "gpqa_counts"},
        {"model_id": "bare", "theta": 0.0, "theta_source": "manual_point"},
    ])
    irt = {
        "models": ["fitted"], "theta": np.array([[0.9], [1.0], [1.1]]),
        "a": np.ones((3, 2)), "b": np.zeros((3, 2)), "floor": np.full(2, 0.25),
    }
    models, draws = build_capability_draws(catalogue, irt, np.random.default_rng(5))
    assert models == ["fitted", "counted"]
    assert draws.shape == (3, 2)
    assert np.allclose(draws[:, 0], [0.9, 1.0, 1.1])


def test_catalogue_rejects_duplicate_ids_and_bad_reasoning_class():
    frame = pd.DataFrame([
        {"model_id": "m", "reasoning_class": "reasoning", "release": "2025-01-01", "theta": 0.0},
        {"model_id": "m", "reasoning_class": "sometimes", "release": "2025-01-01", "theta": 0.0},
    ])
    with pytest.raises(ValueError, match="duplicate model_id|reasoning_class"):
        validate_catalogue(frame)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_catalogue.py -v`

Expected: FAIL because `mc_cost.catalogue` does not exist.

- [ ] **Step 3: Implement the minimal catalogue and theta inversion**

Use a DataFrame, not a catalogue class hierarchy. Define these exact columns in `src/mc_cost/catalogue.py`:

```python
CATALOGUE_COLUMNS = [
    "model_id", "display_name", "release", "theta", "theta_p5", "theta_p95",
    "theta_source", "gpqa_correct", "gpqa_total", "family", "reasoning_class", "judge_pool",
    "candidate_pool", "observed_judge", "observed_candidate", "price_in", "price_out",
]
```

Normalize names only for joining source files; preserve canonical `model_id` in outputs:

```python
def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def theta_draws_from_accuracy(correct, total, irt_draws, rng):
    if not (0 <= correct <= total) or total <= 0:
        raise ValueError("GPQA correct/total counts must satisfy 0 <= correct <= total")
    accuracy = rng.beta(correct + 0.5, total - correct + 0.5, size=len(irt_draws["a"]))
    result = np.empty_like(accuracy)
    for draw, target in enumerate(accuracy):
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = (lo + hi) / 2
            p = irt_draws["floor"] + (1 - irt_draws["floor"]) * expit(
                irt_draws["a"][draw] * (mid - irt_draws["b"][draw])
            )
            if p.mean() < target:
                lo = mid
            else:
                hi = mid
        result[draw] = (lo + hi) / 2
    return result
```

`build_capability_draws` aligns all eligible models to the same posterior-draw axis. It reuses exact `fixed_3pl__gpqa__theta` columns for fitted models, calls `theta_draws_from_accuracy` for rows with compatible GPQA counts, and converts explicitly supplied `theta_p5`/`theta_p95` intervals to normal draws for manual metadata. It does not use ECI or raw accuracy as `theta`, and it does not create draws from a point estimate with no uncertainty source. Save the model labels as Unicode plus a dense numeric `(draw, model)` matrix with `np.savez_compressed`, never object arrays.

`validate_catalogue` must reject duplicate IDs, unparsable dates, reasoning classes outside `{reasoning, non_reasoning}`, non-finite present `theta`, negative prices, and pool-enabled rows missing release or a supported capability uncertainty source (fixed-3PL posterior, GPQA counts, or explicit `theta_p5`/`theta_p95`). Rows outside pools may retain missing fields so the dashboard can explain exclusion.

Create `configs/mc_cost/default.json` with production defaults and no environment-specific absolute paths:

```json
{
  "paths": {
    "canonical_models": "docs/canonical-models.md",
    "model_pool": "model-sampler/model_pool.json",
    "irt_capabilities": "irt_results/easy_item_prediction/fit_capabilities.csv",
    "irt_items": "irt_results/easy_item_prediction/fit_difficulties.csv",
    "irt_posterior": "irt_results/easy_item_prediction/irt_posterior.npz",
    "judge_logs": "logs/imported_runs/gpqa_judging",
    "judge_models": "logs/imported_runs_2/gpqa-pilot-rd2/models.csv",
    "epoch_responses": "external_data/epoch-gpqa/responses.parquet",
    "candidate_log_roots": ["data/gpqa", "external_data/epoch-gpqa"],
    "output": "analysis/mc_cost"
  },
  "fit": {"warmup": 1000, "samples": 1000, "chains": 4, "seed": 0},
  "simulation": {
    "trials": 1000,
    "attempted_items": 198,
    "presentations": 2,
    "reuse_responses": true,
    "accuracy_target": "higher_theta",
    "seed": 0
  },
  "dashboard": {
    "judge_counts": [1, 2, 4, 8],
    "anchor_counts": [1, 2, 4, 8],
    "comparator_counts": [1, 2, 4, 8],
    "attempted_item_counts": [10, 25, 50, 100, 198],
    "presentation_counts": [1, 2],
    "judge_pool_column": "judge_pool",
    "candidate_pool_column": "candidate_pool",
    "manual_triples": [],
    "manual_triple_limit": 500
  }
}
```

- [ ] **Step 4: Run the catalogue tests**

Run: `uv run pytest tests/test_mc_cost_catalogue.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/__init__.py src/mc_cost/catalogue.py configs/mc_cost/default.json tests/test_mc_cost_catalogue.py
git commit -m "feat: add Monte Carlo model catalogue"
```

---

### Task 3: Prepare Judgment Features and Role-Specific Token Rows

**Files:**
- Create: `src/mc_cost/data.py`
- Test: `tests/test_mc_cost_data.py`

**Interfaces:**
- Consumes: the existing `scripts.hibayes.prepare_gpqa.prepare` result, catalogue rows, fixed-3PL difficulty rows, and Inspect `.eval` logs.
- Produces: `prepare_judgments(...) -> pd.DataFrame`, `candidate_token_rows(log_roots, catalogue, items) -> pd.DataFrame`, `judge_token_rows(judgments) -> pd.DataFrame`, and `write_prepared_tables(...)`.

- [ ] **Step 1: Write failing preparation tests**

Create `tests/test_mc_cost_data.py`:

```python
import pandas as pd

from mc_cost.data import judge_token_rows, prepare_judgments


def test_prepare_judgments_builds_signed_and_symmetric_features():
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "j", "item": "i", "left_model": "a", "right_model": "b",
        "length_difference": 0.4, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 1, "left_chars": 120, "right_chars": 80,
        "judge_input_tokens": 300, "judge_output_tokens": 40,
        "judge_reasoning_tokens": 10, "judge_total_tokens": 340,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": -0.5, "release": "2024-01-01", "family": "judge"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01", "family": "a"},
        {"model_id": "b", "theta": 0.0, "release": "2023-01-01", "family": "b"},
    ])
    items = pd.DataFrame([{"irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
                           "item": "i", "difficulty": 0.7}])

    got = prepare_judgments(attempts, catalogue, items)

    row = got.iloc[0]
    assert row.theta_gap == 1.0
    assert row.oversight == 1.0
    assert row.theta_gap_x_oversight == 1.0
    assert row.theta_gap_x_difficulty == 0.7


def test_judge_tokens_define_reasoning_as_output_subset():
    rows = pd.DataFrame([{
        "judge": "j", "item": "i", "chosen_left": 1, "left_chars": 100, "right_chars": 200,
        "judge_input_tokens": 500, "judge_output_tokens": 80,
        "judge_reasoning_tokens": 30, "judge_total_tokens": 580,
    }])
    got = judge_token_rows(rows)
    assert got.iloc[0].ordinary_output_tokens == 50
    assert got.iloc[0].output_tokens == 80
    assert got.iloc[0].transcript_chars == 300
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_data.py -v`

Expected: FAIL because `mc_cost.data` does not exist.

- [ ] **Step 3: Implement deterministic feature preparation**

Use release days relative to `2020-01-01`, then standardize every continuous predictor from training rows and save the means/scales beside the prepared table. The unscaled table must retain:

```python
JUDGMENT_COLUMNS = [
    "chosen_left", "judge", "item", "left_model", "right_model",
    "theta_gap", "oversight", "release_gap_days", "relative_release_days",
    "difficulty", "theta_gap_x_oversight", "theta_gap_x_difficulty",
    "length_difference", "same_family_x_theta_gap", "self_comparison",
    "reasoning_difference", "left_chars", "right_chars",
]
```

Implement the signed features directly:

```python
frame["theta_gap"] = frame["left_theta"] - frame["right_theta"]
frame["oversight"] = (frame["left_theta"] + frame["right_theta"]) / 2 - frame["judge_theta"]
frame["release_gap_days"] = frame["left_release_day"] - frame["right_release_day"]
frame["relative_release_days"] = (
    (frame["left_release_day"] + frame["right_release_day"]) / 2
    - frame["judge_release_day"]
)
frame["theta_gap_x_oversight"] = frame["theta_gap"] * frame["oversight"]
frame["theta_gap_x_difficulty"] = frame["theta_gap"] * frame["difficulty"]
frame["same_family_x_theta_gap"] = frame["same_model_family"] * frame["theta_gap"]
```

Extract candidate usage by reusing `inspect_ai.log.list_eval_logs`/`read_eval_log` and the `sample_tokens`/`sample_correct` semantics in `scripts/cost/eval_cost.py`. Join catalogue and item metadata, then emit one row per deduplicated `(benchmark, model, item, epoch)` with role `candidate`, `response_chars`, correctness, model `theta`, model release day, item difficulty, visible-reasoning indicator, and token fields. Emit judge rows from the prepared attempts with role `judge`, `transcript_chars = left_chars + right_chars`, and `ordinary_output_tokens = max(output_tokens - reasoning_tokens, 0)`.

Reject, with model/item names, any judgment row missing judge/left/right `theta`, release, or item difficulty. Count zero-usage/error candidate samples and exclude them from token fitting.

- [ ] **Step 4: Run data tests**

Run: `uv run pytest tests/test_mc_cost_data.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/data.py tests/test_mc_cost_data.py
git commit -m "feat: prepare simulator training data"
```

---

### Task 4: Fit the Hierarchical Judgment Model in NumPyro

**Files:**
- Create: `src/mc_cost/model.py`
- Test: `tests/test_mc_cost_model.py`

**Interfaces:**
- Consumes: standardized judgment rows from Task 3 and a fixed `JudgmentSpec(global_features, include_length)`.
- Produces: `encode_judgments(frame, catalogue, spec) -> JudgmentData`, `judgment_model(data, conditional_judges)`, `fit_judgment_model(...) -> PosteriorArtifact`, `left_logits(...) -> np.ndarray`, and `save_artifact`/`load_artifact`.

- [ ] **Step 1: Write failing trace and presentation tests**

Create `tests/test_mc_cost_model.py`:

```python
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from mc_cost.model import JudgmentData, judgment_model, left_logits


def tiny_data():
    return JudgmentData(
        y=jnp.array([1, 0]), judge=jnp.array([0, 0]), item=jnp.array([0, 0]),
        left_model=jnp.array([0, 1]), right_model=jnp.array([1, 0]),
        theta_gap=jnp.array([1.0, -1.0]),
        global_x=jnp.zeros((2, 9)), length_difference=jnp.array([0.2, -0.2]),
        judge_covariates=jnp.array([[0.0, 0.0]]), include_length=True,
        num_judges=1, num_items=1, num_models=2,
    )


def test_judgment_model_declares_pooled_judge_and_model_effects():
    trace = handlers.trace(handlers.seed(judgment_model, 2)).get_trace(tiny_data(), True)
    assert trace["judge_effects"]["value"].shape == (1, 2)
    assert trace["candidate_effects"]["value"].shape == (2,)
    assert trace["obs"]["value"].shape == (2,)


def test_swapping_candidates_negates_content_but_keeps_position_term():
    posterior = {
        "judge_effects": np.array([[[0.3, 1.5]]]),
        "global_beta": np.array([[0.2] + [0.0] * 8]),
        "length_mean": np.array([0.4]), "length_deviation": np.array([[0.1]]),
        "candidate_effects": np.array([[0.2, -0.1]]), "item_effects": np.array([[0.0]]),
    }
    forward = left_logits(posterior, tiny_data())[0, 0]
    swapped = left_logits(posterior, tiny_data())[0, 1]
    assert np.isclose((forward + swapped) / 2, 0.3)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_model.py -v`

Expected: FAIL because `mc_cost.model` does not exist.

- [ ] **Step 3: Implement the approved hierarchy**

Use these nine global columns in this order and store the order in artifact metadata:

```python
GLOBAL_FEATURES = (
    "oversight_sd", "release_gap_days_sd", "relative_release_days_sd",
    "difficulty_sd", "theta_gap_x_oversight_sd", "theta_gap_x_difficulty_sd",
    "same_family_x_theta_gap_sd", "self_comparison", "reasoning_difference_sd",
)
```

The model core must implement the approved pooling structure:

```python
def judgment_model(data: JudgmentData, conditional_judges: bool = True):
    covariates = data.judge_covariates if conditional_judges else data.judge_covariates[:, :0]
    judge_design = jnp.concatenate([jnp.ones((data.num_judges, 1)), covariates], axis=1)
    population = numpyro.sample(
        "judge_population", dist.Normal(0, 0.5).expand([2, judge_design.shape[1]]).to_event(2)
    )
    scales = numpyro.sample("judge_scales", dist.HalfNormal(0.5).expand([2]).to_event(1))
    corr = numpyro.sample("judge_corr", dist.LKJCholesky(2, 2.0))
    raw = numpyro.sample("judge_raw", dist.Normal(0, 1).expand([data.num_judges, 2]).to_event(2))
    judge_effects = judge_design @ population.T + (raw @ corr.T) * scales
    numpyro.deterministic("judge_effects", judge_effects)

    if data.global_x.shape[1]:
        global_beta = numpyro.sample(
            "global_beta", dist.Normal(0, 0.5).expand([data.global_x.shape[1]]).to_event(1)
        )
        global_term = jnp.sum(global_beta * data.global_x, axis=1)
    else:
        global_term = 0.0
    if data.include_length:
        length_mean = numpyro.sample("length_mean", dist.Normal(0, 0.5))
        length_scale = numpyro.sample("length_scale", dist.HalfNormal(0.5))
        length_raw = numpyro.sample(
            "length_raw", dist.Normal(0, 1).expand([data.num_judges]).to_event(1)
        )
        length_deviation = length_scale * length_raw
        numpyro.deterministic("length_deviation", length_deviation)
        length_term = (
            length_mean + length_deviation[data.judge]
        ) * data.length_difference
    else:
        length_term = 0.0
    candidate_scale = numpyro.sample("candidate_scale", dist.HalfNormal(0.5))
    candidate_raw = numpyro.sample(
        "candidate_raw", dist.Normal(0, 1).expand([data.num_models]).to_event(1)
    )
    candidate = candidate_scale * (candidate_raw - candidate_raw.mean())
    numpyro.deterministic("candidate_effects", candidate)
    item_scale = numpyro.sample("item_scale", dist.HalfNormal(0.5))
    item = item_scale * numpyro.sample(
        "item_raw", dist.Normal(0, 1).expand([data.num_items]).to_event(1)
    )
    numpyro.deterministic("item_effects", item)

    logits = (
        judge_effects[data.judge, 0]
        + judge_effects[data.judge, 1] * data.theta_gap
        + global_term
        + length_term
        + candidate[data.left_model] - candidate[data.right_model]
        + item[data.item]
    )
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=data.y)
```

`fit_judgment_model` uses `NUTS`, `MCMC`, the configured warmup/samples/chains/seed, saves divergences and `numpyro.diagnostics.summary` to JSON, and saves numeric posterior arrays with `np.savez_compressed(..., allow_pickle=False-compatible dtypes)`. Metadata JSON stores feature order, category maps, standardization means/scales, training ranges, source hashes, and `conditional_judges`.

Implement `left_logits` as a pure NumPy function used by both simulation and tests. Do not duplicate the linear predictor in the CLI or dashboard.

- [ ] **Step 4: Run model tests**

Run: `uv run pytest tests/test_mc_cost_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/model.py tests/test_mc_cost_model.py
git commit -m "feat: fit hierarchical judgment model"
```

---

### Task 5: Fit Candidate Transcript-Length and Role-Specific Token Models

**Files:**
- Create: `src/mc_cost/tokens.py`
- Test: `tests/test_mc_cost_tokens.py`

**Interfaces:**
- Consumes: candidate/judge token rows and catalogue reasoning classes.
- Produces: `fit_token_model(frame, role, ...) -> TokenArtifact`, `draw_transcript_lengths(candidate_artifact, rows, rng, posterior_index, latent_cache=None) -> np.ndarray`, and `draw_token_usage(artifact, rows, rng, posterior_index, latent_cache=None) -> pd.DataFrame` with `input_tokens`, `output_tokens`, `reasoning_tokens`, and `ordinary_output_tokens`.

- [ ] **Step 1: Write failing token-invariant tests**

Create `tests/test_mc_cost_tokens.py`:

```python
import numpy as np
import pandas as pd

from mc_cost.tokens import draw_token_usage, draw_transcript_lengths


def test_reasoning_is_a_subset_of_billed_output():
    artifact = {
        "role": "judge", "models": ["seen"], "reasoning_classes": ["reasoning"],
        "posterior": {
            "input_mean": np.array([[np.log1p(500)]]),
            "output_mean": np.array([[np.log1p(100)]]),
            "reasoning_logit": np.array([[0.0]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{"model": "seen", "reasoning_class": "reasoning", "transcript_chars": 500}])
    got = draw_token_usage(artifact, rows, np.random.default_rng(1), posterior_index=0)
    assert got.loc[0, "output_tokens"] == 100
    assert got.loc[0, "reasoning_tokens"] == 50
    assert got.loc[0, "ordinary_output_tokens"] == 50


def test_non_reasoning_has_structural_zero_reasoning_tokens():
    artifact = {
        "role": "candidate", "models": [], "reasoning_classes": ["non_reasoning"],
        "posterior": {"class_input_mean": np.array([[5.0]]), "class_output_mean": np.array([[4.0]]),
                      "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0])},
    }
    rows = pd.DataFrame([{"model": "unseen", "reasoning_class": "non_reasoning"}])
    got = draw_token_usage(artifact, rows, np.random.default_rng(2), posterior_index=0)
    assert got.loc[0, "reasoning_tokens"] == 0


def test_unseen_transcript_length_uses_reasoning_class_population():
    artifact = {
        "role": "candidate", "models": [],
        "reasoning_classes": ["non_reasoning", "reasoning"],
        "length_features": [
            "theta_sd", "release_day_sd", "difficulty_sd", "correct", "visible_reasoning",
        ],
        "posterior": {
            "length_class_mean": np.array([[np.log1p(100), np.log1p(200)]]),
            "length_beta": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            "length_model_scale": np.array([0.0]), "length_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{
        "model": "unseen", "reasoning_class": "reasoning",
        "theta_sd": 1.0, "release_day_sd": 1.0, "difficulty_sd": 1.0,
        "correct": 0, "visible_reasoning": 1,
    }])
    got = draw_transcript_lengths(
        artifact, rows, np.random.default_rng(4), posterior_index=0, latent_cache={},
    )
    assert got.tolist() == [200]
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_tokens.py -v`

Expected: FAIL because `mc_cost.tokens` does not exist.

- [ ] **Step 3: Implement one hierarchical token model per role**

For candidate rows, also fit `log1p(response_chars)` with reasoning-class means, partially pooled model-configuration intercepts, model `theta`, model release date, item difficulty, correctness, and visible reasoning. This is the conditional transcript-length distribution used only when no observed model-item length exists. Keep its posterior arrays in the candidate token artifact so fitting and simulation cannot accidentally pair incompatible length/token models.

Encode `log1p(input_tokens)` and `log1p(output_tokens)` with reasoning-class means, partially pooled model intercepts, item difficulty, correctness, and—for judge input only—`log1p(left_chars + right_chars)`. For reasoning configurations, model `logit((reasoning_tokens + 0.5) / (output_tokens + 1.0))`; non-reasoning rows do not enter that likelihood and predict zero reasoning.

Use the same artifact format as Task 4. Store observed model IDs, reasoning classes, feature order/scales, posterior arrays, and empirical 90% interval coverage. For unseen `(model, role)` combinations, draw one class-conditioned model deviation per trial and reuse it for every request in that trial; draw request residuals independently. Apply that same within-trial cache rule to unseen transcript-length effects.

Clamp simulated tokens to non-negative integers only after inverse transforms:

```python
output = np.maximum(0, np.rint(np.expm1(log_output))).astype(int)
if reasoning_class == "reasoning":
    reasoning = np.minimum(output, np.rint(output * expit(reasoning_logit)).astype(int))
else:
    reasoning = np.zeros_like(output)
ordinary = output - reasoning
```

- [ ] **Step 4: Run token tests**

Run: `uv run pytest tests/test_mc_cost_tokens.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/tokens.py tests/test_mc_cost_tokens.py
git commit -m "feat: model per-role token usage"
```

---

### Task 6: Implement Automatic and Manual Comparison Selection

**Files:**
- Create: `src/mc_cost/design.py`
- Test: `tests/test_mc_cost_design.py`

**Interfaces:**
- Consumes: validated catalogue and `AutomaticConfig` or manual rows.
- Produces: `automatic_comparisons(catalogue, config, rng) -> pd.DataFrame` and `manual_comparisons(catalogue, rows) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write failing sampling tests**

Create `tests/test_mc_cost_design.py`:

```python
import numpy as np
import pandas as pd
import pytest

from mc_cost.design import AutomaticConfig, automatic_comparisons, manual_comparisons


def catalogue():
    return pd.DataFrame([
        {"model_id": "j1", "theta": 0.0, "release": "2024-01-01", "judge_pool": True, "candidate_pool": False},
        {"model_id": "j2", "theta": 0.5, "release": "2024-06-01", "judge_pool": True, "candidate_pool": False},
        {"model_id": "a", "theta": 2.0, "release": "2026-01-01", "judge_pool": False, "candidate_pool": True},
        {"model_id": "old-strong", "theta": 0.3, "release": "2025-01-01", "judge_pool": False, "candidate_pool": True},
        {"model_id": "old-weak", "theta": -1.0, "release": "2023-01-01", "judge_pool": False, "candidate_pool": True},
    ])


def test_automatic_uses_shared_anchor_and_only_older_comparators():
    config = AutomaticConfig(j=2, m=1, n=2, attempted_items=10, presentations=2,
                             selected_judges=("j1", "j2"), require_stronger=True)
    got = automatic_comparisons(catalogue(), config, np.random.default_rng(4))
    assert set(got["judge"]) == {"j1", "j2"}
    assert got["m1"].nunique() == 1
    assert set(got["m2"]) == {"old-strong", "old-weak"}


def test_automatic_reports_eligible_count_instead_of_lowering_n():
    config = AutomaticConfig(j=1, m=1, n=9, attempted_items=10, presentations=2,
                             selected_judges=("j1",), require_stronger=True)
    with pytest.raises(ValueError, match="requested N=9.*eligible"):
        automatic_comparisons(catalogue(), config, np.random.default_rng(1))


def test_manual_retains_duplicates_and_warns_without_blocking():
    rows = pd.DataFrame([{"judge": "j2", "m1": "old-weak", "m2": "a"}] * 2)
    comparisons, warnings = manual_comparisons(catalogue(), rows)
    assert len(comparisons) == 2
    assert "automatic_rule_violation" in set(warnings["code"])


def test_manual_keeps_known_identity_with_missing_prediction_metadata():
    models = pd.concat([catalogue(), pd.DataFrame([{
        "model_id": "undated", "theta": np.nan, "release": None,
        "judge_pool": False, "candidate_pool": False,
    }])], ignore_index=True)
    rows = pd.DataFrame([{"judge": "j1", "m1": "a", "m2": "undated"}])
    comparisons, warnings = manual_comparisons(models, rows)
    assert comparisons.loc[0, "predictable"] == False
    assert "missing_prediction_metadata" in set(warnings["code"])
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_design.py -v`

Expected: FAIL because `mc_cost.design` does not exist.

- [ ] **Step 3: Implement exact selection rules**

Use one frozen dataclass:

```python
@dataclass(frozen=True)
class AutomaticConfig:
    j: int
    m: int
    n: int
    attempted_items: int
    presentations: int = 2
    selected_judges: tuple[str, ...] = ()
    require_stronger: bool = True
```

After selecting judges once, calculate anchors against the maximum selected-judge date and, when enabled, maximum selected-judge `theta`. Sample the `M` anchors once, then reuse their `N` comparator sets across every judge:

```python
anchor_mask = candidates["release_day"].gt(judges["release_day"].max())
if config.require_stronger:
    anchor_mask &= candidates["theta"].gt(judges["theta"].max())
anchors = rng.choice(candidates.loc[anchor_mask, "model_id"], config.m, replace=False)
```

Comparator eligibility is exactly `release_day < anchor_release_day`; do not filter by comparator `theta`, judge `theta`, family, or observed status. Manual mode requires only `judge,m1,m2` and accepts optional `judge_reasoning_class,m1_reasoning_class,m2_reasoning_class` overrides for configurations whose catalogue class is absent or intentionally changed. It rejects unknown identities, malformed columns, or invalid reasoning-class values. It retains row order and duplicates, adds a `predictable` boolean, and returns warnings for automatic-rule violations and missing/out-of-support features. A known row with missing prediction metadata remains in the output as a named partial result with no probability estimate; it is never silently dropped or imputed.

- [ ] **Step 4: Run design tests**

Run: `uv run pytest tests/test_mc_cost_design.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/design.py tests/test_mc_cost_design.py
git commit -m "feat: select automatic and manual comparisons"
```

---

### Task 7: Implement Posterior-Predictive Trials and Cost Accounting

**Files:**
- Create: `src/mc_cost/simulation.py`
- Test: `tests/test_mc_cost_simulation.py`

**Interfaces:**
- Consumes: comparison rows, attempted items, observed response cells, aligned capability draws, IRT item posterior, judgment artifact, token artifacts, catalogue prices, and seed.
- Produces: `SimulationResult(trials, summary, warnings)`, `simulate(...)`, and `summarize_trials(...)`.

- [ ] **Step 1: Write failing simulation tests**

Create `tests/test_mc_cost_simulation.py`:

```python
import numpy as np
import pandas as pd

from mc_cost.simulation import cost_rows, summarize_trials


def test_cost_prices_total_output_once():
    usage = pd.DataFrame([{
        "trial": 0, "role": "judge", "model": "j", "input_tokens": 100,
        "output_tokens": 80, "reasoning_tokens": 30, "ordinary_output_tokens": 50,
    }])
    catalogue = pd.DataFrame([{"model_id": "j", "price_in": 2.0, "price_out": 10.0}])
    got, warnings = cost_rows(usage, catalogue)
    assert np.isclose(got.loc[0, "cost"], (100 * 2 + 80 * 10) / 1_000_000)
    assert warnings.empty


def test_missing_price_keeps_calls_and_tokens_but_suppresses_currency():
    usage = pd.DataFrame([{
        "trial": 0, "role": "candidate", "model": "unpriced", "calls": 2,
        "input_tokens": 100, "output_tokens": 80,
        "reasoning_tokens": 30, "ordinary_output_tokens": 50,
    }])
    catalogue = pd.DataFrame([{
        "model_id": "unpriced", "price_in": np.nan, "price_out": np.nan,
    }])
    got, warnings = cost_rows(usage, catalogue)
    assert got.loc[0, "calls"] == 2 and got.loc[0, "output_tokens"] == 80
    assert np.isnan(got.loc[0, "cost"])
    assert warnings.query("code == 'missing_price'").iloc[0].model == "unpriced"


def test_summary_uses_p5_median_p95_and_keeps_random_usable_count():
    trials = pd.DataFrame({
        "trial": list(range(5)), "metric": ["usable_items"] * 5,
        "value": [0, 1, 2, 3, 4],
    })
    got = summarize_trials(trials)
    row = got.iloc[0]
    assert row["p50"] == 2
    assert row["p5"] < row["p50"] < row["p95"]


def test_same_unseen_judge_effect_is_reused_within_trial():
    from mc_cost.simulation import LatentCache
    cache = LatentCache(np.random.default_rng(3))
    first = cache.normal((0, "judge", "unseen"), 0.0, 1.0)
    second = cache.normal((0, "judge", "unseen"), 0.0, 1.0)
    assert first == second
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_simulation.py -v`

Expected: FAIL because `mc_cost.simulation` does not exist.

- [ ] **Step 3: Implement the trial loop with shared latent draws**

Use one RNG seeded from the saved configuration and one `LatentCache` keyed by `(trial, effect_type, identity)`. For each trial:

```python
for trial in range(config.trials):
    posterior_index = trial % posterior_draws
    comparisons = comparison_factory(rng)
    attempted = rng.choice(item_ids, config.attempted_items, replace=False)
    correctness = draw_correctness(comparisons, attempted, irt, observed_cells,
                                   posterior_index, rng, latent_cache)
    usable = correctness.query("left_correct and right_correct")
    presented = counterbalance(usable, config.presentations)
    logits = scenario_left_logits(judgment_artifact, presented, posterior_index, latent_cache)
    presented["chose_left"] = rng.binomial(1, expit(logits))
    presented["chose_m1"] = np.where(
        presented["m1_is_left"], presented["chose_left"], 1 - presented["chose_left"]
    )
    usage = draw_scenario_tokens(presented, candidate_token_artifact,
                                 judge_token_artifact, posterior_index, rng, latent_cache)
```

Observed model-item cells sample one observed attempt for that cell. Unseen cells use the matching fixed-3PL draw. Observed transcript lengths sample the observed model/item distribution; unseen lengths use `draw_transcript_lengths` from the Task 5 candidate artifact. Candidate generation is counted once per unique attempted `(model, item)` when `reuse_responses` is false and zero otherwise. Judge calls are counted only for usable item/order rows.

`cost_rows(...) -> tuple[pd.DataFrame, pd.DataFrame]` left-joins prices by exact model configuration. It always returns calls and token counts. If either required price is missing, it returns `NaN` for that model's currency subtotal and a `missing_price` warning naming the model; totals with missing constituent prices also remain `NaN`. It prices `input_tokens * price_in + output_tokens * price_out` exactly once and never adds `reasoning_tokens` again.

Write tidy trial metrics rather than one wide bespoke table. Required metric names are:

```python
TRIAL_METRICS = (
    "usable_items", "judgment_calls", "candidate_calls", "choice_accuracy",
    "candidate_input_tokens", "candidate_output_tokens",
    "candidate_ordinary_output_tokens", "candidate_reasoning_tokens",
    "judge_input_tokens", "judge_output_tokens",
    "judge_ordinary_output_tokens", "judge_reasoning_tokens", "cost",
)
```

`summarize_trials` groups by configuration, judge, M1, M2, model, role, metric, and warning state where those keys apply, then calculates `p5`, `p50`, `p95`, and `n_trials`. Define `choice_accuracy` as choosing the higher-`theta` candidate, report ties as a separate count rather than scoring them, and retain raw `P(choose M1)` draws for manual rows.

Store the exact pre-filter upper bound separately, without artificial intervals: automatic mode uses `J * M * N * attempted_items * presentations`; manual mode replaces `J * M * N` with the retained manual-row count. Assert every simulated `judgment_calls` draw is no larger than that bound.

- [ ] **Step 4: Run simulation tests**

Run: `uv run pytest tests/test_mc_cost_simulation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/simulation.py tests/test_mc_cost_simulation.py
git commit -m "feat: simulate judgments and cost"
```

---

### Task 8: Add Held-Out Validation, Feature Selection, and Ranking Recovery

**Files:**
- Create: `src/mc_cost/validation.py`
- Test: `tests/test_mc_cost_validation.py`

**Interfaces:**
- Consumes: prepared rows, fit callables, model/token artifacts, and simulated vote rows.
- Produces: `cross_validate_judgments(...)`, `choose_judgment_spec(...)`, `validate_tokens(...)`, `support_distance(...)`, `ranking_recovery(votes, reference, generating_theta=None) -> tuple[pd.DataFrame, pd.DataFrame]`, and `decision.json`.

- [ ] **Step 1: Write failing gate/support/recovery tests**

Create `tests/test_mc_cost_validation.py`:

```python
import numpy as np
import pandas as pd

from mc_cost.validation import choose_judge_gate, ranking_recovery, support_distance


def test_judge_covariates_require_improvement_beyond_one_se():
    conditional = pd.DataFrame({"fold": ["a", "b", "c"], "log_loss": [0.49, 0.52, 0.50]})
    pooled = pd.DataFrame({"fold": ["a", "b", "c"], "log_loss": [0.50, 0.51, 0.52]})
    assert not choose_judge_gate(conditional, pooled)["enabled"]

    conditional["log_loss"] = [0.40, 0.41, 0.42]
    assert choose_judge_gate(conditional, pooled)["enabled"]


def test_joint_support_flags_far_points_using_training_loo_threshold():
    train = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    query = np.array([[0.05, 0.05], [10.0, 10.0]])
    got = support_distance(train, query)
    assert not got.iloc[0].outside_support
    assert got.iloc[1].outside_support


def test_ranking_recovery_returns_component_results_and_warning():
    votes = pd.DataFrame([
        {"fit": "gpqa", "major": "a", "minor": "b", "item_id": "i1", "picked_major": 1},
        {"fit": "gpqa", "major": "c", "minor": "d", "item_id": "i1", "picked_major": 1},
    ])
    reference = pd.DataFrame({"model": ["a", "b", "c", "d"], "score": [4, 3, 2, 1]})
    results, warnings = ranking_recovery(votes, reference)
    assert results["component"].nunique() == 2
    assert "disconnected_graph" in set(warnings["code"])
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_validation.py -v`

Expected: FAIL because `mc_cost.validation` does not exist.

- [ ] **Step 3: Implement fixed candidate specifications and validation**

Evaluate only these nested judgment specifications:

```python
JUDGMENT_SPECS = {
    "baseline": {"global_features": (), "include_length": False},
    "core": {
        "global_features": (
            "oversight_sd", "difficulty_sd", "theta_gap_x_oversight_sd",
            "theta_gap_x_difficulty_sd", "same_family_x_theta_gap_sd",
            "self_comparison", "reasoning_difference_sd",
        ),
        "include_length": True,
    },
    "core_release": {
        "global_features": (
            "oversight_sd", "difficulty_sd", "theta_gap_x_oversight_sd",
            "theta_gap_x_difficulty_sd", "same_family_x_theta_gap_sd",
            "self_comparison", "reasoning_difference_sd", "release_gap_days_sd",
            "relative_release_days_sd",
        ),
        "include_length": True,
    },
}
```

The judge-specific capability-gap slope is present in every specification, so `baseline` is the requested capability-gap-only hierarchy (plus intercept/item/candidate residuals). `encode_judgments` subsets `global_x` in the declared order and omits the length parameters entirely when `include_length` is false.

Run leave-one-judge-out folds for conditional versus pooled judge populations and leave-one-candidate-out folds for candidate residual prediction. Save per-row probabilities and per-fold log loss/Brier score. Pick the simplest feature specification whose mean log loss is within one standard error of the best. Enable conditional judge covariates only when `pooled_loss - conditional_loss > standard_error(paired_fold_difference)`.

For token models, leave one model configuration out by role/reasoning class and report log-scale MAE plus empirical coverage of the 90% predictive interval. Validate candidate transcript-length draws the same way and stratify their calibration by release-date band, item difficulty, correctness, and visible reasoning. For support, standardize with training means/scales, compute nearest-neighbour distances, and use the 95th percentile of training leave-one-out distances as the warning threshold.

For ranking recovery, reshape simulated votes to the existing columns (`fit`, `major`, `minor`, `item_id`, `picked_major`), then reuse:

```python
ROOT = Path(__file__).resolve().parents[2]
ELO_DIR = ROOT / "scripts" / "analyses" / "judge-elo"
IRT_PROXY_DIR = ROOT / "scripts" / "irt_proxy"
sys.path.insert(0, str(ELO_DIR))
sys.path.insert(0, str(IRT_PROXY_DIR))
from elo import collect_pair_wins, components, fit_bt
from metrics import lomo_predictions
```

Call `scripts/irt_proxy/metrics.py:lomo_predictions` on the resulting Elo/easy/hard frame; do not copy Bradley–Terry or LOMO implementations. Label forward ranking against injected `theta` as `generating_theta_consistency` and observed downsample recovery as `observed_full_data_recovery`. Call `components` first, fit and score each connected component separately, and never compute one Kendall tau across components. Return component-level partial rows plus a `disconnected_graph` warning naming component membership whenever more than one component exists.

Write `decision.json` with selected feature specification, judge-covariate gate, token coverage by role/class, predictor condition number/correlations, support threshold, hashes of prepared data, and validation timestamp. `simulate` must refuse to use an artifact whose data hash differs or whose decision file is absent.

Add posterior-predictive replay tables for observed aggregate choice rate, per-judge differences, position effects, and slices by capability gap, oversight, item difficulty, release-date difference, and transcript-length difference. Within the length-difference check, also stratify judgment-probability calibration by item difficulty, correctness, and visible reasoning. Downsample the observed experiment across the configured `J`, `M`, `N`, and attempted-item grids, compare simulated interval width/ranking recovery with empirical resampling, and store whether the reference is `observed_full_data` or `fitted_posterior` on every row. Keep failed slices and their reasons instead of reducing them into one overall score.

- [ ] **Step 4: Run validation tests**

Run: `uv run pytest tests/test_mc_cost_validation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/validation.py tests/test_mc_cost_validation.py
git commit -m "feat: validate simulator extrapolation"
```

---

### Task 9: Add the Reproducible Command-Line Workflow

**Files:**
- Create: `src/mc_cost/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_mc_cost_cli.py`

**Interfaces:**
- Consumes: Tasks 1–8 public functions and `configs/mc_cost/default.json`.
- Produces: the `mc-cost` command with six subcommands and atomic artifacts under `analysis/mc_cost`.

- [ ] **Step 1: Write failing CLI contract tests**

Create `tests/test_mc_cost_cli.py`:

```python
import json
import pytest

from mc_cost.cli import main


def test_cli_exposes_six_commands(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("catalogue", "prepare", "fit", "validate", "simulate", "export-dashboard"):
        assert command in help_text


def test_manual_simulation_requires_exact_columns(tmp_path, capsys):
    manual = tmp_path / "manual.csv"
    manual.write_text("judge,m1\nj,a\n")
    code = main(["simulate", "--manual", str(manual), "--config", "configs/mc_cost/default.json"])
    assert code == 2
    assert "judge,m1,m2" in capsys.readouterr().err


def test_saved_config_records_seed_and_source_hashes(tmp_path):
    from mc_cost.cli import write_run_config
    path = tmp_path / "config.json"
    write_run_config(path, {"seed": 7}, {"catalogue": "abc", "model": "def"})
    assert json.loads(path.read_text()) == {
        "config": {"seed": 7}, "source_hashes": {"catalogue": "abc", "model": "def"}
    }
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_cli.py -v`

Expected: FAIL because `mc_cost.cli` does not exist.

- [ ] **Step 3: Implement `argparse` subcommands and atomic writes**

Add to `pyproject.toml`:

```toml
mc-cost = "mc_cost.cli:main"
```

The exact commands are:

```bash
mc-cost catalogue --config configs/mc_cost/default.json
mc-cost prepare --config configs/mc_cost/default.json
mc-cost fit --config configs/mc_cost/default.json
mc-cost validate --config configs/mc_cost/default.json
mc-cost simulate --config configs/mc_cost/default.json --judges 4 --anchors 8 --comparators 4 --items 100 --presentations 2 --reuse-responses --accuracy-target higher_theta --seed 0
mc-cost simulate --config configs/mc_cost/default.json --manual comparisons.csv --items 100 --presentations 2 --seed 0
mc-cost export-dashboard --config configs/mc_cost/default.json --out interactive_viewer/data/mc_cost.json
```

`catalogue` writes `catalogue.csv` and aligned `capability_draws.npz`; `prepare` writes the three Parquet tables plus `scales.json`; `fit` writes baseline/core/core-release judgment artifacts plus the candidate artifact containing transcript-length/token posteriors and the judge token artifact; `validate` writes held-out tables and `decision.json`; `simulate` writes a timestamp-free deterministic run ID derived from config/source hashes; `export-dashboard` runs the declared grid and manual triple bank before atomically replacing JSON.

Every subcommand supports `--seed`. `fit` additionally supports `--warmup`, `--samples`, and `--chains`; `simulate`/`export-dashboard` support `--trials`. `simulate` exposes mutually exclusive `--reuse-responses`/`--generate-responses` flags and `--accuracy-target` (initially only `higher_theta`); resolved values are always saved in the run config. Validate a supplied manual CSV's header before loading upstream artifacts so schema errors remain actionable. Missing upstream artifacts produce exit code 2 and name the command that creates them.

- [ ] **Step 4: Run CLI tests and installation smoke check**

Run:

```bash
uv run pytest tests/test_mc_cost_cli.py -q
uv run mc-cost --help
```

Expected: tests PASS and help lists all six commands.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/cli.py pyproject.toml tests/test_mc_cost_cli.py
git commit -m "feat: add simulator command-line workflow"
```

---

### Task 10: Export and Display the Static Interactive Dashboard

**Files:**
- Create: `interactive_viewer/mc_cost.html`
- Create: `interactive_viewer/mc_cost.js`
- Create: `interactive_viewer/mc_cost.css`
- Modify: `interactive_viewer/index.html`
- Test: `tests/test_mc_cost_dashboard.py`

**Interfaces:**
- Consumes: `interactive_viewer/data/mc_cost.json` generated by `mc-cost export-dashboard`.
- Produces: automatic-grid controls, unrestricted manual triple selection, capability-gap/scaling plots, warnings, and per-model token/cost tables.

- [ ] **Step 1: Write failing dashboard schema/markup tests**

Create `tests/test_mc_cost_dashboard.py`:

```python
import json
from pathlib import Path

from mc_cost.simulation import validate_dashboard_payload


def test_dashboard_payload_requires_quantiles_and_both_modes():
    payload = {
        "schema_version": 1, "quantiles": [0.05, 0.5, 0.95],
        "catalogue": [], "automatic": [], "manual": [], "scaling": [],
        "ranking_recovery": [], "warnings": [],
    }
    validate_dashboard_payload(payload)


def test_dashboard_markup_has_accessible_controls():
    html = Path("interactive_viewer/mc_cost.html").read_text()
    for control in ("mode", "judge-count", "anchor-count", "comparator-count",
                    "attempted-items", "manual-rows", "run-summary", "token-table",
                    "ranking-recovery"):
        assert f'id="{control}"' in html
    assert "<label" in html
    assert "plotly.min.js" in html
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_mc_cost_dashboard.py -v`

Expected: FAIL because the dashboard files and payload validator do not exist.

- [ ] **Step 3: Implement the export schema and static page**

Add `validate_dashboard_payload` and `dashboard_payload` to `src/mc_cost/simulation.py`. Emit exactly:

```json
{
  "schema_version": 1,
  "generated_at": "ISO-8601 UTC",
  "source_hashes": {},
  "quantiles": [0.05, 0.5, 0.95],
  "catalogue": [],
  "automatic": [],
  "manual": [],
  "scaling": [],
  "ranking_recovery": [],
  "warnings": []
}
```

The Python exporter precomputes exact simulation summaries for every feasible point in the configured Cartesian product of `J`, `M`, `N`, attempted items, presentations, strength filter, and exported pool definitions. For manual mode, `dashboard.manual_triples` is a list of objects with required `judge,m1,m2` and optional `judge_reasoning_class,m1_reasoning_class,m2_reasoning_class`; always export those exact six-field scenario identities, then use the saved seed to sample default-catalogue-class triples without replacement up to `manual_triple_limit`. Record total/exported counts and omitted identities in a `manual_bank_truncated` warning. The UI filters exact joint scenarios from this grid; scaling curves vary one selected axis while holding the other current controls fixed. Infeasible grid cells remain in `warnings`; do not silently clamp counts. If a selected triple/class combination is unavailable, show the exact `manual_triples` object to add and the `mc-cost export-dashboard` command to rerun; JavaScript must not relabel a default-class prediction as an override.

Set `generated_at` from the saved validation decision timestamp, not the export wall clock. Sort object keys and all exported row arrays before serialization. Repeating `export-dashboard` against unchanged artifacts must therefore produce byte-identical JSON.

Build `mc_cost.html` with native number/range/select inputs and an editable manual-row table. `mc_cost.js` only filters/joins exported rows and calls the existing local Plotly bundle. It must not contain logistic, Bernoulli, IRT, token, or cost formulas. Show:

- automatic controls for named judge/anchor/comparator pools, `J`, `M`, `N`, attempted items, presentations, and strength filter;
- manual add/remove rows with judge/M1/M2 catalogue selects and reasoning/non-reasoning overrides when a selected configuration needs one;
- capability-gap versus accuracy bands;
- one-axis scaling bands;
- component-safe ranking-recovery bands and partial-result reasons;
- design/call/cost summary;
- per-model candidate/judge input, total output, ordinary output, reasoning, and cost p5/p50/p95;
- per-manual-row `P(choose M1)` p5/p50/p95 and expected higher-`theta` choice accuracy;
- observed/predicted and extrapolation badges;
- a clear message when a custom pool needs `mc-cost export-dashboard` regeneration.

Add an ordinary link in `interactive_viewer/index.html` to `mc_cost.html`; do not modify the existing viewer state or plots.

- [ ] **Step 4: Run tests and browser smoke check**

Run:

```bash
uv run pytest tests/test_mc_cost_dashboard.py -q
uv run python -m http.server 8000 -d interactive_viewer
```

Open `http://localhost:8000/mc_cost.html`. Verify automatic/manual mode switching, keyboard labels, plot updates, warning visibility, token-table intervals, light/dark themes, and a narrow viewport. Stop the server after the check.

Expected: pytest PASS; the page loads local Plotly and produces no browser console errors.

- [ ] **Step 5: Commit**

```bash
git add src/mc_cost/simulation.py interactive_viewer/mc_cost.html interactive_viewer/mc_cost.js interactive_viewer/mc_cost.css interactive_viewer/index.html tests/test_mc_cost_dashboard.py
git commit -m "feat: add Monte Carlo simulator dashboard"
```

---

### Task 11: Run an End-to-End Smoke Pipeline and Finish the Runbook

**Files:**
- Modify: `docs/plans/2026-08-14-MC-cost-simulation.md`
- Test: all files from Tasks 1–10

**Interfaces:**
- Consumes: complete CLI and one tiny synthetic/smoke configuration.
- Produces: verified pipeline artifacts, updated run instructions, and no uncommitted generated outputs.

- [ ] **Step 1: Add a small integration test before the full-data run**

Append to `tests/test_mc_cost_cli.py`:

```python
def test_tiny_pipeline_is_seed_reproducible(tmp_path, tiny_mc_config):
    from mc_cost.cli import main

    config = tiny_mc_config(tmp_path, seed=9, warmup=10, samples=10, trials=20)
    for command in ("catalogue", "prepare", "fit", "validate", "simulate", "export-dashboard"):
        assert main([command, "--config", str(config)]) == 0
    first = (tmp_path / "viewer" / "data" / "mc_cost.json").read_bytes()
    assert main(["export-dashboard", "--config", str(config)]) == 0
    assert (tmp_path / "viewer" / "data" / "mc_cost.json").read_bytes() == first
```

The `tiny_mc_config` fixture creates two judges, three candidates, four items, observed token rows for one reasoning and one non-reasoning configuration, and a two-draw fixed-3PL artifact. It uses no repository data and performs no network access.

- [ ] **Step 2: Run the integration test and fix only pipeline-boundary failures**

Run: `uv run pytest tests/test_mc_cost_cli.py::test_tiny_pipeline_is_seed_reproducible -v`

Expected: PASS twice with byte-identical dashboard JSON.

- [ ] **Step 3: Update the original draft into a runbook**

Replace the unfinished ending of `docs/plans/2026-08-14-MC-cost-simulation.md` with:

````markdown
## Status

Implemented from the reviewed design in
`docs/superpowers/specs/2026-08-14-monte-carlo-cost-simulator-design.md`.

## Run order

```bash
uv run python scripts/irt_proxy/run.py
uv run mc-cost catalogue --config configs/mc_cost/default.json
uv run mc-cost prepare --config configs/mc_cost/default.json
uv run mc-cost fit --config configs/mc_cost/default.json
uv run mc-cost validate --config configs/mc_cost/default.json
uv run mc-cost simulate --config configs/mc_cost/default.json --judges 4 --anchors 8 --comparators 4 --items 100
uv run mc-cost export-dashboard --config configs/mc_cost/default.json
uv run python -m http.server 8000 -d interactive_viewer
```

The dashboard is at `http://localhost:8000/mc_cost.html`. Edit durable model
metadata in `analysis/mc_cost/catalogue.csv`, rerun `validate`, then rerun
`export-dashboard`; the browser does not fit or approximate the model.
````

- [ ] **Step 4: Run fresh verification**

Run:

```bash
uv run pytest tests/test_mc_cost_catalogue.py tests/test_mc_cost_data.py tests/test_mc_cost_model.py tests/test_mc_cost_tokens.py tests/test_mc_cost_design.py tests/test_mc_cost_simulation.py tests/test_mc_cost_validation.py tests/test_mc_cost_cli.py tests/test_mc_cost_dashboard.py -q
uv run pytest scripts/irt_proxy/test_item_banks.py scripts/irt_proxy/test_run.py -q
uv run pytest -q
git diff --check
git status --short
```

Expected: all test commands PASS; `git diff --check` prints nothing; generated `analysis/mc_cost` and `interactive_viewer/data/mc_cost.json` remain ignored/untracked according to repository policy; pre-existing user changes remain untouched unless a task explicitly integrated with them.

- [ ] **Step 5: Commit the runbook and final boundary fixes**

```bash
git add tests/test_mc_cost_cli.py
git add -f docs/plans/2026-08-14-MC-cost-simulation.md
git commit -m "docs: finish Monte Carlo simulator runbook"
```

---

## Review Checkpoints

After Tasks 1, 4, 5, 8, and 10, stop for a fresh correctness review before continuing. These gates cover the IRT contract, judgment model, token accounting, extrapolation claims, and dashboard/statistical-engine boundary respectively.

The implementation is complete only when the held-out decision artifact exists, the tiny pipeline is seed-reproducible, reasoning tokens are never double-billed, manual duplicates survive end to end, infeasible `J`/`M`/`N` configurations fail explicitly, and the static dashboard displays the exact Python-generated percentiles.
