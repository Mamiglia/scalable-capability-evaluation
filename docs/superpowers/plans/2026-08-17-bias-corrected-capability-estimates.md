# Bias-Corrected Capability Estimates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce cross-fitted, bias-corrected model capability estimates from easy-item pairwise judgments and validate them against hard-item benchmark performance.

**Architecture:** Extend the existing `scripts/irt_proxy` analysis rather than create a second pipeline. Normalize all compatible GPQA judgment sources into one left-choice frame, freeze item parameters from a candidate-disjoint reference IRT fit, fit judge-specific nuisance effects on exact-model training folds, and infer held-out candidate abilities with those nuisance posteriors cut off from target feedback. Local work is limited to code and tiny tests; all production preparation, IRT, and GLM fitting runs as restartable SWC Slurm jobs.

**Tech Stack:** Python 3.13+, Pandas, NumPy, SciPy, NumPyro/JAX, Inspect eval logs, pytest, Matplotlib, existing IRT/Bradley–Terry/style-feature helpers, Slurm on SWC.

**Spec:** `docs/2026-08-17-correcting-biases.md`

## Global Constraints

- Do not run production preparation, IRT, GLM fitting, or posterior target inference on the laptop. Local execution is limited to unit tests with synthetic fixtures and lightweight file inspection.
- Run production fits on SWC compute nodes, never on the gateway. The gateway is limited to `ls`, `tail`, `squeue`, `sacct`, `sbatch`, and file transfer.
- Never delete, replace, or modify existing Ceph data. Stage into a new run directory with `rsync --ignore-existing`; write outputs through temporary names and atomic `mv`; list failed artifacts for the user instead of deleting them.
- Use the existing Linux wheelhouse under `/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/mc_cost_preview_20260815/wheelhouse`; no downloads or installs on the gateway and no new dependency.
- Pool compatible natural GPQA judgments across pilot-rd2, data expansion, fine-tuning evaluation rounds, and HF. Preserve source, prompt condition, judge endpoint, and base/fine-tuned identity.
- Deliberate manipulation/style-attack rows stay outside the primary fit and are used only for coefficient validation.
- One run config names exactly one benchmark. The first production run is GPQA. MBPP uses the same commands in a separate versioned run only after its candidate-disjoint reference-response preflight passes; never combine GPQA and MBPP latent ability scales.
- Fit item parameters on a reference response pool excluding every evaluated candidate model, then freeze them before model folds are assigned.
- Cross-fit five deterministic folds over exact candidate configuration IDs. Do not group related models into families.
- A held-out candidate is excluded globally from calibration in every source. Its hard responses and hard-derived capability remain hidden until scoring.
- Judges may cover different target pools. Report disconnected components and require each reported target component to contain at least two fixed calibration anchors.
- Do not filter judges using ECI, hard accuracy, or held-out performance. A calibrated negative or near-zero capability slope remains a model result.
- Reuse `scripts/hibayes/style_features.py`, `scripts/irt_proxy/item_banks.py`, `scripts/irt_proxy/metrics.py`, and the existing Bradley–Terry helpers. Do not add embeddings, model-name predictors, another package, or another generic framework.
- Production GLM and target fits use four parallel chains, 1,000 warmup and 1,000 posterior samples per chain, seed 0. Unit tests use pure functions, fake posterior arrays, or at most 20/20 tiny MCMC draws.
- Figures follow the `nature-figures` skill: 8 pt base text, 7 pt ticks/legend, 1 pt result lines, capless 95% intervals, outside legends, outward 5 pt spines, editable SVG text, PNG at 300 dpi, and final rendered width at or below 3.0 inches.
- Preserve all unrelated user changes currently present in the worktree.

---

## File Map

### Create

- `scripts/irt_proxy/bias_model.py` — hierarchical calibration model and cut-posterior target inference.
- `scripts/irt_proxy/bias_crossfit.py` — exact-model folds, leakage-safe frame construction, anchor checks, baselines, and fold reduction.
- `scripts/irt_proxy/correct_biases.py` — `prepare`, `fit-reference`, `fit-fold`, and `reduce` CLI.
- `scripts/irt_proxy/correction_figures.py` — compact capability and corrected-gap figures.
- `scripts/irt_proxy/bias_reference_fit.sbatch` — SWC preparation/reference IRT job.
- `scripts/irt_proxy/bias_fold_fit.sbatch` — SWC twenty-task array for two IRT forms × five folds × corrected/uncorrected models.
- `scripts/irt_proxy/test_bias_model.py`
- `scripts/irt_proxy/test_bias_crossfit.py`
- `scripts/irt_proxy/test_correction_figures.py`

### Modify

- `scripts/irt_proxy/config.yaml` — pooled-source paths and bias-correction fit settings.
- `scripts/irt_proxy/inputs.py` — HF/fine-tuning loaders, exact-candidate mode, and transcript-feature joins.
- `scripts/irt_proxy/item_banks.py` — candidate-disjoint reference fit and fixed-item candidate scoring.
- `scripts/irt_proxy/metrics.py` — cross-fitted ranking/value metrics and model-pair gap rows.
- `scripts/irt_proxy/test_inputs.py`
- `scripts/irt_proxy/test_item_banks.py`
- `scripts/irt_proxy/test_prediction.py`
- `docs/2026-08-17-correcting-biases.md` — record implemented commands and any approved design changes.

### Generated, not committed

- `analysis/bias_correction/input/{votes,responses,features,folds}.parquet`
- `analysis/bias_correction/reference/{items.csv,anchors.csv,posterior.npz}`
- `analysis/bias_correction/folds/{fixed_3pl,2pl}/fold-<0..4>/{corrected,uncorrected}.npz`
- `analysis/bias_correction/tables/{predictions,coefficients,coverage,metrics,gap_pairs}.csv`
- `analysis/bias_correction/figures/{capability_predictions,gap_curve}.{png,svg}`

---

### Task 1: Normalize Every Compatible Judgment Source

**Files:**
- Modify: `scripts/irt_proxy/config.yaml`
- Modify: `scripts/irt_proxy/inputs.py`
- Test: `scripts/irt_proxy/test_inputs.py`

**Interfaces:**
- Consumes: viewer `pairs/*__*__full.json`, HF raw judge `.eval` files plus their `logs.json` manifests, `judge_results/judge_ft/pairs.parquet`, base/FT `verdicts_*.jsonl`, and canonical response tables.
- Produces: `load_correction_data(root: Path, config: Mapping) -> tuple[pd.DataFrame, pd.DataFrame]`; votes have one row per presentation order and responses have exact source/model/item/epoch transcript rows.

- [ ] **Step 1: Write failing source-loader tests**

Add tests to `scripts/irt_proxy/test_inputs.py`:

```python
def test_correction_votes_keep_deepseek_and_effort_candidates(tmp_path):
    from inputs import load_votes

    pairs = tmp_path / "gpqa" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "deepseek-v4-flash-0731__gpt-5-high__full.json").write_text(
        json.dumps({"i": {"j": _verdict()}})
    )

    got = load_votes(tmp_path, {"gpqa": "gpqa"}, all_candidates=True)

    assert got[["major", "minor"]].iloc[0].tolist() == [
        "deepseek-v4-flash-0731", "gpt-5-high",
    ]


def test_finetuning_votes_keep_base_and_ft_as_distinct_judges(tmp_path):
    from inputs import load_finetuning_votes

    pairs = pd.DataFrame([{
        "pair_uid": "i|a|b", "order": 0, "item": "i",
        "model_1": "a", "model_2": "b", "text_1": "short",
        "text_2": "a longer answer", "split": "eval_mm",
    }])
    pair_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pair_path)
    for judge, logits in {"llama_base": (-1.0, -2.0), "llama_ft": (-3.0, -1.0)}.items():
        pd.DataFrame([{
            "pair_uid": "i|a|b", "order": 0,
            "logit_1": logits[0], "logit_2": logits[1], "split": "eval_mm",
        }]).to_json(tmp_path / f"{judge}.jsonl", orient="records", lines=True)

    got = load_finetuning_votes(pair_path, {
        "llama_base": tmp_path / "llama_base.jsonl",
        "llama_ft": tmp_path / "llama_ft.jsonl",
    })

    assert got.set_index("judge")["chosen_left"].to_dict() == {
        "llama_base": 1.0, "llama_ft": 0.0,
    }
```

Add:

```python
def test_hf_vote_loader_uses_raw_item_samples(tmp_path):
    from types import SimpleNamespace as NS
    from inputs import load_hf_votes

    pair = tmp_path / "raw" / "pilot-j-gpqa-a-vs-b"
    pair.mkdir(parents=True)
    (pair / "logs.json").write_text(json.dumps({"run.eval": {"status": "success"}}))
    eval_path = tmp_path / "evals" / "run.eval"
    eval_path.parent.mkdir()
    eval_path.touch()
    log = NS(
        status="success",
        eval=NS(metadata={
            "major_path": "data/gpqa/hash/a/0/a.eval",
            "minor_path": "data/gpqa/hash/b/0/b.eval",
        }),
        samples=[NS(id="item_1", scores={"judge": NS(answer="1")})],
    )

    got = load_hf_votes(tmp_path / "raw", tmp_path / "evals", reader=lambda _: log)

    assert got.loc[0, ["source", "item_id", "left_model", "right_model", "chosen_left"]].tolist() == [
        "hf-judging-2026-08-17", "item", "a", "b", 1.0,
    ]
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
.venv/bin/pytest scripts/irt_proxy/test_inputs.py -q
```

Expected: FAIL because `all_candidates`, `load_finetuning_votes`, and `load_hf_votes` do not exist.

- [ ] **Step 3: Add the minimal loaders**

Change `load_votes` to preserve its current default and bypass candidate filtering only for this analysis:

Inside the existing pair loop, replace the candidate predicate with:

```python
excluded = major == minor or (
    not all_candidates and (
        is_deepseek(major) or is_deepseek(minor)
        or not is_standard_candidate(major)
        or not is_standard_candidate(minor)
    )
)
if excluded:
    continue
```

Add:

```python
def load_finetuning_votes(pair_path: Path, verdict_paths: Mapping[str, Path]) -> pd.DataFrame:
    pairs = pd.read_parquet(pair_path)
    keys = ["pair_uid", "order", "split"]
    rows = []
    for judge, path in verdict_paths.items():
        verdicts = pd.read_json(path, lines=True)
        merged = verdicts.merge(pairs, on=keys, validate="one_to_one")
        rows.append(pd.DataFrame({
            "source": "judge-ft", "benchmark": "gpqa", "condition": "natural",
            "judge": judge, "item_id": merged["item"].astype(str),
            "left_model": merged["model_1"].map(canonical_model),
            "right_model": merged["model_2"].map(canonical_model),
            "left_text": merged["text_1"], "right_text": merged["text_2"],
            "chosen_left": (merged["logit_1"] > merged["logit_2"]).astype(float),
        }))
    return pd.concat(rows, ignore_index=True)
```

Implement `load_hf_votes(manifest_root, eval_root, reader=read_eval_log)` by indexing `.eval` basenames under `eval_root`, reading only successful logs named by each manifest, parsing `sample.id` as `<item>_<left-slot-of-major>`, extracting the scorer answer, and orienting major/minor into left/right. Retain the exact judge model string and serialized provider-routing fields from `log.eval.model_args`. Reject missing/duplicate raw evals; aggregate headers never create verdict rows.

Add `read_response_tree(...)` for `model/<epoch>/*.eval` trees and `resolve_hf_responses(...)` with this order: exact recorded path, downloaded HF response tree, canonical Epoch row. Raise one error listing unresolved `(benchmark, model)` identities.

Add aliases only for names present in HF manifests (`glm4.6`, `sonnet4.5`, `gpt5`, `opus4.7`, `gpt4.1`, `granite`, `ling`). Do not introduce fuzzy matching.

- [ ] **Step 4: Attach existing style features without another extractor**

Add:

```python
BIAS_FEATURES = (
    "log_chars", "personal_pronouns", "hedging", "backtracking",
    "uncertainty", "confidence", "questions",
)

def pair_feature_rows(votes: pd.DataFrame, lexicon: Mapping) -> pd.DataFrame:
    def measured(text):
        return {"log_chars": np.log1p(len(text)), **feature_rates(text, lexicon)}

    left = pd.DataFrame([measured(text) for text in votes["left_text"]])
    right = pd.DataFrame([measured(text) for text in votes["right_text"]])
    result = votes.reset_index(drop=True).copy()
    for name in BIAS_FEATURES:
        result[f"delta_{name}"] = left[name] - right[name]
    return result
```

For viewer sources, join exact transcripts from the corresponding response logs before calling this function and retain the existing length-equality guard from `add_style_differences`. Do not read `analysis/hibayes/style_rates.csv` when exact text is available.

- [ ] **Step 5: Run focused and existing input tests**

Run:

```bash
.venv/bin/pytest scripts/irt_proxy/test_inputs.py scripts/hibayes/test_style_features.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/irt_proxy/config.yaml scripts/irt_proxy/inputs.py scripts/irt_proxy/test_inputs.py
git commit -m "feat: load pooled bias-correction judgments"
```

---

### Task 2: Freeze a Candidate-Disjoint Reference Item Bank

**Files:**
- Modify: `scripts/irt_proxy/item_banks.py`
- Test: `scripts/irt_proxy/test_item_banks.py`

**Interfaces:**
- Consumes: canonical response rows plus the union of evaluated candidate IDs.
- Produces: `fit_reference_bank(..., irt_model) -> tuple[pd.DataFrame, dict]` and `score_fixed_item_capabilities(...) -> pd.DataFrame` with anchor `theta`, `theta_p5`, and `theta_p95`.

- [ ] **Step 1: Write failing exclusion and fixed-item scoring tests**

Add:

```python
def test_reference_bank_excludes_every_candidate(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": model, "benchmark": "gpqa", "item": item, "epoch": 1,
         "answer": "A", "correct": model == "reference", "source_priority": 1}
        for model in ("reference", "candidate") for item in ("i1", "i2")
    ])
    seen = {}

    def fake_fit(data, **_kwargs):
        seen["models"] = list(data.models)
        return {"theta": np.zeros((2, 1)), "a": np.ones((2, 2)),
                "b": np.zeros((2, 2))}

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    item_banks.fit_reference_bank(
        responses, {"candidate"}, floor=0.25, irt_model="fixed_3pl",
        warmup=2, samples=2, seed=0,
    )

    assert seen["models"] == ["reference"]


def test_fixed_items_score_more_correct_model_higher():
    from item_banks import score_fixed_item_capabilities

    responses = pd.DataFrame([
        {"model": model, "benchmark": "gpqa", "item": item,
         "answer": "A", "correct": correct}
        for model, values in (("strong", (1, 1, 1)), ("weak", (1, 0, 0)))
        for item, correct in zip(("i1", "i2", "i3"), values)
    ])
    posterior = {
        "a": np.ones((4, 3)), "b": np.zeros((4, 3)),
        "floor": np.full(3, 0.25), "items": [("gpqa", f"i{i}") for i in (1, 2, 3)],
    }

    got = score_fixed_item_capabilities(responses, posterior, max_draws=4)

    assert got.set_index("model").loc["strong", "theta"] > got.set_index("model").loc["weak", "theta"]
```

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_item_banks.py -q
```

Expected: FAIL because the two functions are undefined.

- [ ] **Step 3: Implement the reference fit and one-dimensional anchor scorer**

Use existing `irt_cells`, `encode`, `fit`, and `assign_splits`. `fit_reference_bank` filters the requested benchmark and `~model.isin(candidate_models)` before calling either fixed-floor 3PL or ordinary 2PL. The production reference command runs both forms from the same candidate-disjoint rows and saves separate item splits/posteriors.

For each candidate and each retained item-posterior draw, find the MAP ability with SciPy's installed scalar optimizer:

```python
def _theta_map(correct, a, b, floor):
    def loss(theta):
        p = floor + (1 - floor) / (1 + np.exp(-a * (theta - b)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return 0.5 * theta**2 - np.sum(correct * np.log(p) + (1 - correct) * np.log1p(-p))
    return minimize_scalar(loss, bounds=(-6, 6), method="bounded").x
```

Summarize those draw-wise MAP values with mean/5th/95th percentiles. Reject candidates with fewer than two scored reference-bank items. This is anchor scoring only; held-out target rows are filtered later and never passed here during their fold.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest scripts/irt_proxy/test_item_banks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/irt_proxy/item_banks.py scripts/irt_proxy/test_item_banks.py
git commit -m "feat: freeze candidate-disjoint IRT anchors"
```

---

### Task 3: Build Exact-Model Folds and Leakage-Safe Frames

**Files:**
- Create: `scripts/irt_proxy/bias_crossfit.py`
- Create: `scripts/irt_proxy/test_bias_crossfit.py`

**Interfaces:**
- Produces: `assign_exact_model_folds(models, n_splits=5, seed=0)`, `build_fold_frames(...)`, and `anchor_coverage(...)`.
- Later tasks consume calibration rows with fixed `delta_theta` and target rows containing easy judgments only.

- [ ] **Step 1: Write the failing fold-boundary tests**

```python
def test_exact_model_folds_do_not_group_families():
    from bias_crossfit import assign_exact_model_folds

    got = assign_exact_model_folds(
        ["qwen-low", "qwen-high", "other-a", "other-b", "other-c"],
        n_splits=5, seed=0,
    )

    assert got["model"].nunique() == 5
    assert got["fold"].nunique() == 5


def test_fold_frames_exclude_targets_globally_and_use_easy_target_rows_only():
    from bias_crossfit import build_fold_frames

    votes = pd.DataFrame([
        {"source": source, "item_id": item, "left_model": left,
         "right_model": right, "judge": "j", "chosen_left": 1.0}
        for source in ("pilot", "hf")
        for item in ("easy", "hard")
        for left, right in (("target", "anchor-a"), ("anchor-a", "anchor-b"))
    ])
    folds = pd.DataFrame({"model": ["target", "anchor-a", "anchor-b"], "fold": [0, 1, 2]})
    splits = pd.DataFrame({"benchmark": ["gpqa", "gpqa"], "item": ["easy", "hard"],
                           "half": ["easy", "hard"]})
    anchors = pd.DataFrame({"model": ["anchor-a", "anchor-b"], "theta": [0.0, 1.0]})

    calibration, target = build_fold_frames(votes, anchors, splits, folds, fold=0)

    assert "target" not in set(calibration["left_model"]) | set(calibration["right_model"])
    assert set(target["item_id"]) == {"easy"}
    assert set(target["source"]) == {"pilot", "hf"}
```

Add:

```python
def test_anchor_coverage_pools_different_judge_target_edges():
    from bias_crossfit import anchor_coverage

    rows = pd.DataFrame([
        {"judge": "j1", "left_model": "target", "right_model": "anchor-a"},
        {"judge": "j2", "left_model": "target", "right_model": "anchor-b"},
    ])

    eligible = anchor_coverage(rows, {"anchor-a", "anchor-b"}).iloc[0]
    insufficient = anchor_coverage(rows.iloc[:1], {"anchor-a", "anchor-b"}).iloc[0]

    assert bool(eligible["eligible"])
    assert eligible["n_anchors"] == 2
    assert not bool(insufficient["eligible"])
    assert insufficient["reason"] == "fewer than 2 anchors"
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_crossfit.py -q
```

Expected: import failure because `bias_crossfit.py` does not exist.

- [ ] **Step 3: Implement deterministic folds and frames**

```python
def assign_exact_model_folds(models, n_splits=5, seed=0):
    models = np.array(sorted(set(models)), dtype=object)
    shuffled = np.random.default_rng(seed).permutation(models)
    return pd.DataFrame({
        "model": shuffled,
        "fold": np.arange(len(shuffled)) % n_splits,
    }).sort_values("model").reset_index(drop=True)
```

`build_fold_frames` must:

1. derive the target set from `folds`;
2. keep calibration rows only when both candidates are outside the target set;
3. join both calibration candidates to anchor `theta` and calculate `delta_theta`;
4. keep target rows only on frozen easy items and only when at least one candidate is a target; and
5. never join target hard accuracy.

Use the existing `components` helper for pooled target/anchor graph coverage. Component IDs are sorted, source-qualified, and carried to every output.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_crossfit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/irt_proxy/bias_crossfit.py scripts/irt_proxy/test_bias_crossfit.py
git commit -m "feat: add exact-model cross-fitting boundaries"
```

---

### Task 4: Fit the Hierarchical Judge Calibration Model

**Files:**
- Create: `scripts/irt_proxy/bias_model.py`
- Create: `scripts/irt_proxy/test_bias_model.py`

**Interfaces:**
- Produces: `fit_calibration(frame, feature_columns, warmup, samples, seed) -> dict`.
- The returned dict contains posterior arrays plus ordered `judges`, `sources`, `conditions`, and `features` labels; target inference must use those exact orders.

- [ ] **Step 1: Write pure predictor and model-shape tests**

```python
def test_linear_predictor_reverses_capability_and_features_not_position():
    from bias_model import linear_predictor

    forward = linear_predictor(
        alpha=np.array([0.3]), source=np.array([[0.1]]), condition=np.array([[0.0]]),
        discrimination=np.array([2.0]), delta_theta=np.array([0.5]),
        beta=np.array([[1.5]]), design=np.array([[0.2]]), judge=np.array([0]),
        source_index=np.array([0]), condition_index=np.array([0]),
    )[0]
    reverse = linear_predictor(
        alpha=np.array([0.3]), source=np.array([[0.1]]), condition=np.array([[0.0]]),
        discrimination=np.array([2.0]), delta_theta=np.array([-0.5]),
        beta=np.array([[1.5]]), design=np.array([[-0.2]]), judge=np.array([0]),
        source_index=np.array([0]), condition_index=np.array([0]),
    )[0]

    assert np.isclose(forward + reverse, 0.8)
```

Add:

```python
def test_model_sites_have_expected_shapes():
    from numpyro import handlers
    from bias_model import judge_bias_model

    values = handlers.trace(handlers.seed(judge_bias_model, rng_seed=0)).get_trace(
        obs=np.array([1.0, 0.0]), judge=np.array([0, 1]),
        source_index=np.array([0, 1]), condition_index=np.array([0, 1]),
        delta_theta=np.array([1.0, -1.0]), design=np.ones((2, 2)),
        n_judges=2, n_sources=2, n_conditions=2,
    )

    assert values["judge_beta"]["value"].shape == (2, 2)
    assert values["judge_discrimination"]["value"].shape == (2,)
    assert np.allclose(values["judge_source"]["value"][:, 0], 0)
    assert np.allclose(values["judge_condition"]["value"][:, 0], 0)
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_model.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement one direct NumPyro model**

Port the non-centred hierarchy already used in `scripts/hibayes/model.py`; do not depend on the separate `hibayes` environment.

```python
def judge_bias_model(obs, judge, source_index, condition_index, delta_theta,
                     design, n_judges, n_sources, n_conditions):
    alpha_mean = numpyro.sample("alpha_mean", dist.Normal(0, 1))
    alpha_scale = numpyro.sample("alpha_scale", dist.HalfNormal(0.5))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0, 1).expand([n_judges]))
    alpha = numpyro.deterministic("judge_alpha", alpha_mean + alpha_scale * alpha_raw)

    d_mean = numpyro.sample("d_mean", dist.Normal(1, 0.5))
    d_scale = numpyro.sample("d_scale", dist.HalfNormal(0.5))
    d_raw = numpyro.sample("d_raw", dist.Normal(0, 1).expand([n_judges]))
    discrimination = numpyro.deterministic(
        "judge_discrimination", d_mean + d_scale * d_raw,
    )

    n_features = design.shape[1]
    beta_mean = numpyro.sample("beta_mean", dist.Normal(0, 0.5).expand([n_features]))
    beta_scale = numpyro.sample("beta_scale", dist.HalfNormal(0.5).expand([n_features]))
    beta_raw = numpyro.sample(
        "beta_raw", dist.Normal(0, 1).expand([n_judges, n_features]).to_event(2),
    )
    beta = numpyro.deterministic("judge_beta", beta_mean + beta_scale * beta_raw)

    source_free = numpyro.sample(
        "source_free", dist.Normal(0, 0.25).expand([n_judges, max(n_sources - 1, 0)]).to_event(2),
    ) if n_sources > 1 else jnp.zeros((n_judges, 0))
    condition_free = numpyro.sample(
        "condition_free", dist.Normal(0, 0.25).expand([n_judges, max(n_conditions - 1, 0)]).to_event(2),
    ) if n_conditions > 1 else jnp.zeros((n_judges, 0))
    source = jnp.concatenate([jnp.zeros((n_judges, 1)), source_free], axis=1)
    condition = jnp.concatenate([jnp.zeros((n_judges, 1)), condition_free], axis=1)
    numpyro.deterministic("judge_source", source)
    numpyro.deterministic("judge_condition", condition)

    logits = alpha[judge] + source[judge, source_index] + condition[judge, condition_index]
    logits += discrimination[judge] * delta_theta
    logits += jnp.sum(beta[judge] * design, axis=1)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)
```

`fit_calibration` sorts and encodes labels, standardizes nuisance differences using calibration rows only, and runs `MCMC(NUTS(judge_bias_model), ..., num_chains=4, chain_method="parallel")`. It returns arrays plus standardization means/scales, divergence counts, and `numpyro.diagnostics.summary(..., group_by_chain=True)` so R-hat/ESS checks are persisted. Reject zero-variance features rather than divide by zero.

Add `simulate_calibration(seed=0, n_judges=3, n_rows=3000)` using fixed truth `alpha=[-0.3, 0, 0.3]`, `discrimination=[0.5, 1, 1.5]`, and two nuisance coefficients `[0.8, -0.5]`. It returns the generated frame and truth table. The HPC `validate-model` command fits this frame with 300/300 draws and writes whether every population effect has the correct sign and its truth lies inside the 90% posterior interval.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_model.py -q
```

Expected: PASS without a production MCMC run.

- [ ] **Step 5: Commit**

```bash
git add scripts/irt_proxy/bias_model.py scripts/irt_proxy/test_bias_model.py
git commit -m "feat: fit hierarchical judge bias calibration"
```

---

### Task 5: Infer Held-Out Abilities Without Bias Feedback

**Files:**
- Modify: `scripts/irt_proxy/bias_model.py`
- Modify: `scripts/irt_proxy/test_bias_model.py`

**Interfaces:**
- Produces: `infer_targets(target_frame, anchors, calibration, warmup, samples, seed, max_calibration_draws=200) -> pd.DataFrame`.
- Returns one row per target with `theta`, `theta_p5`, `theta_p95`; frozen calibration arrays are empirical nuisance priors and are never sampled or updated.

- [ ] **Step 1: Write a cut-likelihood test**

Add:

```python
def test_target_inference_uses_frozen_bias_offset():
    from bias_model import infer_targets

    frame = pd.DataFrame([
        {"judge": "j", "source": "s", "condition": "c", "left_model": "t",
         "right_model": anchor, "chosen_left": 1.0, "delta_log_chars": 1.0}
        for anchor in ("a", "b") for _ in range(10)
    ])
    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 0.0]})

    def calibration(beta):
        return {
            "judges": ["j"], "sources": ["s"], "conditions": ["c"],
            "features": ["delta_log_chars"], "feature_mean": np.array([0.0]),
            "feature_scale": np.array([1.0]), "alpha": np.zeros((2, 1)),
            "source": np.zeros((2, 1, 1)), "condition": np.zeros((2, 1, 1)),
            "discrimination": np.ones((2, 1)),
            "beta": np.full((2, 1, 1), beta),
        }

    neutral = infer_targets(frame, anchors, calibration(0.0), warmup=20, samples=20, seed=0)
    offset = infer_targets(frame, anchors, calibration(1.0), warmup=20, samples=20, seed=0)

    assert neutral.loc[0, "theta"] > 0
    assert offset.loc[0, "theta"] < neutral.loc[0, "theta"]
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_model.py::test_target_inference_uses_frozen_bias_offset -q
```

Expected: FAIL because `infer_targets` is undefined.

- [ ] **Step 3: Implement the empirical cut posterior**

Use one global calibration draw per likelihood component, not a different nuisance draw per verdict:

```python
def cut_target_model(obs, left_target, right_target, left_anchor, right_anchor,
                     judge, source_index, condition_index, design, calibration,
                     n_targets):
    theta = numpyro.sample("theta", dist.Normal(0, 2).expand([n_targets]))
    left = jnp.where(left_target >= 0, theta[jnp.maximum(left_target, 0)], left_anchor)
    right = jnp.where(right_target >= 0, theta[jnp.maximum(right_target, 0)], right_anchor)
    delta = left - right

    logits = calibration["alpha"][:, judge]
    logits += calibration["source"][:, judge, source_index]
    logits += calibration["condition"][:, judge, condition_index]
    logits += calibration["discrimination"][:, judge] * delta[None, :]
    logits += jnp.sum(calibration["beta"][:, judge, :] * design[None, :, :], axis=-1)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs[None, :]).sum(axis=1)
    numpyro.factor("cut_obs", jax.scipy.special.logsumexp(log_likelihood)
                    - jnp.log(log_likelihood.shape[0]))
```

Subsample calibration draws deterministically with `np.linspace` when more than 200 are present. Standardize target nuisance features with the saved calibration means/scales. Reject unseen judge/source/condition levels with a coverage reason; do not silently encode them as baseline.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/irt_proxy/bias_model.py scripts/irt_proxy/test_bias_model.py
git commit -m "feat: infer held-out abilities with a cut posterior"
```

---

### Task 6: Run Cross-Fitted Corrected and Baseline Predictions

**Files:**
- Modify: `scripts/irt_proxy/bias_crossfit.py`
- Modify: `scripts/irt_proxy/test_bias_crossfit.py`
- Modify: `scripts/irt_proxy/test_prediction.py`

**Interfaces:**
- Produces: `run_fold(...) -> dict[str, pd.DataFrame]` and `reduce_folds(...) -> dict[str, pd.DataFrame]`.
- Prediction schema: `benchmark, fold, component, method, model, predictor, prediction, prediction_p5, prediction_p95, observed` plus coverage counts.

- [ ] **Step 1: Write a dependency-injected fold test**

Add:

```python
def test_run_fold_hides_target_outcome_until_after_inference():
    from bias_crossfit import run_fold

    seen = {}
    calibration = pd.DataFrame([{
        "left_model": "a", "right_model": "b", "delta_theta": 1.0,
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
    }])
    target = pd.DataFrame([{
        "left_model": "held", "right_model": "a", "item_id": "easy",
        "judge": "j", "source": "s", "condition": "c", "chosen_left": 1.0,
    }])
    hard = pd.DataFrame({"model": ["a", "b", "held"], "hard_accuracy": [0.8, 0.4, 0.9]})

    def fake_fit(frame, **_kwargs):
        seen["fit_models"] = set(frame["left_model"]) | set(frame["right_model"])
        return {"posterior": "fixed"}

    def fake_infer(frame, anchors, fit, **_kwargs):
        assert "hard_accuracy" not in frame
        seen["target_items"] = set(frame["item_id"])
        return pd.DataFrame({"model": ["held"], "theta": [1.2],
                             "theta_p5": [0.8], "theta_p95": [1.6]})

    got = run_fold(
        fold=0, calibration_frame=calibration, target_frame=target,
        anchors=pd.DataFrame({"model": ["a", "b"], "theta": [1.0, 0.0]}),
        hard_accuracy=hard, fit_calibration_fn=fake_fit,
        infer_targets_fn=fake_infer,
    )["predictions"]

    assert seen == {"fit_models": {"a", "b"}, "target_items": {"easy"}}
    assert set(got["method"]) == {"corrected", "uncorrected", "raw_elo", "easy_accuracy", "mean"}
    assert set(got["model"]) == {"held"}
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_crossfit.py scripts/irt_proxy/test_prediction.py -q
```

Expected: FAIL because `run_fold` and `reduce_folds` do not exist.

- [ ] **Step 3: Implement the five methods with existing helpers**

For each IRT model and fold:

1. run `fit_calibration(..., feature_columns=BIAS_FEATURES)` and `infer_targets(...)` for `corrected`;
2. repeat with `feature_columns=()` for `uncorrected`;
3. use existing item-level counterbalancing and Bradley–Terry helpers for `raw_elo`;
4. calculate target easy accuracy from canonical response cells; and
5. use the calibration-model mean hard accuracy for `mean`.

The corrected arm also reruns conditional target inference with each nuisance feature's saved coefficient set to zero, producing diagnostic methods named `corrected_without_<feature>`. These diagnostics reuse the full corrected GLM posterior; they do not launch seven extra GLM fits.

Save posterior-predictive calibration summaries by judge and source: observed left-choice rate, posterior predicted rate, Brier score, log loss, and 90% predictive-interval coverage.

Map every predictor to hard accuracy using only calibration models:

```python
def calibrate_to_hard(train_predictor, train_hard, test_predictor):
    design = np.column_stack([np.ones(len(train_predictor)), train_predictor])
    intercept, slope = np.linalg.lstsq(design, train_hard, rcond=None)[0]
    return np.clip(intercept + slope * np.asarray(test_predictor), 0, 1)
```

Align raw Elo to the frozen anchor scale within each connected component before this mapping. Keep components with fewer than two anchors in coverage only, with no predictions.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest scripts/irt_proxy/test_bias_crossfit.py scripts/irt_proxy/test_prediction.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/irt_proxy/bias_crossfit.py scripts/irt_proxy/test_bias_crossfit.py scripts/irt_proxy/test_prediction.py
git commit -m "feat: cross-fit corrected capability predictions"
```

---

### Task 7: Add Cross-Fitted Metrics and Nature-Style Figures

**Files:**
- Modify: `scripts/irt_proxy/metrics.py`
- Create: `scripts/irt_proxy/correction_figures.py`
- Create: `scripts/irt_proxy/test_correction_figures.py`
- Modify: `scripts/irt_proxy/test_prediction.py`

**Interfaces:**
- Produces: `correction_metrics(predictions)`, `capability_gap_rows(predictions)`, and `render_correction_figures(predictions, gap_pairs, out_dir)`.

- [ ] **Step 1: Write metric and render tests**

Add:

```python
def prediction_fixture():
    return pd.DataFrame([
        {"benchmark": "gpqa", "fold": model % 5, "component": "c0",
         "method": method, "model": f"m{model}", "predictor": value,
         "prediction": value, "prediction_p5": max(0, value - 0.05),
         "prediction_p95": min(1, value + 0.05), "observed": model / 5}
        for method in ("corrected", "uncorrected", "raw_elo", "easy_accuracy", "mean")
        for model, value in enumerate(np.linspace(0.1, 0.9, 5))
    ])


def test_correction_metrics_and_gap_pairs_use_out_of_fold_predictions():
    from metrics import capability_gap_rows, correction_metrics

    predictions = prediction_fixture()
    metrics = correction_metrics(predictions)
    pairs = capability_gap_rows(predictions)

    assert {"tau_a", "pair_accuracy", "mae", "rmse", "calibration_slope",
            "calibration_intercept", "interval_coverage"}.issubset(metrics.columns)
    assert len(pairs.query("method == 'corrected'")) == 10
    assert np.allclose(pairs["estimated_gap"], pairs["prediction_left"] - pairs["prediction_right"])


def test_correction_figures_are_small_and_keep_svg_text(tmp_path):
    import struct
    from correction_figures import render_correction_figures
    from metrics import capability_gap_rows

    predictions = prediction_fixture()
    render_correction_figures(predictions, capability_gap_rows(predictions), tmp_path)

    for stem in ("capability_predictions", "gap_curve"):
        png, svg = tmp_path / f"{stem}.png", tmp_path / f"{stem}.svg"
        width, _ = struct.unpack(">II", png.read_bytes()[16:24])
        assert width <= 900
        assert "<text" in svg.read_text()
        assert "Bias-corrected capability" not in svg.read_text()
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_prediction.py scripts/irt_proxy/test_correction_figures.py -q
```

Expected: FAIL because the new functions/modules do not exist.

- [ ] **Step 3: Implement metrics using the existing model-level unit**

Reuse `kendall_tau_a`. For uncertainty, resample model IDs and recompute method differences on matched models; never resample verdict rows. Build fixed 0.05-wide true-gap bins and retain empty-bin counts in the table while omitting empty marks from figures.

- [ ] **Step 4: Implement the two compact figures**

Use the standing style exactly:

```python
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
})
```

- `capability_predictions`: two columns (`Raw Elo`, `Corrected`), shared 0–1 axes, observed hard accuracy on x and predicted on y, 2.9 × 1.7 inches maximum.
- `gap_curve`: one 1.8 × 1.6 inch panel; raw Elo grey, uncorrected orange, corrected blue; 1 pt lines and 0.15-alpha model-bootstrap bands.
- Hide top/right spines, move left/bottom outward 5 pt, use 0.8 pt dashed grey reference lines, set explicit endpoint ticks, and place a frame-free legend below the axes.
- Save PNG at 300 dpi and SVG with editable text.

- [ ] **Step 5: Render and visually inspect**

Run the synthetic rendering test, then use `view_image` on both PNGs. Check size, tick overlap, outside legends, clipping, line/band agreement, and visible endpoint ticks. Adjust only if the rendered evidence fails the standing checklist.

- [ ] **Step 6: Run tests**

```bash
.venv/bin/pytest scripts/irt_proxy/test_prediction.py scripts/irt_proxy/test_correction_figures.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/irt_proxy/metrics.py scripts/irt_proxy/correction_figures.py scripts/irt_proxy/test_prediction.py scripts/irt_proxy/test_correction_figures.py
git commit -m "feat: report corrected capability estimates"
```

---

### Task 8: Add the Restartable SWC Fit Commands

**Files:**
- Create: `scripts/irt_proxy/correct_biases.py`
- Create: `scripts/irt_proxy/bias_reference_fit.sbatch`
- Create: `scripts/irt_proxy/bias_fold_fit.sbatch`
- Modify: `scripts/irt_proxy/config.yaml`
- Modify: `scripts/irt_proxy/test_run.py`
- Modify: `docs/2026-08-17-correcting-biases.md`

**Interfaces:**
- CLI commands: `validate-model`, `prepare`, `fit-reference`, `fit-fold --irt-model fixed_3pl|2pl --fold N --model corrected|uncorrected`, and `reduce`.
- Slurm reference job creates prepared Parquet inputs, the frozen reference posterior, item splits, and anchors.
- Slurm array maps tasks 0–19 to two IRT forms × five folds × corrected/uncorrected models.

- [ ] **Step 1: Write CLI/output-contract tests**

Add to `scripts/irt_proxy/test_run.py`:

```python
def test_bias_cli_parses_only_declared_fold_arms():
    from correct_biases import parse_args

    args = parse_args(["fit-fold", "--irt-model", "fixed_3pl", "--fold", "3", "--model", "corrected"])
    assert (args.command, args.irt_model, args.fold, args.model) == (
        "fit-fold", "fixed_3pl", 3, "corrected",
    )
    with pytest.raises(SystemExit):
        parse_args(["fit-fold", "--irt-model", "fixed_3pl", "--fold", "5", "--model", "corrected"])
    with pytest.raises(SystemExit):
        parse_args(["fit-fold", "--irt-model", "fixed_3pl", "--fold", "1", "--model", "other"])
    with pytest.raises(SystemExit):
        parse_args(["fit-fold", "--irt-model", "3pl", "--fold", "1", "--model", "corrected"])


def test_bias_table_writer_enforces_contract(tmp_path):
    from correct_biases import write_tables

    tables = {name: pd.DataFrame({"x": [1]}) for name in (
        "predictions.csv", "coefficients.csv", "coverage.csv", "metrics.csv", "gap_pairs.csv",
    )}
    write_tables(tables, tmp_path)
    assert set(path.name for path in tmp_path.glob("*.csv")) == set(tables)
    with pytest.raises(ValueError, match="coverage.csv"):
        write_tables({key: value for key, value in tables.items() if key != "coverage.csv"}, tmp_path)
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv/bin/pytest scripts/irt_proxy/test_run.py -q
```

Expected: FAIL because `correct_biases.py` is absent.

- [ ] **Step 3: Implement the four commands**

`validate-model` performs Task 4's synthetic coefficient-recovery fit and writes `analysis/bias_correction/synthetic_recovery.csv`. `prepare` loads and validates source data, writes Parquet atomically, and prints source/judge/model/item counts. `fit-reference` runs Task 2 for fixed-floor 3PL and ordinary 2PL and writes separately keyed posteriors, item splits, and anchors. `fit-fold` runs one Task 6 arm and refuses to replace an existing final artifact. `reduce` requires all twenty fold-arm artifacts, calculates tables/figures, and never refits.

The local test suite may call `prepare` on tiny fixtures. The production commands are marked in `--help` as SWC compute-node operations.

- [ ] **Step 4: Write the reference Slurm job**

Create `scripts/irt_proxy/bias_reference_fit.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=bias-reference
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/bias-reference-%j.out
#SBATCH --error=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/bias-reference-%j.out
set -euo pipefail

BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
RUN=$BASE/staging/bias_correction_20260817_v1
SITE=$RUN/site
PYTHON=/usr/bin/python3.12
WHEELS=$BASE/staging/mc_cost_preview_20260815/wheelhouse
cd "$RUN"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [ ! -f "$SITE/.ready" ]; then
  mkdir -p "$SITE"
  PYTHONPATH="$WHEELS/pip-26.0.1-py3-none-any.whl" \
    "$PYTHON" -m pip install --target "$SITE" --no-index --find-links "$WHEELS" \
      inspect-ai==0.3.251 jax==0.11.0 jaxlib==0.11.0 numpyro==0.21.0 \
      numpy==2.5.1 pandas==3.0.5 pyarrow==25.0.0 scipy==1.18.0 \
      pyyaml zipfile-zstd==0.0.4
  PYTHONPATH="$SITE:$RUN/src:$RUN/scripts/irt_proxy" \
    "$PYTHON" -c "import inspect_ai,jax,numpyro,pandas,pyarrow,scipy,yaml"
  touch "$SITE/.ready"
fi
export PYTHONPATH="$SITE:$RUN/src:$RUN/scripts/irt_proxy"

[ -f analysis/bias_correction/synthetic_recovery.csv ] || \
  "$PYTHON" scripts/irt_proxy/correct_biases.py validate-model --config scripts/irt_proxy/config.yaml
[ -f analysis/bias_correction/input/manifest.json ] || \
  "$PYTHON" scripts/irt_proxy/correct_biases.py prepare --config scripts/irt_proxy/config.yaml
[ -f analysis/bias_correction/reference/posterior.npz ] || \
  "$PYTHON" scripts/irt_proxy/correct_biases.py fit-reference --config scripts/irt_proxy/config.yaml
```

The `.ready` marker is created by the reference job before the dependent array starts, preventing concurrent installation races.

- [ ] **Step 5: Write the fold-array job**

Create `scripts/irt_proxy/bias_fold_fit.sbatch` with the same CPU/memory/time/requeue/log directives plus `#SBATCH --array=0-19%3`. It must require `$RUN/site/.ready` and the frozen reference posterior, then run:

```bash
FORM_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
INNER=$((SLURM_ARRAY_TASK_ID % 10))
FOLD=$((INNER / 2))
if [ "$FORM_INDEX" -eq 0 ]; then IRT_MODEL=fixed_3pl; else IRT_MODEL=2pl; fi
if [ $((INNER % 2)) -eq 0 ]; then MODEL=corrected; else MODEL=uncorrected; fi
OUT="analysis/bias_correction/folds/$IRT_MODEL/fold-$FOLD/$MODEL.npz"
[ -f "$OUT" ] || "$PYTHON" scripts/irt_proxy/correct_biases.py fit-fold \
  --config scripts/irt_proxy/config.yaml --irt-model "$IRT_MODEL" \
  --fold "$FOLD" --model "$MODEL"
```

`fit-fold` writes to a same-directory temporary file and atomically renames it. Requeues see the final file and skip.

- [ ] **Step 6: Add exact staging/submission instructions**

Document these commands in the design note:

```bash
RUN=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/bias_correction_20260817_v1
ssh swc "test ! -e '$RUN' && mkdir -p '$RUN/interactive_viewer/data' '$RUN/data' '$RUN/external_data' '$RUN/judge_results/judge_ft' '$RUN/judge_results/judging_variations'"
rsync -a --ignore-existing pyproject.toml uv.lock scripts src configs swc:"$RUN"/
rsync -a --ignore-existing interactive_viewer/data/gpqa-pilot-rd2 interactive_viewer/data/data-expansion-gpqa swc:"$RUN"/interactive_viewer/data/
rsync -a --ignore-existing data/hf-judging-2026-08-17 data/gpqa swc:"$RUN"/data/
rsync -a --ignore-existing external_data/epoch-gpqa swc:"$RUN"/external_data/
rsync -a --ignore-existing judge_results/judge_ft/pairs.parquet swc:"$RUN"/judge_results/judge_ft/
rsync -a --ignore-existing --include='*/' --include='verdicts_*.jsonl' --exclude='*' judge_results/judge_ft/ swc:"$RUN"/judge_results/judge_ft/
rsync -a --ignore-existing judge_results/judging_variations/verdicts.csv swc:"$RUN"/judge_results/judging_variations/
REF_JOB=$(ssh swc "cd '$RUN' && sbatch --parsable scripts/irt_proxy/bias_reference_fit.sbatch")
ssh swc "cd '$RUN' && sbatch --dependency=afterok:$REF_JOB scripts/irt_proxy/bias_fold_fit.sbatch"
```

At execution time, first use read-only `ssh swc` checks to confirm `$RUN` is absent and the wheelhouse contains every pinned wheel. If `$RUN` already exists, choose a new versioned directory; never reuse or remove it.

After the array completes, submit `reduce` as a small CPU job or pull only the twenty compact fold artifacts and run `reduce` locally. Do not run any fit command locally.

- [ ] **Step 7: Run all local tests**

```bash
.venv/bin/pytest scripts/irt_proxy -q
.venv/bin/pytest tests -q
```

Expected: PASS. These commands use synthetic fixtures and do not invoke production fitting.

- [ ] **Step 8: Shell-check Slurm scripts**

```bash
bash -n scripts/irt_proxy/bias_reference_fit.sbatch
bash -n scripts/irt_proxy/bias_fold_fit.sbatch
```

Expected: both exit 0.

- [ ] **Step 9: Commit**

```bash
git add scripts/irt_proxy/correct_biases.py scripts/irt_proxy/bias_reference_fit.sbatch scripts/irt_proxy/bias_fold_fit.sbatch scripts/irt_proxy/config.yaml scripts/irt_proxy/test_run.py docs/2026-08-17-correcting-biases.md
git commit -m "feat: run bias correction safely on SWC"
```

---

## Final Verification

- [ ] Run every local test and both `bash -n` checks from Task 8.
- [ ] Inspect `git diff --check` and confirm no unrelated user file is staged.
- [ ] Stage to a new Ceph run directory with `rsync --ignore-existing`; do not alter existing Ceph data.
- [ ] Submit the reference job, then the dependent fold array.
- [ ] Monitor with `squeue`, `sacct`, and `tail` only; run no Python on the gateway.
- [ ] Confirm each fit reports no divergences, `r_hat <= 1.01`, and adequate bulk/tail ESS before reduction.
- [ ] Confirm the synthetic recovery table has the correct sign for every planted population effect and 90% intervals covering the planted values.
- [ ] Run `reduce`, visually inspect both figures, and verify editable SVG text and final canvas dimensions.
- [ ] Request code review before claiming the implementation complete.
