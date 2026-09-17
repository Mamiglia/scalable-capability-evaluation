# Results Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the five conditions (intercept, pairwise judge, direct judge, surface-feature ridge, embedding ridge) on the semester forecasting task against the accuracy oracle, per domain and semester, and produce the results table and figures.

**Architecture:** Reusable steps in `src/analysis/` (loading and matching are done; baselines, embeddings, judges, calibration, stats, plots follow), a `# %%` cell runner in `scripts/analyses/results.py`. Judge conversions come from `src/protocols/pairwise/{linear,calibrated}.py` (merged from the teammate's branch) applied offline to matched tables; direct calibration is implemented here. Embeddings are read from the HF `.npz` files on the T7 volume.

**Tech Stack:** Python 3.13, pandas, NumPy, scikit-learn, SciPy, matplotlib. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-results-analysis-design.md` (revised version)

## Global Constraints

- Every protocol judges only the frozen train set (`<semester>_train` in `data/meta/<domain>/items.csv`), at most 100 items per task; never a widened set.
- Every fit, calibration and normalisation statistic is computed on the references only; target θ\* is read only by the metric.
- Nothing in `src/` imports `scripts/`.
- The analysis never launches judging, collection or an unscoped LFS pull; judge runs are read from `logs/` and the unjudged remainder is reported in a `missing` column.
- Transcript logs are read through `reference.tables.DATA` (env `SMALL_JUDGE_DATA=/Volumes/T7/small-judge-remote/data`); embeddings through env `SMALL_JUDGE_EMBEDDINGS=/Volumes/T7/small-judge-remote/transcript-embeddings`. Real-data checks set both.
- Every function has a docstring stating what it computes, its inputs (types; index and columns of tables) and its output (shape, index, columns, NaN meaning). Non-obvious steps get an inline comment.
- Names describe what a thing is (`accuracy_hard`, `best judge on references`), never a stage label.
- Figures: 8pt Arial, panels ≤ ~2in, 1pt result lines, 0.8pt dashed grey reference lines, no top/right spines, legend outside, `svg.fonttype: none`, no titles on the canvas.
- Commit messages: `area: what changed`, no attribution trailers.
- `docs/` is gitignored; the spec and this plan are not committed.

## Starting state

Committed on `methodology`: `src/analysis/{__init__,loading,matching}.py`, `src/analysis/challenges/__init__.py`, `tables.load_responses`, the `SMALL_JUDGE_DATA` override, and the merge of `origin/calibrated-bt` (`src/protocols/pairwise/linear.py`: `weighted_theta(verdicts, refs, targets)`; `src/protocols/pairwise/calibrated.py`: `calibrate`, `estimate`, `estimate_all`, `loo`, `shrink`, `JudgeCalibrated.frontier_prior`). Untracked drafts of `baselines.py`, `embeddings.py`, `calibration.py`, `stats.py`, `plots.py`, `tests/test_analysis.py` and `scripts/analyses/results.py` exist from an earlier plan; each task below replaces its file with the content shown. Do not diff against the drafts.

Run every command from the repository root. Tests: `uv run pytest tests/test_analysis.py -q`. Seven tests elsewhere in the suite (`test_cli`, `test_paths`, `test_runner`, `test_task`) fail before this work and are unrelated.

Real-data checks: prefix commands with `PYTHONPATH=src SMALL_JUDGE_DATA=/Volumes/T7/small-judge-remote/data SMALL_JUDGE_EMBEDDINGS=/Volumes/T7/small-judge-remote/transcript-embeddings uv run` and they print `[WARNING]` lines about excluded historical models; that is expected. Never run `git lfs pull` or any download in these tasks.

## File structure

| File | Responsibility |
| --- | --- |
| `src/analysis/baselines.py` | `Intercept`, `AccuracyOracle` (Protocol subclasses) |
| `src/analysis/embeddings.py` | `EmbeddingRegression(variant)` reading the `.npz` files |
| `src/analysis/judges.py` | `pairwise_theta`, `pairwise_reference_loo`, `direct_theta`, `direct_reference_loo` over matched tables |
| `src/analysis/calibration.py` | `transcript_protocols(dom)`; `forecasts(dom, task, protocols, runs)` with one row per target × protocol × variant × judge × condition |
| `src/analysis/stats.py` | `mse_row`, `core_condition`, `judge_summaries`, `scores`, `summary`, `task_metrics` (raw / delta / paired-bootstrap z) |
| `src/analysis/plots.py` | `forecast_figure`, `bar_figure`, `semester_table` for a chosen variant per protocol |
| `scripts/analyses/results.py` | the `# %%` runner with the three toggles |
| `tests/test_analysis.py` | baselines, direct calibration, reference LOO, stats on toy data |

---

### Task 4: Baselines (intercept and oracle)

**Files:**
- Create: `src/analysis/baselines.py`
- Create: `tests/test_analysis.py` (first test)

**Interfaces:**
- Consumes: `tables.load(domain)` (items table with `<semester>_set`), `tables.load_responses`.
- Produces: `Intercept()` (name `intercept`), `AccuracyOracle(domain)` (name `accuracy_hard`); both `protocols.Protocol` instances callable on a `Task`.

- [ ] **Step 1: Write the failing test**

`tests/test_analysis.py`:

```python
"""Baselines, judge conversions, scores and judge summaries on toy data (the toy task from test_protocols)."""

import numpy as np
import pandas as pd

from analysis.baselines import AccuracyOracle, Intercept
from test_protocols import toy_task


def test_intercept_and_oracle(monkeypatch):
    t = toy_task()  # references a, b, c with theta 0, 1, 2 on train items "0", "1"; targets t, u
    assert Intercept()(t).eq(1.0).all()
    # a HARD item "9" over two repeats: a 0/2, b 1/2, c 2/2, t 2/2, u 0/2 -> theta = 2 * accuracy on the toy
    rows = [dict(bench="b", item="9", model=m, repeat=r, correct=int(r < k))
            for m, k in {"a": 0, "b": 1, "c": 2, "t": 2, "u": 0}.items() for r in (0, 1)]
    rows += [dict(bench="b", item="0", model=m, repeat=0, correct=1) for m in "abctu"]  # a train item: ignored
    items = pd.DataFrame({"bench": "b", "item": ["0", "1", "9"], "s_set": ["easy", "easy", "hard"]})
    monkeypatch.setattr("analysis.baselines.tables.load_responses", lambda dom: pd.DataFrame(rows))
    monkeypatch.setattr("analysis.baselines.tables.load", lambda dom: (None, items))
    oracle = AccuracyOracle(None)
    assert oracle.name == "accuracy_hard"
    np.testing.assert_allclose(oracle(t).to_numpy(), [2.0, 0.0], atol=1e-9)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_analysis.py -q`
Expected: FAIL (ImportError: no `Intercept` / `AccuracyOracle`).

- [ ] **Step 3: Write the module**

```python
"""Baselines that read no transcript: the intercept, and theta* calibrated on HARD-item accuracy.

Intercept: every target forecast as the references' mean theta*. Accuracy on the saturated
train items is 1 for every model, so a regression of theta* on it is exactly this intercept;
a condition that cannot beat it has learned nothing.

Oracle (accuracy_hard): OLS of theta* on mean accuracy over every repeat on this task's HARD
items, the responses theta* itself was fit on. Calibrated on the 8 references, so it is not
exact: with weak references and strong targets the line extrapolates.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from capability.task import Cohort, Task
from configs.domain import DomainConfig
from protocols import Protocol
from reference import tables


class Intercept(Protocol):
    """Forecast every target as the mean theta* of the previous semester's references.

    Uses task.references only (not history), like the judge conversions.
    """

    name = "intercept"

    def __call__(self, task: Task) -> pd.Series:
        self.fit(task.references)
        return self.predict(task.test)

    def fit(self, train: Cohort) -> None:
        """Store the mean theta* of `train` (NaN if none is finite)."""
        self.mean = float(train.theta.mean())

    def predict(self, test: Cohort) -> pd.Series:
        """The stored mean for every model of `test`, indexed like test.models."""
        return pd.Series(self.mean, index=test.models.index)


class AccuracyOracle(Protocol):
    """OLS of the training models' theta* on their HARD-item accuracy, applied to the targets.

    Accuracy per model = mean of `correct` over every repeat and every item whose
    `<semester>_set` is "hard" in the frozen items table. Protocol name accuracy_hard.
    """

    name = "accuracy_hard"

    def __init__(self, domain: DomainConfig):
        self.responses = tables.load_responses(domain)  # bench, item, model, repeat, correct
        self.items = tables.load(domain)[1]  # bench, item, ..., <semester>_set (easy / hard / "")
        self.estimator: LinearRegression | None = None

    def accuracy(self, cohort: Cohort) -> pd.Series:
        """HARD-item accuracy per model of `cohort` (index = cohort.models.index; NaN for a
        model with no HARD response on this task)."""
        hard = self.items[self.items[f"{cohort.semester}_set"] == "hard"][["bench", "item"]]
        return self.responses.merge(hard).groupby("model").correct.mean().reindex(cohort.models.index)

    def fit(self, train: Cohort) -> None:
        """Fit theta* ~ accuracy on the models of `train` with both known; fewer than two
        usable models leaves the estimator None (every forecast NaN)."""
        x, y = self.accuracy(train), train.theta
        known = np.isfinite(x) & np.isfinite(y)
        self.estimator = LinearRegression().fit(x[known].to_frame(), y[known]) if known.sum() >= 2 else None

    def predict(self, test: Cohort) -> pd.Series:
        """theta-hat per model of `test` (index = test.models.index); NaN where the model has
        no HARD accuracy or nothing was fit."""
        x = self.accuracy(test)
        prediction = pd.Series(np.nan, index=test.models.index)
        if self.estimator is not None and (known := np.isfinite(x)).any():
            prediction[known] = self.estimator.predict(x[known].to_frame())
        return prediction
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_analysis.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/baselines.py tests/test_analysis.py
git commit -m "analysis: intercept baseline and HARD-item accuracy oracle"
```

---

### Task 5: Embedding ridge from the HF `.npz` files

**Files:**
- Create: `src/analysis/embeddings.py`

**Interfaces:**
- Consumes: `protocols.features.TabularRegression`, `Cohort.transcripts` (model → (bench, item) → Transcript), `Cohort.models["path_<bench>"]`.
- Produces: `EmbeddingRegression(variant="full")`, name `embedding`; `EMBEDDINGS` root from env.

The `.npz` layout (verified on T7): path `<log path minus .eval>.<variant>.npz`, e.g. `gpqa/b0a0d9c9/deepseek-r1-0528/0/LiRiryJsGb3h5UuxX8bmJv.full.npz` for the frozen log `gpqa/b0a0d9c9/deepseek-r1-0528/0/LiRiryJsGb3h5UuxX8bmJv.eval`; arrays `ids` (str sample id, e.g. `rec06pnAkLOr2t2mp`), `epochs` (int32, 1 = first attempt), `correct` (bool), `n_tokens`, `n_chunks`, `embedding` (float16 `[n, 4096]`, L2-normalised), `meta` (json str). The cohort's transcripts are already the correct first attempts, so a model's row is the mean over the vectors whose id is in its transcript keys for that bench.

- [ ] **Step 1: Write the module**

```python
"""Ridge on the mean transcript embedding of a model's correct train transcripts.

Vectors come from the HF dataset LASR-G5/transcript-embeddings (Qwen3-Embedding-8B), one
.npz per collection log at the log's path with `.eval` replaced by `.<variant>.npz`, under
the directory named by SMALL_JUDGE_EMBEDDINGS. Variant "full" embeds the transcript with its
reasoning (what a judge shown reasoning sees), "answer" the reasoning-stripped text. A
model's feature row is the mean of its train transcripts' L2-normalised vectors; the
estimator is ridge with the penalty chosen by leave-one-out over the references, as for the
surface-feature GLM.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from capability.task import Cohort
from configs.logs import logger
from protocols.features import TabularRegression

EMBEDDINGS = Path(os.environ.get("SMALL_JUDGE_EMBEDDINGS", "transcript-embeddings"))
VARIANTS = ("full", "answer")
ALPHAS = np.logspace(-2, 4, 13)  # unit-norm inputs need stronger shrinkage than standardised features


def npz_path(log_path: str, variant: str) -> Path:
    """The embedding file of a data/ log: same relative path, `.eval` -> `.<variant>.npz`."""
    return EMBEDDINGS / (log_path.rsplit(".eval", 1)[0] + f".{variant}.npz")


def read_vectors(path: Path, ids: set[str]) -> np.ndarray | None:
    """The [k, dim] float32 first-attempt vectors of the samples in `ids`, or None when the
    file is missing (reported) or holds none of them."""
    if not path.exists():
        logger.warning("%s: no embedding file; pull it from LASR-G5/transcript-embeddings", path)
        return None
    with np.load(path) as f:
        keep = np.isin(f["ids"], list(ids)) & (f["epochs"] == 1)
        return f["embedding"][keep].astype(np.float32) if keep.any() else None


class EmbeddingRegression(TabularRegression):
    """Fit reference embeddings to theta*, predict target theta* (protocol name "embedding")."""

    def __init__(self, variant: str = "full"):
        assert variant in VARIANTS, variant
        self.variant = variant
        super().__init__("embedding", [], RidgeCV(alphas=ALPHAS))

    def get_features(self, cohort: Cohort) -> pd.DataFrame:
        """One row per model of `cohort` (index = cohort.models.index): the mean vector over its
        correct first-attempt train transcripts across benches; a row of NaN for a model with
        no readable vector. Columns are "0".."<dim-1>"."""
        rows: dict[str, np.ndarray | None] = {}
        for model, transcripts in cohort.transcripts.items():
            ids_by_bench: dict[str, set[str]] = {}
            for bench, item in transcripts:
                ids_by_bench.setdefault(bench, set()).add(item)
            vectors = []
            for bench, ids in ids_by_bench.items():
                path = cohort.models.loc[model, f"path_{bench}"]
                if isinstance(path, str) and (v := read_vectors(npz_path(path, self.variant), ids)) is not None:
                    vectors.append(v)
            rows[model] = np.concatenate(vectors).mean(axis=0) if vectors else None
        dim = next((len(v) for v in rows.values() if v is not None), 0)
        table = pd.DataFrame({m: (v if v is not None else np.full(dim, np.nan)) for m, v in rows.items()}).T
        table.columns = [str(i) for i in range(dim)]
        return table.reindex(cohort.models.index)
```

- [ ] **Step 2: Check on one semester**

Run (real-data prefix): `python -c "from analysis import loading; from analysis.embeddings import EmbeddingRegression; d = loading.domain('science'); t = loading.tasks(d, ['2026-I'])[0]; p = EmbeddingRegression(); print(p.get_features(t.references).shape, p(t).round(2).to_dict())"`
Expected: `(8, 4096)` and eight finite forecasts. If a warning names a missing `.npz`, report it; do not download anything.

- [ ] **Step 3: Commit**

```bash
git add src/analysis/embeddings.py
git commit -m "analysis: embedding ridge from the HF transcript embeddings"
```

---

### Task 6: Judge conversions over matched tables

**Files:**
- Create: `src/analysis/judges.py`
- Modify: `tests/test_analysis.py` (add two tests)

**Interfaces:**
- Consumes: `protocols.pairwise.linear.weighted_theta(verdicts, refs, targets) -> Series`; `protocols.pairwise.calibrated.{calibrate, estimate, loo, shrink}`.
- Produces: `PAIRWISE = ("linear", "calibrated")`, `DIRECT = ("raw", "item_z")`; `pairwise_theta(verdicts, refs, models, conversion, prior=None) -> Series`; `pairwise_reference_loo(verdicts, refs, conversion) -> Series`; `direct_theta(ratings, refs, models, normalisation) -> Series`; `direct_reference_loo(ratings, refs, normalisation) -> Series`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from analysis import judges


def toy_ratings():
    """One bench, two items; references a, b, c (theta 0, 1, 2) rated 2/4/6 on an easy item and 1/2/3
    on a hard one; targets t (rated like c) and u (rated like a)."""
    base = {"a": 0, "b": 1, "c": 2, "t": 2, "u": 0}
    rows = []
    for model, level in base.items():
        for item, scale, offset in [("0", 2, 2), ("1", 1, 1)]:
            rows.append(dict(bench="b", model=model, item=item, score=offset + scale * level, semester="s"))
    return pd.DataFrame(rows)


def test_direct_theta_raw_and_item_z():
    ratings, refs = toy_ratings(), pd.Series({"a": 0.0, "b": 1.0, "c": 2.0})
    raw = judges.direct_theta(ratings, refs, ["t", "u"], "raw")
    np.testing.assert_allclose(raw.to_numpy(), [2.0, 0.0], atol=1e-9)
    z = judges.direct_theta(ratings, refs, ["t", "u"], "item_z")
    np.testing.assert_allclose(z.to_numpy(), [2.0, 0.0], atol=1e-9)
    # a constant item contributes z = 0, not NaN
    flat = pd.concat([ratings, pd.DataFrame(dict(bench="b", model=list("abctu"), item="9", score=5.0, semester="s"))])
    assert judges.direct_theta(flat, refs, ["t"], "item_z").notna().all()
    # the leave-one-out error on the references is exact here: each reference is on the others' line
    loo = judges.direct_reference_loo(ratings, refs, "raw")
    np.testing.assert_allclose(loo.to_numpy(), refs.to_numpy(), atol=1e-9)


def test_pairwise_reference_loo_recovers_references():
    rng = np.random.default_rng(0)
    # 8 references as in the real task: a leave-one-out fold keeps 21 pairs, enough for the per-bench slope
    theta = {f"r{i}": float(t) for i, t in enumerate(np.linspace(-2.5, 2.5, 8), 1)}
    refs = pd.Series(theta)
    rows = []
    for a in theta:
        for b in theta:
            if a >= b:
                continue
            for item in range(80):
                p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
                rows.append(("g", "j", str(item), a, b, float(rng.random() < p), "s", False))
    verdicts = pd.DataFrame(rows, columns=["bench", "judge", "item", "model_left", "model_right", "pick_left", "semester", "imported"])
    for conversion in judges.PAIRWISE:
        loo = judges.pairwise_reference_loo(verdicts, refs, conversion)
        assert loo.index.equals(refs.index) and ((loo - refs).abs() < 0.8).all(), conversion
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_analysis.py -q`
Expected: the two new tests FAIL (no module `analysis.judges`).

- [ ] **Step 3: Write the module**

```python
"""Judge conversions applied offline to matched verdict and rating tables, and the reference
leave-one-out error that selects a judge without target labels.

Pairwise: "linear" is JudgeLinear's per-bench weighted least squares on pair logits
(protocols.pairwise.linear); "calibrated" is JudgeCalibrated's calibrated Bradley-Terry with
shrinkage (protocols.pairwise.calibrated). Direct: OLS of theta* on a model's mean rating,
"raw" or "item_z" (each rating z-scored per (bench, item) with the references' mean and sd).
Every statistic comes from the references passed in; targets never enter a fit.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from protocols.pairwise import calibrated
from protocols.pairwise.linear import weighted_theta

PAIRWISE = ("linear", "calibrated")
DIRECT = ("raw", "item_z")


def pairwise_theta(verdicts: pd.DataFrame, refs: pd.Series, models: list[str], conversion: str,
                   prior: tuple[float, float] | None = None) -> pd.Series:
    """theta-hat of `models` from `verdicts` (bench, item, model_left, model_right, pick_left, ...)
    with the references `refs` (theta* indexed by model) fixed; index = models, NaN without evidence.

    conversion "linear": weighted_theta. "calibrated": calibrate on the reference pairs, 1-D ML
    per model, shrunk by the judge's leave-one-out error on the references toward `prior`
    (mean, variance); prior None means the references' own mean and variance.
    """
    if conversion == "linear":
        return weighted_theta(verdicts, refs, list(models)).reindex(models)
    assert conversion == "calibrated", conversion
    alpha, beta = calibrated.calibrate(verdicts, refs)
    if alpha.empty:
        return pd.Series(np.nan, index=models)
    sigma2 = float(((calibrated.loo(verdicts, refs) - refs) ** 2).mean())
    est = pd.Series({m: calibrated.estimate(verdicts, m, refs, alpha, beta) for m in models}, dtype=float)
    return calibrated.shrink(est, sigma2, prior or (float(refs.mean()), float(refs.var()))).reindex(models)


def pairwise_reference_loo(verdicts: pd.DataFrame, refs: pd.Series, conversion: str) -> pd.Series:
    """Each reference's theta-hat from the reference-vs-reference verdicts only, with the
    conversion calibrated on the other references; index = refs.index."""
    among = verdicts[verdicts.model_left.isin(refs.index) & verdicts.model_right.isin(refs.index)]
    out = {}
    for r in refs.index:
        others = refs.drop(r)
        out[r] = pairwise_theta(among, others, [r], conversion, prior=(float(others.mean()), float(others.var())))[r]
    return pd.Series(out, dtype=float).reindex(refs.index)


def valid_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Rows with a numeric score in [1, 10], one per (model, bench, item): repeated judging
    epochs of one transcript are averaged first, so every transcript weighs the same."""
    values = pd.to_numeric(ratings.score, errors="coerce")
    kept = ratings.assign(score=values)[values.between(1, 10)]
    return kept.groupby(["model", "bench", "item"], as_index=False).score.mean()


def normalise(ratings: pd.DataFrame, refs: pd.Series, normalisation: str) -> pd.DataFrame:
    """Ratings (model, bench, item, score) after the normalisation: "raw" unchanged; "item_z"
    replaces each score by (score - mean) / sd with mean and sd over the references' scores
    of that (bench, item); an item with no spread among the references (sd 0 or a single
    reference) gives z = 0, an item no reference rated is dropped."""
    if normalisation == "raw":
        return ratings
    assert normalisation == "item_z", normalisation
    stats = ratings[ratings.model.isin(refs.index)].groupby(["bench", "item"]).score.agg(["mean", "std"])
    merged = ratings.merge(stats, left_on=["bench", "item"], right_index=True)
    z = ((merged.score - merged["mean"]) / merged["std"]).where(merged["std"] > 0, 0.0).fillna(0.0)
    return merged.assign(score=z)[["model", "bench", "item", "score"]]


def direct_theta(ratings: pd.DataFrame, refs: pd.Series, models: list[str], normalisation: str) -> pd.Series:
    """theta-hat of `models` from `ratings` (bench, model, item, score): OLS of the references'
    theta* on their mean (normalised) rating, applied to each model's mean; index = models,
    NaN for a model without valid ratings or when fewer than two references have both."""
    means = normalise(valid_ratings(ratings), refs, normalisation).groupby("model").score.mean()
    x, y = means.reindex(refs.index), refs
    known = np.isfinite(x) & np.isfinite(y)
    out = pd.Series(np.nan, index=models)
    if known.sum() < 2:
        return out
    line = LinearRegression().fit(x[known].to_frame(), y[known])
    target = means.reindex(models)
    if (has := np.isfinite(target)).any():
        out[has] = line.predict(target[has].to_frame())
    return out


def direct_reference_loo(ratings: pd.DataFrame, refs: pd.Series, normalisation: str) -> pd.Series:
    """Each reference's theta-hat from a calibration (and normalisation) fit on the other
    references only; index = refs.index."""
    among = ratings[ratings.model.isin(refs.index)]
    return pd.Series({r: direct_theta(among, refs.drop(r), [r], normalisation)[r] for r in refs.index}, dtype=float)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_analysis.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/judges.py tests/test_analysis.py
git commit -m "analysis: offline pairwise conversions, direct calibration with item z-scores, reference LOO"
```

---

### Task 7: Calibration (the forecasts table)

**Files:**
- Create: `src/analysis/calibration.py`

**Interfaces:**
- Consumes: `forecast(task, protocol)` from `capability.evaluate`; `PROTOCOLS["glm_regex"]`; Tasks 4–6; `JudgeRun` from `analysis.loading` (fields `protocol, judge, condition, harness, table, missing`; `harness.domain` is the DomainConfig); `JudgeCalibrated.frontier_prior(self, semester)`, which reads only `self.domain`.
- Produces: `transcript_protocols(dom, embedding_variants=("full", "answer")) -> list[tuple[Protocol, str]]`; `judge_forecast(run, task, variant) -> tuple[pd.Series, float]`; `forecasts(dom, task, protocols, runs) -> pd.DataFrame` with `COLUMNS = ["domain", "semester", "protocol", "variant", "judge", "condition", "judge_loo_mse", "model", "theta_hat", "theta", "missing"]`.

- [ ] **Step 1: Write the module**

```python
"""Calibration: every signal to theta-hat on the theta* scale, one forecast row per target.

Transcript protocols (intercept, oracle, surface-feature ridge, embedding ridge) fit
themselves on the task. Judge runs are converted offline from their matched tables with
every variant (pairwise: linear and calibrated; direct: raw and item_z), and each judge run
carries its reference leave-one-out error so stats can pick the best judge without labels.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis import judges
from analysis.baselines import AccuracyOracle, Intercept
from analysis.embeddings import EmbeddingRegression
from analysis.loading import JudgeRun
from capability.evaluate import forecast
from capability.task import Task
from configs.domain import DomainConfig
from protocols import PROTOCOLS, Protocol
from protocols.pairwise.calibrated import JudgeCalibrated

COLUMNS = ["domain", "semester", "protocol", "variant", "judge", "condition", "judge_loo_mse",
           "model", "theta_hat", "theta", "missing"]


def transcript_protocols(dom: DomainConfig, embedding_variants: tuple[str, ...] = ("full", "answer")) -> list[tuple[Protocol, str]]:
    """(protocol, variant) pairs needing no judge: intercept, oracle, glm_regex, and one
    embedding ridge per variant. Fresh instances per domain (the oracle reads that domain's
    frozen tables)."""
    out = [(Intercept(), "-"), (AccuracyOracle(dom), "-"), (PROTOCOLS["glm_regex"], "-")]
    return out + [(EmbeddingRegression(v), v) for v in embedding_variants]


def judge_forecast(run: JudgeRun, task: Task, variant: str) -> tuple[pd.Series, float]:
    """(theta-hat indexed by target, reference leave-one-out MSE) for one judge run under one
    variant. Pairwise "calibrated" shrinks toward JudgeCalibrated's frontier prior for the
    semester, as the protocol does (frontier_prior needs only a `.domain`); "linear" needs
    none. Direct variants are OLS on mean ratings, raw or item-z-scored."""
    refs, targets = task.references.theta, list(task.test.models.index)
    if run.protocol == "pairwise":
        prior = (JudgeCalibrated.frontier_prior(SimpleNamespace(domain=run.harness.domain), task.semester)
                 if variant == "calibrated" else None)
        theta_hat = judges.pairwise_theta(run.table, refs, targets, variant, prior)
        loo = judges.pairwise_reference_loo(run.table, refs, variant)
    else:
        theta_hat = judges.direct_theta(run.table, refs, targets, variant)
        loo = judges.direct_reference_loo(run.table, refs, variant)
    return theta_hat, float(((loo - refs) ** 2).mean())


def forecasts(dom: DomainConfig, task: Task, protocols: list[tuple[Protocol, str]], runs: list[JudgeRun]) -> pd.DataFrame:
    """One row per target x (protocol, variant) x judge x condition, columns COLUMNS.

    judge/condition are "-" and judge_loo_mse NaN for transcript protocols; theta_hat is NaN
    where a protocol has nothing to say; missing is the run's unjudged count on its rows.
    """
    frames = []
    for p, variant in protocols:
        frames.append(forecast(task, p).assign(protocol=p.name, variant=variant, judge="-", condition="-",
                                               judge_loo_mse=np.nan, missing=0))
    for run in runs:
        for variant in judges.PAIRWISE if run.protocol == "pairwise" else judges.DIRECT:
            theta_hat, loo_mse = judge_forecast(run, task, variant)
            frames.append(pd.DataFrame({"semester": task.semester, "model": task.truth.index,
                                        "theta_hat": theta_hat.reindex(task.truth.index).to_numpy(float),
                                        "theta": task.truth.to_numpy()})
                          .assign(protocol=run.protocol, variant=variant, judge=run.judge, condition=run.condition,
                                  judge_loo_mse=loo_mse, missing=run.missing))
    return pd.concat(frames, ignore_index=True).assign(domain=dom.name)[COLUMNS]
```

- [ ] **Step 2: Check on one semester**

Run (real-data prefix): `python -c "from analysis import loading, matching, calibration; d = loading.domain('science'); t = loading.tasks(d, ['2026-I'])[0]; runs = [matching.match(r, t) for r in loading.judge_runs(d, t)]; f = calibration.forecasts(d, t, calibration.transcript_protocols(d), runs); print(f.groupby(['protocol', 'variant', 'judge', 'condition']).agg(n=('theta_hat', 'count'), loo=('judge_loo_mse', 'first')).round(3))"`
Expected: 8 per transcript protocol/variant; 8 per judge run per variant with a finite `loo`.

- [ ] **Step 3: Commit**

```bash
git add src/analysis/calibration.py
git commit -m "analysis: forecasts table over transcript protocols and every judge conversion variant"
```

---

### Task 8: Stats and per-task metrics

**Files:**
- Create: `src/analysis/stats.py`
- Modify: `tests/test_analysis.py` (add three tests)

**Interfaces:**
- Consumes: the forecasts table from Task 7.
- Produces: `BASELINE = "intercept"`, `ORACLE = "accuracy_hard"`, `JUDGED = ("pairwise", "direct")`, `BEST = "best judge on references"`, `MEAN = "mean over judges"`, `KEYS = ["domain", "protocol", "variant", "judge", "condition"]`, `METRICS = ("mse", "delta", "z")`; `mse_row(g)`; `core_condition(f)`; `judge_summaries(f)`; `scores(f)`; `summary(scores)`; `one_per_target(f)`; `paired_metrics(errors)`; `task_metrics(f)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from analysis import stats


def toy_forecasts():
    """Two semesters, two targets each (theta* 1): the intercept forecasts 3 for a and 2 for b (errors 4 and
    1, MSE 2.5), the oracle is exact, glm is off by 1, and a direct judge protocol (variant raw) has two
    judges: "weak" exact with a poor reference LOO, "strong" off by 1 with the better LOO; "strong" also
    has a reasoning-hidden run off by 4."""
    base = dict(domain="d", variant="-", judge="-", condition="-", judge_loo_mse=np.nan, missing=0, theta=1.0)
    rows = []
    for sem in ("s1", "s2"):
        for model, intercept in (("a", 3.0), ("b", 2.0)):
            rows += [dict(base, semester=sem, model=model, protocol=stats.BASELINE, theta_hat=intercept),
                     dict(base, semester=sem, model=model, protocol=stats.ORACLE, theta_hat=1.0),
                     dict(base, semester=sem, model=model, protocol="glm_regex", theta_hat=2.0),
                     dict(base, semester=sem, model=model, protocol="direct", variant="raw", judge="weak",
                          judge_loo_mse=0.9, condition="reasoning-shown", theta_hat=1.0),
                     dict(base, semester=sem, model=model, protocol="direct", variant="raw", judge="strong",
                          judge_loo_mse=0.1, condition="reasoning-shown", theta_hat=2.0),
                     dict(base, semester=sem, model=model, protocol="direct", variant="raw", judge="strong",
                          judge_loo_mse=0.1, condition="reasoning-hidden", theta_hat=5.0)]
    return pd.DataFrame(rows)


def test_scores_pool_and_summary():
    s = stats.scores(toy_forecasts())
    pooled = s[(s.semester == "all") & (s.domain == "d") & (s.judge == "-")].set_index("protocol").mse
    assert pooled[stats.BASELINE] == 2.5 and pooled[stats.ORACLE] == 0.0 and pooled["glm_regex"] == 1.0
    assert s[(s.semester == "s1") & (s.protocol == stats.BASELINE)].n.item() == 2
    r = stats.summary(s)
    # the core rows only: per-condition judge rows would make (protocol, judge) non-unique
    r = r[(r.domain == "d") & r.condition.isin(["-", "core"])].set_index(["protocol", "judge"])
    assert r.skill[("glm_regex", "-")] == 0.6 and r.skill[(stats.ORACLE, "-")] == 1.0
    # the best judge on references is "strong" (lowest LOO), off by 1; the mean over judges is off by 0.5
    assert r.mse[("direct", stats.BEST)] == 1.0 and r.mse[("direct", stats.MEAN)] == 0.5


def test_core_condition_falls_back_to_hidden_reasoning():
    f = toy_forecasts()
    shown = stats.core_condition(f)
    assert set(shown[shown.protocol == "direct"].condition) == {"reasoning-shown"}
    hidden = stats.core_condition(f[f.condition != "reasoning-shown"])
    assert set(hidden[hidden.protocol == "direct"].condition) == {"reasoning-hidden"}


def test_task_metrics_are_paired_against_the_intercept():
    m = stats.task_metrics(toy_forecasts())
    s1 = m[(m.domain == "d") & (m.semester == "s1")].set_index(["protocol", "judge"])
    # glm: squared errors 1, 1 against the intercept's 4, 1 -> delta -1.5; the paired deltas (-3, 0) vary, so z is finite
    assert s1.delta[("glm_regex", "-")] == -1.5 and s1.baseline_mse[("glm_regex", "-")] == 2.5 and np.isfinite(s1.z[("glm_regex", "-")])
    # the intercept against itself: delta 0, no spread -> z NaN
    assert s1.delta[(stats.BASELINE, "-")] == 0.0 and np.isnan(s1.z[(stats.BASELINE, "-")])
    # judge rows enter as the best judge on references only
    assert s1.delta[("direct", stats.BEST)] == -1.5 and (m.judge != stats.MEAN).all()
    pooled = m[(m.domain == "all") & (m.semester == "all") & (m.protocol == "glm_regex")]
    assert pooled.n.item() == 4 and pooled.delta.item() == -1.5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_analysis.py -q`
Expected: the three new tests FAIL (no module `analysis.stats`).

- [ ] **Step 3: Write the module**

```python
"""Stats: MSE of theta-hat against theta* with a bootstrap interval over targets, pooled over
semesters and domains; judge summaries (best judge on references, mean over judges); skill
relative to the intercept and the oracle; and the per-task metrics the figures and tables
use: raw MSE, delta from the intercept, and the paired-bootstrap z of that delta.
"""

import numpy as np
import pandas as pd

KEYS = ["domain", "protocol", "variant", "judge", "condition"]
BASELINE, ORACLE = "intercept", "accuracy_hard"
JUDGED = ("pairwise", "direct")
BEST, MEAN = "best judge on references", "mean over judges"
METRICS = ("mse", "delta", "z")


def mse_row(g: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """n, mse and a percentile-bootstrap 95% interval (lo, hi) of the mean squared error of
    g.theta_hat against g.theta, NaN forecasts dropped. The replication unit is the row
    (one target forecast); callers pool rows only where that is the right unit."""
    err = (g.theta_hat - g.theta).dropna().to_numpy() ** 2
    if len(err) == 0:
        return dict(n=0, mse=np.nan, lo=np.nan, hi=np.nan)
    boot = np.random.default_rng(seed).choice(err, size=(draws, len(err))).mean(axis=1)
    return dict(n=len(err), mse=float(err.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def core_condition(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The judge rows of the core condition: reasoning-shown where a (domain, semester,
    protocol, variant, judge) has it, otherwise reasoning-hidden (semesters whose subjects
    have no reasoning were judged hidden only). Smaller-train-set variants ("-<k>pq") are
    never core. Transcript-protocol rows pass through unchanged."""
    judged = forecasts[forecasts.protocol.isin(JUDGED)]
    key = ["domain", "semester", "protocol", "variant", "judge"]
    shown = judged[judged.condition == "reasoning-shown"]
    has_shown = shown.set_index(key).index.unique()
    hidden = judged[(judged.condition == "reasoning-hidden") & ~judged.set_index(key).index.isin(has_shown)]
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], shown, hidden], ignore_index=True)


def judge_summaries(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Two derived judge rows per (domain, semester, judge protocol, variant), on the core
    condition, condition set to "core": judge BEST = the rows of the judge with the lowest
    reference leave-one-out MSE that semester (known without target labels; a tie keeps one
    judge); judge MEAN = the rows of every judge (pooling equal-sized per-judge row sets
    makes the MSE the mean of the per-judge MSEs)."""
    core = core_condition(forecasts)
    judged = core[core.protocol.isin(JUDGED)]
    key = ["domain", "semester", "protocol", "variant"]
    best = judged.loc[judged.groupby(key).judge_loo_mse.transform("min") == judged.judge_loo_mse]
    best = best.drop_duplicates([*key, "model"])
    return pd.concat([best.assign(judge=BEST), judged.assign(judge=MEAN)], ignore_index=True).assign(condition="core")


def scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per domain x semester x protocol x variant x judge x condition (n, missing,
    mse, lo, hi), plus semester "all" per domain and domain "all" per protocol x variant x
    judge x condition (pooled over every semester of every domain). The judge summary rows
    are included with condition "core"; their interval is NaN because their pooled rows are
    not independent targets (the unit there is the judge)."""
    rows = []
    for f in (forecasts, judge_summaries(forecasts)):
        for k, g in f.groupby(["semester", *KEYS], sort=False):
            rows.append(dict(zip(["semester", *KEYS], k), missing=int(g.missing.max()), **mse_row(g)))
        for k, g in f.groupby(KEYS, sort=False):
            rows.append(dict(zip(KEYS, k), semester="all", missing=int(g.groupby("semester").missing.max().sum()), **mse_row(g)))
        for k, g in f.groupby(KEYS[1:], sort=False):
            rows.append(dict(zip(KEYS[1:], k), domain="all", semester="all",
                             missing=int(g.groupby(["domain", "semester"]).missing.max().sum()), **mse_row(g)))
    table = pd.DataFrame(rows)[["domain", "semester", *KEYS[1:], "n", "missing", "mse", "lo", "hi"]]
    table.loc[table.condition == "core", ["lo", "hi"]] = np.nan
    return table


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Pooled (semester "all") rows with the domain's intercept and oracle MSE alongside and
    skill = (intercept - mse) / (intercept - oracle): 1 at the oracle, 0 at the intercept,
    negative below it. For domain "all" the anchors are the domain-"all" rows."""
    pooled = scores[scores.semester == "all"].copy()
    anchors = pooled[pooled.protocol.isin([BASELINE, ORACLE])].pivot(index="domain", columns="protocol", values="mse")
    pooled["intercept"] = pooled.domain.map(anchors[BASELINE])
    pooled["oracle"] = pooled.domain.map(anchors[ORACLE])
    pooled["skill"] = (pooled.intercept - pooled.mse) / (pooled.intercept - pooled.oracle)
    return pooled


def one_per_target(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The forecast rows with exactly one row per target and condition: transcript protocols
    as they are, judge protocols as the best judge on references (condition "core"). The
    mean over judges is excluded because its rows are not one forecast per target."""
    best = judge_summaries(forecasts)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], best[best.judge == BEST]], ignore_index=True)


def paired_metrics(errors: pd.DataFrame, seed: int = 0, draws: int = 2000) -> dict:
    """From aligned squared errors (columns `err_condition`, `err_baseline`, one row per target):
    n, mse, baseline_mse, delta = mean(condition - baseline), and z = delta / sd of the
    paired bootstrap of delta (targets resampled with replacement; NaN when the paired
    differences do not vary)."""
    c, b = errors.err_condition.to_numpy(float), errors.err_baseline.to_numpy(float)
    if len(c) == 0:
        return dict(n=0, mse=np.nan, baseline_mse=np.nan, delta=np.nan, z=np.nan)
    idx = np.random.default_rng(seed).integers(0, len(c), size=(draws, len(c)))
    sd = float((c[idx] - b[idx]).mean(axis=1).std(ddof=1))
    delta = float((c - b).mean())
    return dict(n=len(c), mse=float(c.mean()), baseline_mse=float(b.mean()), delta=delta, z=delta / sd if sd > 0 else np.nan)


def task_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per (domain, semester) x protocol x variant x judge x condition with the three
    metrics paired against the intercept on the same targets (one_per_target rows only), plus
    semester "all" per domain and domain "all" per condition over the pooled targets.
    Columns: domain, semester, protocol, variant, judge, condition, n, mse, baseline_mse,
    delta, z. A target the condition did not forecast is dropped from both sides."""
    frame = one_per_target(forecasts)
    frame = frame.assign(err_condition=(frame.theta_hat - frame.theta) ** 2)
    base = frame[frame.protocol == BASELINE][["domain", "semester", "model", "err_condition"]].rename(columns={"err_condition": "err_baseline"})
    paired = frame.merge(base, on=["domain", "semester", "model"]).dropna(subset=["err_condition", "err_baseline"])
    rows = []
    for k, g in paired.groupby(["semester", *KEYS], sort=False):
        rows.append(dict(zip(["semester", *KEYS], k), **paired_metrics(g)))
    for k, g in paired.groupby(KEYS, sort=False):
        rows.append(dict(zip(KEYS, k), semester="all", **paired_metrics(g)))
    for k, g in paired.groupby(KEYS[1:], sort=False):
        rows.append(dict(zip(KEYS[1:], k), domain="all", semester="all", **paired_metrics(g)))
    return pd.DataFrame(rows)[["domain", "semester", *KEYS[1:], "n", "mse", "baseline_mse", "delta", "z"]]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_analysis.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/stats.py tests/test_analysis.py
git commit -m "analysis: score tables, judge summaries, and paired per-task metrics against the intercept"
```

---

### Task 9: Figures and tables

**Files:**
- Create: `src/analysis/plots.py`

**Interfaces:**
- Consumes: `forecasts` (Task 7), `task_metrics` output (Task 8), a `variants` mapping protocol → variant (`{"pairwise": "linear", "direct": "raw", "embedding": "full"}`).
- Produces: `CONDITIONS`, `LABELS`; `chosen(metrics, variants)`; `forecast_figure(forecasts, protocol, variant, path, semesters=None)`; `bar_figure(metrics, metric, variants, path)`; `semester_table(metrics, metric, variants, domain=None) -> pd.DataFrame`.

- [ ] **Step 1: Write the module**

```python
"""Figures and tables of the results section.

forecast_figure   one panel per domain, semesters along x; within a semester the targets
                  sorted by theta*, black = true theta, orange = forecast, dotted connectors
bar_figure        four bars (the conditions) at the mean of a per-(domain, semester) metric,
                  jittered circles for the individual (domain, semester) values, the
                  intercept as a dashed line
semester_table    semester x condition for one domain, or pooled over the domains that have
                  the semester, with an "all" row

Judge protocols appear as the best judge on references (task_metrics has no other judge
rows); one variant per protocol is drawn, chosen by the runner's toggles.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import BASELINE, BEST, JUDGED, METRICS, one_per_target

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "savefig.dpi": 300,
})
CONDITIONS = ["glm_regex", "embedding", "direct", "pairwise"]  # the four bars, in this order
LABELS = {BASELINE: "intercept", "glm_regex": "length+regex", "embedding": "embedding", "direct": "direct judge",
          "pairwise": "pairwise judge"}
COLORS = dict(zip(CONDITIONS, ["#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6"]))
TRUE, PRED = "black", "#eb6834"
METRIC_LABELS = {"mse": "MSE (θ units)", "delta": "MSE − intercept MSE", "z": "z of MSE − intercept (paired bootstrap)"}


def chosen(metrics: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """The metric rows of the chosen variant per protocol ("-" for protocols without one)."""
    return metrics[[variants.get(p, "-") == v for p, v in zip(metrics.protocol, metrics.variant)]]


def offset_spines(ax) -> None:
    """Nature-panel axes: left and bottom spines 3pt outward."""
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))


def forecast_figure(forecasts: pd.DataFrame, protocol: str, variant: str, path, semesters: list[str] | None = None) -> None:
    """True theta (black) and the condition's forecast (orange) for every target: one panel per
    domain (panel width follows its number of semesters), semesters along x, targets within
    a semester sorted by theta*, a dotted connector per target. `semesters` restricts the
    x-axis (e.g. the latest one). Judge protocols are drawn as the best judge on references."""
    f = one_per_target(forecasts)
    f = f[(f.protocol == protocol) & (f.variant == variant)]
    if semesters:
        f = f[f.semester.isin(semesters)]
    domains = list(dict.fromkeys(f.domain))
    widths = [f[f.domain == d].semester.nunique() for d in domains]
    fig, axes = plt.subplots(1, len(domains), figsize=(0.9 * sum(widths) + 0.6 * len(domains), 2.2), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, dom in zip(np.atleast_1d(axes), domains):
        d = f[f.domain == dom]
        sems = sorted(dict.fromkeys(d.semester))  # "2024-I" < "2024-II" < "2025-I": chronological
        for i, sem in enumerate(sems):
            g = d[d.semester == sem].sort_values("theta")
            x = i + np.linspace(-0.35, 0.35, len(g))  # targets spread within the semester's slot
            for xi, t, p in zip(x, g.theta, g.theta_hat):
                ax.plot([xi, xi], [t, p], color="grey", lw=0.6, ls=":")
            ax.plot(x, g.theta, "o", color=TRUE, ms=3)
            ax.plot(x, g.theta_hat, "o", color=PRED, ms=3)
        ax.set_xticks(range(len(sems)), sems, rotation=90 if len(sems) > 3 else 0)
        ax.text(0, 1.02, dom, transform=ax.transAxes, color="grey", ha="left")
        offset_spines(ax)
    np.atleast_1d(axes)[0].set_ylabel("θ")
    fig.legend(handles=[plt.Line2D([], [], color=TRUE, marker="o", ms=3, ls="", label="true θ"),
                        plt.Line2D([], [], color=PRED, marker="o", ms=3, ls="", label=f"forecast ({LABELS.get(protocol, protocol)})")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, seed: int = 0) -> None:
    """Four bars, one per condition in CONDITIONS, at the mean over (domain, semester) of the
    metric ("mse", "delta" or "z"); the individual (domain, semester) values as jittered
    circles; the intercept as a dashed grey line (its mean raw MSE, or 0 for delta and z)."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[(m.semester != "all") & (m.domain != "all")]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    for i, p in enumerate(CONDITIONS):
        values = m[m.protocol == p][metric].dropna().to_numpy()
        if len(values) == 0:
            continue
        ax.bar(i, values.mean(), color=COLORS[p], width=0.7)
        ax.plot(i + rng.uniform(-0.2, 0.2, len(values)), values, "o", color="black", ms=2, alpha=0.6, mew=0)
    baseline = m[m.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    ax.axhline(baseline, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    offset_spines(ax)
    # constrained layout reserves room for an "outside" legend, in both the one- and two-panel branches
    fig.legend(handles=[plt.Line2D([], [], color="grey", lw=0.8, ls="--", label="intercept"),
                        plt.Line2D([], [], color="black", marker="o", ms=2, ls="", label="one domain × semester")],
               loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def semester_table(metrics: pd.DataFrame, metric: str, variants: dict[str, str], domain: str | None = None) -> pd.DataFrame:
    """Semester x condition (intercept first, then CONDITIONS) for one domain, or pooled when
    `domain` is None: the mean over the domains that have that semester, plus an "all" row
    (the domain's pooled row, or the mean over every (domain, semester) cell). Columns are
    the LABELS; a condition without a value is NaN."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[m.domain == domain] if domain else m[(m.domain != "all") & (m.semester != "all")]
    per_task = m[m.semester != "all"].pivot_table(index="semester", columns="protocol", values=metric, aggfunc="mean").sort_index()
    if domain:
        pooled = m[m.semester == "all"].set_index("protocol")[metric]
    else:
        pooled = m.groupby("protocol")[metric].mean()
    table = pd.concat([per_task, pooled.to_frame("all").T])
    columns = [p for p in [BASELINE, *CONDITIONS] if p in table.columns]
    return table[columns].rename(columns=LABELS)
```

- [ ] **Step 2: Render on real outputs and look at the PNGs**

After Task 10 has produced `analysis/results/{forecasts,metrics}.csv`:

`PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; f = pd.read_csv('analysis/results/forecasts.csv'); m = pd.read_csv('analysis/results/metrics.csv'); v = {'pairwise': 'linear', 'direct': 'raw', 'embedding': 'full'}; plots.forecast_figure(f, 'pairwise', 'linear', 'analysis/results/forecasts_pairwise.png'); plots.bar_figure(m, 'delta', v, 'analysis/results/bars_delta.png'); print(plots.semester_table(m, 'mse', v).round(2))"`

Open both PNGs (Read tool). Check: no overlapping tick labels, legend outside the panels, nothing clipped, the forecast figure reads like the reference figure (black true, orange forecast, dotted connectors, targets sorted within each semester).

- [ ] **Step 3: Commit**

```bash
git add src/analysis/plots.py
git commit -m "analysis: forecast dot plot, condition bar plot and semester tables"
```

---

### Task 10: The cell runner

**Files:**
- Create (replace the draft): `scripts/analyses/results.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `analysis/results/{forecasts,scores,summary,metrics}.csv`, `tables/<metric>_<domain|pooled>.csv`, `forecasts_<condition>.svg`, `forecasts_<condition>_<semester>.svg`, `bars_<metric>.svg`.

- [ ] **Step 1: Write the runner**

```python
"""Results of the forecasting task: the five conditions against the intercept and the oracle.

Cell script: run top to bottom as `PYTHONPATH=src uv run python scripts/analyses/results.py`,
or cell by cell (`# %%`) in PyCharm / VS Code after the cells above it. State lives in the
module-level names each cell defines. Never launches judging: judge runs are read from logs/
and what a campaign still lacks is reported in the `missing` column.

Transcript logs and embeddings are read from the T7 volume unless SMALL_JUDGE_DATA and
SMALL_JUDGE_EMBEDDINGS say otherwise; the frozen tables always come from data/meta.

Design: docs/superpowers/specs/2026-09-15-results-analysis-design.md
"""

# %% settings
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # every path in src/ is repo-relative
os.environ.setdefault("SMALL_JUDGE_DATA", "/Volumes/T7/small-judge-remote/data")
os.environ.setdefault("SMALL_JUDGE_EMBEDDINGS", "/Volumes/T7/small-judge-remote/transcript-embeddings")

from analysis import calibration, loading, matching, plots, stats  # noqa: E402  (after the env is set)

DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = None  # e.g. ["2026-I"]; None = every frozen semester task
OUT = Path("analysis/results")
# Every variant is computed and scored; these pick the ones the figures and tables show.
# Defaults chosen by the reference leave-one-out error (mean over judge runs, four domains):
# pairwise calibrated 0.34 vs linear 0.45; direct raw 0.43 vs item_z 0.59. Never by target MSE.
PAIRWISE_CONVERSION = "calibrated"  # or "linear"
DIRECT_NORMALISATION = "raw"  # or "item_z"
EMBEDDING_VARIANT = "full"  # or "answer"
VARIANTS = {"pairwise": PAIRWISE_CONVERSION, "direct": DIRECT_NORMALISATION, "embedding": EMBEDDING_VARIANT}
(OUT / "tables").mkdir(parents=True, exist_ok=True)

# %% loading: semester tasks and the judge runs on disk
# domains: name -> DomainConfig; tasks: (domain name, semester) -> Task; runs: same key -> [JudgeRun]
domains = {name: loading.domain(name) for name in DOMAINS}
tasks = {(name, t.semester): t for name, dom in domains.items() for t in loading.tasks(dom, SEMESTERS)}
runs = {key: loading.judge_runs(domains[key[0]], task) for key, task in tasks.items()}
print({key: [(r.protocol, r.judge, r.condition) for r in rs] for key, rs in runs.items()})

# %% matching: restrict judge rows to the frozen train items, count the unjudged
runs = {key: [matching.match(r, tasks[key]) for r in rs] for key, rs in runs.items()}
print(pd.DataFrame([(k[0], k[1], r.protocol, r.judge, r.condition, len(r.table), r.missing) for k, rs in runs.items() for r in rs],
                   columns=["domain", "semester", "protocol", "judge", "condition", "rows", "missing"]).to_string(index=False))

# %% calibration: every signal -> theta-hat; one row per target x protocol x variant x judge x condition
protocols = {name: calibration.transcript_protocols(dom) for name, dom in domains.items()}
forecasts = pd.concat([calibration.forecasts(domains[name], task, protocols[name], runs[(name, sem)])
                       for (name, sem), task in tasks.items()], ignore_index=True)
forecasts.to_csv(OUT / "forecasts.csv", index=False, float_format="%.4f")

# %% stats: score tables, judge summaries, skill, and the paired per-task metrics
scores = stats.scores(forecasts)
summary = stats.summary(scores)
metrics = stats.task_metrics(forecasts)
scores.to_csv(OUT / "scores.csv", index=False, float_format="%.4f")
summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
metrics.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
core = summary[summary.judge.isin(["-", stats.BEST, stats.MEAN])]
print(core[["domain", "protocol", "variant", "judge", "n", "missing", "mse", "intercept", "oracle", "skill"]].round(3).to_string(index=False))

# %% tables: semester x condition, every metric, per domain and pooled
for metric in stats.METRICS:
    for domain in [None, *DOMAINS]:
        table = plots.semester_table(metrics, metric, VARIANTS, domain)
        table.to_csv(OUT / "tables" / f"{metric}_{domain or 'pooled'}.csv", float_format="%.3f")
print(plots.semester_table(metrics, "mse", VARIANTS).round(3).to_string())

# %% figures: forecasts of the best condition (all semesters, and the latest), and the bars
pooled = plots.chosen(metrics, VARIANTS)
pooled = pooled[(pooled.domain == "all") & (pooled.semester == "all") & pooled.protocol.isin(plots.CONDITIONS)]
best = pooled.sort_values("mse").iloc[0]
latest = max(sem for _, sem in tasks)
plots.forecast_figure(forecasts, best.protocol, best.variant, OUT / f"forecasts_{best.protocol}.svg")
plots.forecast_figure(forecasts, best.protocol, best.variant, OUT / f"forecasts_{best.protocol}_{latest}.svg", semesters=[latest])
for metric in stats.METRICS:
    plots.bar_figure(metrics, metric, VARIANTS, OUT / f"bars_{metric}.svg")
print("best condition:", best.protocol, best.variant, "pooled mse", round(best.mse, 3))
print("wrote", sorted(p.name for p in OUT.iterdir()))
```

- [ ] **Step 2: Run end to end on science**

Run: `PYTHONPATH=src uv run python -c "exec(open('scripts/analyses/results.py').read().replace('DOMAINS = [\"science\", \"math\", \"coding\", \"cyber\"]', 'DOMAINS = [\"science\"]'))" 2>&1 | grep -v "^\[WARNING\]"`
Expected: `forecasts.csv`, `scores.csv`, `summary.csv`, `metrics.csv`, `tables/` with 3 metrics × (pooled + science), two forecast SVGs and three bar SVGs in `analysis/results/`; the printed pooled table has the five condition columns and rows 2024-I … 2026-I plus "all".

- [ ] **Step 3: Run Task 9 Step 2 and inspect the figures**

- [ ] **Step 4: Run every domain**

Run: `PYTHONPATH=src uv run python scripts/analyses/results.py 2>&1 | grep -v "^\[WARNING\]"`
Note any domain where an lfs or embedding warning skipped a run, and the `missing` totals.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyses/results.py
git commit -m "analysis: cell runner for the results tables and figures"
```

---

### Task 11: Broken y-axis for off-axis points in the bar plot; axes touching

**Files:**
- Modify: `src/analysis/plots.py` (`bar_figure`; add `outlier_split` and `zigzag`; remove `offset_spines` everywhere)

**Interfaces:**
- Produces: `outlier_split(values) -> tuple[np.ndarray, np.ndarray]` (inliers, outliers); `zigzag(ax, y, transform)` draws a break mark; `bar_figure` signature unchanged.

Also, on every figure the axes must touch: delete `offset_spines` and its three call sites (`forecast_figure`, `bar_figure`) so the left and bottom spines meet at the corner; the user's standing figure spec now says no outward spine offset.

When one (domain, semester) value is far above the others (e.g. a ridge fit that blew up), a single axis squashes every bar to the baseline. Split the y-axis instead: the lower panel covers the inlier range, a short upper panel covers the outliers, and zigzag marks on the left spine show the break. Bars and points are drawn on both panels; each panel's limits clip them.

- [ ] **Step 1: Replace `bar_figure` in `src/analysis/plots.py` with this, and add the two helpers above it**

```python
def outlier_split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inliers, outliers) of a 1-D array: an outlier lies beyond 3 interquartile ranges from
    the quartiles (Tukey's far-out fence). With fewer than 4 values nothing is an outlier."""
    values = np.asarray(values, float)
    if len(values) < 4:
        return values, values[:0]
    q1, q3 = np.percentile(values, [25, 75])
    fence = 3 * (q3 - q1)
    far = (values > q3 + fence) | (values < q1 - fence)
    return values[~far], values[far]


def zigzag(ax, y: float, size: float = 0.02) -> None:
    """A small zigzag across the left spine at axes-fraction height `y`, marking an axis break."""
    x = np.array([-1, -0.5, 0.5, -0.5, 0.5, 1]) * size
    yy = y + np.linspace(-1.5, 1.5, len(x)) * size
    ax.plot(x, yy, color="black", lw=0.8, clip_on=False, transform=ax.transAxes, solid_capstyle="round")


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, seed: int = 0) -> None:
    """Four bars, one per condition in CONDITIONS, at the mean over (domain, semester) of the
    metric ("mse", "delta" or "z"); the individual (domain, semester) values as jittered
    circles; the intercept as a dashed grey line (its mean raw MSE, or 0 for delta and z).
    Values beyond Tukey's far-out fence (outlier_split) go to a short upper panel above a
    zigzag axis break, so one blown-up fit cannot squash the others."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[(m.semester != "all") & (m.domain != "all")]
    rng = np.random.default_rng(seed)
    points = {p: m[m.protocol == p][metric].dropna().to_numpy() for p in CONDITIONS}
    every = np.concatenate([v for v in points.values() if len(v)]) if any(len(v) for v in points.values()) else np.array([])
    baseline = m[m.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    inliers, outliers = outlier_split(every)
    if len(outliers):
        fig, (top, ax) = plt.subplots(2, 1, figsize=(2.6, 2.6), sharex=True, layout="constrained",
                                      gridspec_kw={"height_ratios": [1, 3]})
        panels = [top, ax]
    else:
        fig, ax = plt.subplots(figsize=(2.6, 2.2), layout="constrained")
        top, panels = None, [ax]
    for panel in panels:
        for i, p in enumerate(CONDITIONS):
            values = points[p]
            if len(values) == 0:
                continue
            panel.bar(i, values.mean(), color=COLORS[p], width=0.7)
            panel.plot(i + rng.uniform(-0.2, 0.2, len(values)), values, "o", color="black", ms=2, alpha=0.6, mew=0)
        panel.axhline(baseline, color="grey", lw=0.8, ls="--")
    if top is not None:
        pad = 0.1 * max(np.ptp(inliers), 1e-9)  # np.ptp: the ndarray method is gone in NumPy 2.x
        ax.set_ylim(min(inliers.min(), baseline) - pad, max(inliers.max(), baseline) + pad)
        span = max(np.ptp(outliers), pad)
        top.set_ylim(outliers.min() - 0.5 * span, outliers.max() + 0.5 * span)
        top.spines["bottom"].set_visible(False)
        top.tick_params(axis="x", bottom=False)
        zigzag(ax, 1.0)
        zigzag(top, 0.0)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    (top or ax).set_ylabel(METRIC_LABELS[metric]) if top is None else fig.supylabel(METRIC_LABELS[metric], fontsize=8)
    # constrained layout reserves room for an "outside" legend, in both the one- and two-panel branches
    fig.legend(handles=[plt.Line2D([], [], color="grey", lw=0.8, ls="--", label="intercept"),
                        plt.Line2D([], [], color="black", marker="o", ms=2, ls="", label="one domain × semester")],
               loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Render and look**

Run: `PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; m = pd.read_csv('analysis/results/metrics.csv'); v = {'pairwise': 'linear', 'direct': 'raw', 'embedding': 'full'}; [plots.bar_figure(m, k, v, f'analysis/results/bars_{k}.png') for k in ('mse', 'delta', 'z')]"`
Open the three PNGs (Read tool). On `delta` and `mse` the blown-up length+regex value should sit alone in the short upper panel with zigzag marks on the left spine at the break; the lower panel's bars must be readable. `z` has no far outlier and must render as a single panel. Check labels and legend for overlap.

- [ ] **Step 3: Commit**

```bash
git add src/analysis/plots.py
git commit -m "analysis: zigzag axis break for off-axis bar-plot points; axes touching"
```

---

### Task 12: Rename the baseline to easy-item prediction

**Files:**
- Modify: `src/analysis/baselines.py`, `src/analysis/calibration.py`, `src/analysis/stats.py`, `src/analysis/plots.py`, `scripts/analyses/results.py`, `tests/test_analysis.py`

The user's name for the baseline is "easy-item prediction": what one can predict from accuracy on the easy (saturated) items. "Intercept" was the implementer's arithmetic and must not appear in any output, label, name or docstring.

- [ ] **Step 1: Apply the rename**

Exact substitutions, everywhere in the six files (use a script; then grep to confirm no `ntercept` remains in `src/analysis`, `scripts/analyses/results.py` or `tests/test_analysis.py`):

| old | new |
| --- | --- |
| class `Intercept` | class `EasyItemPrediction` |
| protocol name `"intercept"` (the `name = ...` and `BASELINE = "intercept"`) | `"easy_items"` |
| `from analysis.baselines import AccuracyOracle, Intercept` | `from analysis.baselines import AccuracyOracle, EasyItemPrediction` |
| `Intercept()` | `EasyItemPrediction()` |
| `LABELS[BASELINE]` value `"intercept"` | `"easy-item prediction"` |
| `METRIC_LABELS`: `"MSE − intercept MSE"` / `"z of MSE − intercept (paired bootstrap)"` | `"MSE − easy-item MSE"` / `"z of MSE − easy-item MSE (paired bootstrap)"` |
| legend entries `label="intercept"` | `label="easy-item prediction"` |
| `summary()` columns `intercept` (and the `pooled["intercept"] = ...` line, the `skill` formula, the docstring) | `baseline` |
| runner print column `"intercept"` | `"baseline"` |
| docstrings and comments that say "intercept" as the condition's name (`baselines.py` module and class docstrings, `stats.py` module docstring, `summary` docstring, `plots.py` module docstring and `bar_figure` docstring, `task_metrics` docstring, the runner's docstring and cell comments) | "easy-item prediction" (where the arithmetic is meant, keep a phrase like "the references' mean θ*") |
| tests: `Intercept` import/use, `stats.BASELINE` stays as the symbol | unchanged symbol, new class name |

`stats.BASELINE` remains the symbol name; only its value changes. `ORACLE` is unchanged.

- [ ] **Step 2: Tests and a render**

Run: `uv run pytest tests/test_analysis.py -q` — expect 6 passed. Then `PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; m = pd.read_csv('analysis/results/metrics.csv'); v = {'pairwise': 'calibrated', 'direct': 'raw', 'embedding': 'full'}; plots.bar_figure(m, 'delta', v, 'analysis/results/bars_delta_check.png'); print(plots.semester_table(m, 'mse', v).columns.tolist())"` — the printed columns start with `easy-item prediction`, the legend reads "easy-item prediction". Note `metrics.csv` still holds the old protocol string `intercept` in its `protocol` column until the runner is re-run; for this check only, load it with `.replace({'protocol': {'intercept': 'easy_items'}})` and say so in the report.

- [ ] **Step 3: Commit**

```bash
git add src/analysis/baselines.py src/analysis/calibration.py src/analysis/stats.py src/analysis/plots.py scripts/analyses/results.py tests/test_analysis.py
git commit -m "analysis: the baseline is easy-item prediction, not intercept"
```

---

### Task 13: Easy-item prediction in the forecast dot plots

**Files:**
- Modify: `src/analysis/plots.py` (`forecast_figure` only)

The forecast dot plots must show the baseline too: the easy-item prediction is one value per (domain, semester) (the references' mean θ\*), drawn as a short grey dashed segment spanning the semester's slot, and named in the legend.

- [ ] **Step 1: Change `forecast_figure`**

Signature unchanged. After `f = one_per_target(forecasts)` add

```python
    baseline = f[f.protocol == BASELINE].groupby(["domain", "semester"]).theta_hat.first()  # one value per task
```

before `f = f[(f.protocol == protocol) & (f.variant == variant)]`. Inside the per-semester loop, after the two `ax.plot(x, ...)` calls, add

```python
            if (dom, sem) in baseline.index:  # the easy-item prediction: the same value for every target of the task
                ax.plot([i - 0.42, i + 0.42], [baseline[(dom, sem)]] * 2, color="grey", lw=0.8, ls="--")
```

and extend the legend handles with

```python
                        plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE]),
```

as a third entry (`ncol=3`). `LABELS[BASELINE]` reads "easy-item prediction" after Task 12.

- [ ] **Step 2: Render and look**

Run: `PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; f = pd.read_csv('analysis/results/forecasts.csv'); plots.forecast_figure(f, 'pairwise', 'calibrated', 'analysis/results/forecasts_check.png')"` (if `forecasts.csv` still carries the old protocol string, load it with `.replace({'protocol': {'intercept': 'easy_items'}})` and say so). Open the PNG: a grey dashed level per semester slot, the legend with three entries outside the panels, nothing overlapping.

- [ ] **Step 3: Commit**

```bash
git add src/analysis/plots.py
git commit -m "analysis: forecast dot plots show the easy-item prediction level per semester"
```

---

### Task 14: Judge ensemble conditions

**Files:**
- Modify: `src/analysis/calibration.py` (add `ENSEMBLE`, `core_runs`, `ensemble_runs`; extend `forecasts`)
- Modify: `src/analysis/plots.py` (`CONDITIONS`, `LABELS`, `COLORS`, bar figure width)
- Modify: `tests/test_analysis.py` (one test)

The ensemble pools the three judges of a semester before conversion, per protocol: the pairwise verdict tables are concatenated (every judge's verdict is one more observation of the pair, so the per-bench calibration and the win rates see all of them), the direct rating tables are concatenated (a transcript rated by several judges gets their mean, which `valid_ratings` already does). Each pooled table goes through the same conversions and appears as protocol `pairwise_ensemble` / `direct_ensemble`, judge `ensemble`, condition `core`, with the same variants. Best-judge and mean-over-judges summaries ignore them (`stats.JUDGED` is unchanged), so they behave like transcript conditions in the metrics, bars and tables.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
from types import SimpleNamespace

from analysis import calibration
from analysis.loading import JudgeRun


def test_ensemble_runs_pool_the_core_condition_of_every_judge():
    def run(protocol, judge, condition, n, missing=0):
        table = pd.DataFrame({"bench": "b", "item": [str(i) for i in range(n)], "judge": judge})
        return JudgeRun(protocol, judge, condition, SimpleNamespace(domain=None), table, missing)
    runs = [run("pairwise", "j1", "reasoning-shown", 3, 1), run("pairwise", "j1", "reasoning-hidden", 9),
            run("pairwise", "j2", "reasoning-hidden", 4, 2), run("pairwise", "j3", "reasoning-shown-10pq", 7),
            run("direct", "j1", "reasoning-shown", 5)]
    pooled = calibration.ensemble_runs(runs)
    # pairwise: j1's shown run and j2's hidden run pooled; the 10pq run is never core; direct has one judge only
    assert [(r.protocol, r.judge, r.condition, len(r.table), r.missing) for r in pooled] == [("pairwise", "ensemble", "core", 7, 3)]
    assert sorted(pooled[0].table.judge.unique()) == ["j1", "j2"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_analysis.py -q` — the new test FAILS (no `ensemble_runs`).

- [ ] **Step 3: Extend `src/analysis/calibration.py`**

Add after `COLUMNS`:

```python
ENSEMBLE = "ensemble"  # the judge label of a pooled run
CORE = ("reasoning-shown", "reasoning-hidden")


def core_runs(runs: list[JudgeRun]) -> list[JudgeRun]:
    """One run per (protocol, judge): its reasoning-shown run where judged, else its
    reasoning-hidden run; smaller-train-set runs ("-<k>pq") are never core."""
    out: dict[tuple[str, str], JudgeRun] = {}
    for run in runs:
        if run.condition not in CORE:
            continue
        key = (run.protocol, run.judge)
        if key not in out or run.condition == "reasoning-shown":
            out[key] = run
    return list(out.values())


def ensemble_runs(runs: list[JudgeRun]) -> list[JudgeRun]:
    """Per protocol, one pooled JudgeRun (judge ENSEMBLE, condition "core") over the core runs
    of at least two judges: tables concatenated (pairwise: every judge's verdict is one more
    observation of the pair; direct: a transcript's ratings from several judges average in
    valid_ratings), `missing` summed, `harness` the first member's. Empty when no protocol
    has two judges."""
    out = []
    core = core_runs(runs)
    for protocol in ("pairwise", "direct"):
        members = [r for r in core if r.protocol == protocol]
        if len(members) < 2:
            continue
        table = pd.concat([r.table for r in members], ignore_index=True)
        out.append(JudgeRun(protocol, ENSEMBLE, "core", members[0].harness, table, sum(r.missing for r in members)))
    return out
```

In `forecasts`, replace the judge loop header and the protocol label so the ensemble rows carry their own protocol name:

```python
    for run in runs + ensemble_runs(runs):
        label = f"{run.protocol}_{ENSEMBLE}" if run.judge == ENSEMBLE else run.protocol
        for variant in judges.PAIRWISE if run.protocol == "pairwise" else judges.DIRECT:
            theta_hat, loo_mse = judge_forecast(run, task, variant)
            frames.append(pd.DataFrame({"semester": task.semester, "model": task.truth.index,
                                        "theta_hat": theta_hat.reindex(task.truth.index).to_numpy(float),
                                        "theta": task.truth.to_numpy()})
                          .assign(protocol=label, variant=variant, judge=run.judge, condition=run.condition,
                                  judge_loo_mse=loo_mse, missing=run.missing))
```

and extend the `forecasts` docstring with: "Pooled ensemble runs (ensemble_runs) are added with protocol `<protocol>_ensemble`, judge `ensemble`, condition `core`."

- [ ] **Step 4: Extend `src/analysis/plots.py`**

```python
CONDITIONS = ["glm_regex", "embedding", "direct", "direct_ensemble", "pairwise", "pairwise_ensemble"]  # the bars, in this order
LABELS = {BASELINE: "easy-item prediction", "glm_regex": "length+regex", "embedding": "embedding",
          "direct": "direct judge", "direct_ensemble": "direct judges, pooled",
          "pairwise": "pairwise judge", "pairwise_ensemble": "pairwise judges, pooled"}
COLORS = dict(zip(CONDITIONS, ["#1baf7a", "#4a3aa7", "#eb6834", "#f4a581", "#2a78d6", "#8fb8e8"]))
```

(`LABELS[BASELINE]` is whatever Task 12 set; keep it.) In `bar_figure`, widen both figures from 2.6 to 3.4 in so six rotated labels fit. The runner's `VARIANTS` mapping must also cover the ensembles: in `scripts/analyses/results.py` set `VARIANTS = {"pairwise": PAIRWISE_CONVERSION, "pairwise_ensemble": PAIRWISE_CONVERSION, "direct": DIRECT_NORMALISATION, "direct_ensemble": DIRECT_NORMALISATION, "embedding": EMBEDDING_VARIANT}` (add `scripts/analyses/results.py` to the commit).

- [ ] **Step 5: Tests, one-semester check, commit**

Run: `uv run pytest tests/test_analysis.py -q` — 7 passed. Then (real-data prefix) `python -c "from analysis import loading, matching, calibration; d = loading.domain('science'); t = loading.tasks(d, ['2026-I'])[0]; runs = [matching.match(r, t) for r in loading.judge_runs(d, t)]; f = calibration.forecasts(d, t, calibration.transcript_protocols(d), runs); print(f[f.judge == 'ensemble'].groupby(['protocol', 'variant']).agg(n=('theta_hat', 'count'), loo=('judge_loo_mse', 'first'), missing=('missing', 'first')).round(3))"` — rows for `direct_ensemble` and `pairwise_ensemble`, both variants, n 8.

```bash
git add src/analysis/calibration.py src/analysis/plots.py scripts/analyses/results.py tests/test_analysis.py
git commit -m "analysis: judge ensemble conditions pool the three judges before conversion"
```

---

### Task 15: Point styling, 95% intervals, single-semester bar plots

**Files:**
- Modify: `src/analysis/plots.py` (`bar_figure`, `forecast_figure`; add `swarm_offsets`, `ci95`, `DOMAIN_COLORS`)
- Modify: `scripts/analyses/results.py` (bars per semester)

The user's verdict on the first renders: the scattered points look terrible. New rules for both figures, then the bars get a 95% interval and per-semester versions.

Bar plot: bars are neutral (`#d9d7d0`), colour carries the domain only (`DOMAIN_COLORS`); the mean is a black line across the bar with a capless black error bar for the t-based 95% interval of the mean over the (domain, semester) points (the replication unit; no interval below 3 points); the points are a deterministic beeswarm (no random jitter), one colour per domain, white edge, drawn above the bar; the easy-item prediction is the dashed grey line. `semester=None` draws every task; `semester="2026-I"` draws that semester only (points = domains). The zigzag axis break from Task 11 stays.

Forecast plot: true θ as open black circles, forecasts as filled orange circles with a white edge, the connector a thin grey vertical line, the easy-item level a dashed grey segment per semester (Task 13), constrained layout, legend outside.

- [ ] **Step 1: Add the helpers to `src/analysis/plots.py`** (after `COLORS`)

```python
from scipy import stats as scipy_stats

DOMAIN_COLORS = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#eda100"}
BAR = "#d9d7d0"  # neutral bar; colour is reserved for the domain of a point


def swarm_offsets(values: np.ndarray, width: float = 0.3, min_gap: float | None = None) -> np.ndarray:
    """Deterministic beeswarm x-offsets in [-width, width] for 1-D `values`: points are placed in
    value order, each at the offset nearest the centre that keeps it at least `min_gap` (in the
    values' units; default 1/25 of their range) from every already-placed point closer than
    that vertically. No random jitter, so a figure is reproducible."""
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        return values
    gap = min_gap if min_gap is not None else max(np.ptp(values), 1e-9) / 25
    slots = np.linspace(-width, width, 2 * n + 1)
    slots = slots[np.argsort(np.abs(slots), kind="stable")]  # centre first, then outward
    offsets = np.zeros(n)
    placed: list[tuple[float, float]] = []  # (value, offset)
    for i in np.argsort(values, kind="stable"):
        for s in slots:
            if all(abs(values[i] - v) >= gap or abs(s - o) >= width / max(n, 1) * 2 for v, o in placed):
                offsets[i] = s
                break
        else:
            offsets[i] = slots[-1]
        placed.append((values[i], offsets[i]))
    return offsets


def ci95(values: np.ndarray) -> float:
    """Half-width of the t-based 95% interval of the mean of `values`; NaN with fewer than 3."""
    n = len(values)
    if n < 3:
        return float("nan")
    return float(scipy_stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
```

- [ ] **Step 2: Replace `bar_figure`**

Keep `outlier_split` and `zigzag` from Task 11. New signature `bar_figure(metrics, metric, variants, path, semester=None)`:

```python
def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None) -> None:
    """Bars per condition in CONDITIONS (neutral fill) at the mean over (domain, semester) of the
    metric ("mse", "delta" or "z"), a black line at the mean with a capless 95% t-interval over
    those points (none below 3 points), the points as a deterministic beeswarm coloured by
    domain, the easy-item prediction as a dashed grey line (its mean raw MSE, or 0 for delta
    and z). `semester` restricts to one semester (points are then the domains). Values beyond
    Tukey's far-out fence go to a short upper panel above a zigzag axis break."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[(m.semester != "all") & (m.domain != "all")]
    if semester:
        m = m[m.semester == semester]
    groups = {p: m[m.protocol == p].dropna(subset=[metric]) for p in CONDITIONS}
    every = np.concatenate([g[metric].to_numpy() for g in groups.values() if len(g)]) if any(len(g) for g in groups.values()) else np.array([])
    baseline = m[m.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    inliers, outliers = outlier_split(every)
    if len(outliers):
        fig, (top, ax) = plt.subplots(2, 1, figsize=(3.4, 2.8), sharex=True, layout="constrained",
                                      gridspec_kw={"height_ratios": [1, 3]})
        panels = [top, ax]
    else:
        fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
        top, panels = None, [ax]
    for panel in panels:
        for i, p in enumerate(CONDITIONS):
            g = groups[p]
            if g.empty:
                continue
            values = g[metric].to_numpy()
            panel.bar(i, values.mean(), color=BAR, width=0.72, zorder=1)
            panel.hlines(values.mean(), i - 0.36, i + 0.36, color="black", lw=1.0, zorder=4)
            if np.isfinite(half := ci95(values)):
                panel.errorbar(i, values.mean(), yerr=half, color="black", lw=0.8, capsize=0, zorder=4)
            panel.scatter(i + swarm_offsets(values), values, s=14, c=[DOMAIN_COLORS.get(d, "black") for d in g.domain],
                          edgecolor="white", linewidth=0.5, zorder=3)
        panel.axhline(baseline, color="grey", lw=0.8, ls="--", zorder=0)
    if top is not None:
        pad = 0.1 * max(np.ptp(inliers), 1e-9)
        ax.set_ylim(min(inliers.min(), baseline) - pad, max(inliers.max(), baseline) + pad)
        span = max(np.ptp(outliers), pad)
        top.set_ylim(outliers.min() - 0.5 * span, outliers.max() + 0.5 * span)
        top.spines["bottom"].set_visible(False)
        top.tick_params(axis="x", bottom=False)
        zigzag(ax, 1.0)
        zigzag(top, 0.0)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    if top is None:
        ax.set_ylabel(METRIC_LABELS[metric])
    else:
        fig.supylabel(METRIC_LABELS[metric], fontsize=8)
    domains = [d for d in DOMAIN_COLORS if d in set(m.domain)]
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", color=DOMAIN_COLORS[d], markeredgecolor="white", label=d) for d in domains]
    handles += [plt.Line2D([], [], color="black", lw=1, label="mean ± 95% CI over tasks"),
                plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Restyle `forecast_figure`**

Same signature. Replace the drawing inside the per-semester loop and the figure/legend setup with:

```python
    fig, axes = plt.subplots(1, len(domains), figsize=(0.95 * sum(widths) + 0.5 * len(domains), 2.1), sharey=True,
                             gridspec_kw={"width_ratios": widths}, layout="constrained")
    for ax, dom in zip(np.atleast_1d(axes), domains):
        d = f[f.domain == dom]
        sems = sorted(dict.fromkeys(d.semester))  # "2024-I" < "2024-II" < "2025-I": chronological
        for i, sem in enumerate(sems):
            g = d[d.semester == sem].sort_values("theta")
            x = i + np.linspace(-0.36, 0.36, len(g))  # targets spread within the semester's slot
            ax.vlines(x, np.minimum(g.theta, g.theta_hat), np.maximum(g.theta, g.theta_hat), color="#b0aea6", lw=0.6, zorder=1)
            if (dom, sem) in baseline.index:  # the easy-item prediction: one level per task
                ax.hlines(baseline[(dom, sem)], i - 0.45, i + 0.45, color="grey", lw=0.8, ls="--", zorder=0)
            ax.scatter(x, g.theta, s=16, facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=3)
            ax.scatter(x, g.theta_hat, s=16, facecolor=PRED, edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_xticks(range(len(sems)), sems, rotation=90 if len(sems) > 3 else 0)
        ax.text(0, 1.02, dom, transform=ax.transAxes, color="grey", ha="left")
    np.atleast_1d(axes)[0].set_ylabel("θ")
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="true θ"),
               plt.Line2D([], [], marker="o", ms=4, ls="", color=PRED, label=f"forecast ({LABELS.get(protocol, protocol)})"),
               plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
```

(`baseline` is the per-task level Task 13 introduced; `domains`/`widths` as before. Remove the old `fig.tight_layout()` and `bbox_to_anchor` legend.)

- [ ] **Step 4: Per-semester bars in the runner**

In `scripts/analyses/results.py`, replace the bar loop of the figures cell with:

```python
semesters = sorted({sem for _, sem in tasks})
for metric in stats.METRICS:
    plots.bar_figure(metrics, metric, VARIANTS, OUT / f"bars_{metric}.svg")
    for sem in semesters:
        plots.bar_figure(metrics, metric, VARIANTS, OUT / f"bars_{metric}_{sem}.svg", semester=sem)
```

- [ ] **Step 5: Render, look, test, commit**

Run (real-data prefix not needed): `PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; f = pd.read_csv('analysis/results/forecasts.csv').replace({'protocol': {'intercept': 'easy_items'}}); m = pd.read_csv('analysis/results/metrics.csv').replace({'protocol': {'intercept': 'easy_items'}}); v = {'pairwise': 'calibrated', 'pairwise_ensemble': 'calibrated', 'direct': 'raw', 'direct_ensemble': 'raw', 'embedding': 'full'}; plots.bar_figure(m, 'z', v, 'analysis/results/bars_z.png'); plots.bar_figure(m, 'mse', v, 'analysis/results/bars_mse.png'); plots.bar_figure(m, 'mse', v, 'analysis/results/bars_mse_2026-I.png', semester='2026-I'); plots.forecast_figure(f, 'pairwise', 'calibrated', 'analysis/results/forecasts_check.png')"` (drop the `.replace` once the runner has been re-run with the rename). Open all four PNGs: beeswarm points coloured by domain with no random scatter, black mean line with capless interval (none on the 2026-I figure if fewer than 3 domains), dashed easy-item line, zigzag break on `mse`, open-circle truths and filled forecasts with thin connectors on the forecast plot, legends outside, no overlap. `uv run pytest tests/test_analysis.py -q` — 7 passed.

```bash
git add src/analysis/plots.py scripts/analyses/results.py
git commit -m "analysis: beeswarm points by domain, 95% intervals, per-semester bar plots, restyled forecast dots"
```

---

### Task 16: Bar plot as the user specified it: plain axis, four conditions, method colours, per-domain lines

**Files:**
- Modify: `src/analysis/plots.py` (`bar_figure`; delete `outlier_split`, `zigzag`, `swarm_offsets`, `DOMAIN_COLORS`, `BAR`; change `CONDITIONS`, `COLORS`)
- Modify: `tests/test_analysis.py` only if a test references a deleted name (none do)

User instructions, verbatim: "the zig-zag is also insane. please dont just freestyle it! i can add the zigzag myself. drop the pooled variants. keep the current CIs for now." and earlier "colour the methods (similar for judge cases ...). unfilled circles for scatters - one per domain. and connect them in a line".

So: no axis-break code of any kind; one plain panel; the y-limits come from the values inside Tukey's far-out fence (3 IQR), everything beyond is simply not drawn and is printed to stdout so the user can add it by hand; the four conditions are `glm_regex`, `embedding`, `direct`, `pairwise` (the ensemble rows stay in the CSVs but are not drawn or tabulated); bars carry a colour per method; one open circle per domain at the domain's pooled value (semester "all" row per domain; when `semester` is given, that semester's row), connected across the conditions by a black line whose style identifies the domain; the black mean line and the capless 95% t-interval over the (domain, semester) tasks stay.

- [ ] **Step 1: Constants** (replace the existing `CONDITIONS`, `LABELS`, `COLORS`, `DOMAIN_COLORS`, `BAR` block)

```python
CONDITIONS = ["glm_regex", "embedding", "direct", "pairwise"]  # the bars, in this order
LABELS = {BASELINE: "easy-item prediction", "glm_regex": "length+regex", "embedding": "embedding",
          "direct": "direct judge", "pairwise": "pairwise judge"}
COLORS = dict(zip(CONDITIONS, ["#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6"]))  # one hue per method
DOMAIN_STYLES = {"science": "-", "math": "--", "coding": ":", "cyber": "-."}  # line style identifies the domain
TRUE, PRED = "black", "#eb6834"
```

Keep `METRIC_LABELS`, `chosen`, `ci95`, `series`-free helpers as they are. Delete `outlier_split`, `zigzag`, `swarm_offsets`, `DOMAIN_COLORS`, `BAR` and any import only they used.

- [ ] **Step 2: Replace `bar_figure`**

```python
def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None) -> None:
    """One bar per condition in CONDITIONS, coloured by method, at the mean over the (domain,
    semester) tasks of the metric ("mse", "delta" or "z"), with a capless 95% t-interval over
    those tasks (none below 3); one open circle per domain at the domain's pooled value
    (semester "all", or the given semester's value), the circles of a domain joined across the
    conditions by a black line whose style is DOMAIN_STYLES[domain]; the easy-item prediction
    as a dashed grey line (its mean raw MSE, or 0 for delta and z). A plain single axis: the
    y-limits are set by the values within Tukey's far-out fence (3 IQR) and anything beyond
    is not drawn but printed, so it can be marked by hand."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[m.domain != "all"]
    tasks = m[m.semester != "all"]
    if semester:
        tasks = tasks[tasks.semester == semester]
    per_domain = tasks if semester else m[m.semester == "all"]
    fig, ax = plt.subplots(figsize=(3.2, 2.4), layout="constrained")
    x = {p: i for i, p in enumerate(CONDITIONS)}
    drawn = []
    for p, i in x.items():
        values = tasks[tasks.protocol == p][metric].dropna().to_numpy()
        if len(values) == 0:
            continue
        ax.bar(i, values.mean(), color=COLORS[p], width=0.72, zorder=1)
        ax.hlines(values.mean(), i - 0.36, i + 0.36, color="black", lw=1.0, zorder=4)
        if np.isfinite(half := ci95(values)):
            ax.errorbar(i, values.mean(), yerr=half, color="black", lw=0.8, capsize=0, zorder=4)
        drawn += [values.mean()]
    for d, style in DOMAIN_STYLES.items():
        g = per_domain[per_domain.domain == d].set_index("protocol")[metric].reindex(CONDITIONS)
        keep = [p for p in CONDITIONS if np.isfinite(g[p])]
        if not keep:
            continue
        ax.plot([x[p] for p in keep], g[keep], color="black", lw=0.7, ls=style, zorder=3)
        ax.scatter([x[p] for p in keep], g[keep], s=18, facecolor="white", edgecolor="black", linewidth=0.8, zorder=5)
        drawn += list(g[keep])
    baseline = tasks[tasks.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    ax.axhline(baseline, color="grey", lw=0.8, ls="--", zorder=0)
    drawn = np.asarray(drawn + [baseline], float)
    q1, q3 = np.percentile(drawn, [25, 75])
    inside = drawn[(drawn >= q1 - 3 * (q3 - q1)) & (drawn <= q3 + 3 * (q3 - q1))]
    if len(inside) < len(drawn):  # off-axis values are left to the user to mark by hand
        print(f"{path}: off-axis {sorted(set(drawn) - set(inside))}")
        pad = 0.08 * max(np.ptp(inside), 1e-9)
        ax.set_ylim(min(inside.min(), 0) - pad, inside.max() + pad)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    domains = [d for d in DOMAIN_STYLES if d in set(per_domain.domain)]
    handles = [plt.Line2D([], [], color="black", lw=0.7, ls=DOMAIN_STYLES[d], marker="o", ms=4, markerfacecolor="white", label=d) for d in domains]
    handles += [plt.Line2D([], [], color="black", lw=0.8, label="mean ± 95% CI over tasks"),
                plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handlelength=2.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
```

`semester_table` and the runner need no change except that `CONDITIONS` now has four entries (the runner's `VARIANTS` may keep the ensemble keys; `chosen` ignores unknown protocols only through `CONDITIONS`, so ensemble rows are simply never selected for drawing or tabulating).

- [ ] **Step 3: Render, look, test, commit**

`PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; m = pd.read_csv('analysis/results/metrics.csv'); v = {'pairwise': 'calibrated', 'direct': 'raw', 'embedding': 'full'}; [plots.bar_figure(m, k, v, f'analysis/results/bars_{k}.png') for k in ('mse', 'delta', 'z')]; plots.bar_figure(m, 'mse', v, 'analysis/results/bars_mse_2026-I.png', semester='2026-I')"`. Open the four PNGs: four coloured bars, black mean line and interval, one open circle per domain joined by a styled black line, dashed easy-item line, no axis break, the math 2025-I length+regex value absent from `mse`/`delta` and printed to stdout, legend outside, no overlap. `uv run pytest tests/test_analysis.py -q` — 8 passed.

```bash
git add src/analysis/plots.py
git commit -m "analysis: bar plot on a plain axis, four conditions, method colours, per-domain connected circles"
```

---

### Task 17: Bar plot: compressed off-axis band, coloured solid domain lines, staggered circles

**Files:**
- Modify: `src/analysis/plots.py` (`bar_figure`; replace `DOMAIN_STYLES` with `DOMAIN_COLORS`; add `compress`)

User instructions, verbatim: "show 0-2, next tick at 500ish - but maybe positioned at around 3, with a gap - i just meant ill add zigzag. domains need to be coloured solid lines. error bars should be on domain x semester. stagger domain scatters".

So: one axis; the in-range values (Tukey far-out fence, 3 IQR) keep their true positions; every off-axis value is drawn in a compressed band above the in-range maximum, separated by a gap, at a position that preserves the off-axis values' order, with y-ticks at the drawn positions labelled with the true values (rounded to integers), so the user can add the break glyph by hand; no zigzag drawn. Domains: one solid line per domain in `DOMAIN_COLORS`, open circles with the domain's edge colour, the four domains staggered across the bar width; bars keep the method colours; the mean line and the capless 95% t-interval over the (domain, semester) tasks stay.

- [ ] **Step 1: Constants and helper** (replace `DOMAIN_STYLES`)

```python
DOMAIN_COLORS = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#eda100"}
STAGGER = 0.22  # the four domains' circles sit at -0.22, -0.073, +0.073, +0.22 around the bar centre


def compress(values: np.ndarray) -> tuple[np.ndarray, dict[float, float]]:
    """Map off-axis values into a band above the in-range values, keeping their order.

    values: every y that will be drawn (bar means, domain values, the baseline). Returns
    (position per value, {true off-axis value: drawn position}). In-range values (within
    3 IQR of the quartiles) keep their position. Off-axis values are placed above
    max(in-range) + a gap of 35% of the in-range span, spread over a band of another 35%
    in rank order (a single off-axis value sits at the band's centre). The dict is empty
    when nothing is off-axis."""
    values = np.asarray(values, float)
    q1, q3 = np.percentile(values, [25, 75])
    fence = 3 * (q3 - q1)
    off = (values > q3 + fence) | (values < q1 - fence)
    if not off.any():
        return values.copy(), {}
    inside = values[~off]
    span = max(np.ptp(inside), 1e-9)
    top = inside.max() + 0.35 * span
    distinct = np.unique(values[off])
    band = np.linspace(top + 0.175 * span, top + 0.35 * span, len(distinct)) if len(distinct) > 1 else np.array([top + 0.26 * span])
    mapping = dict(zip(distinct.tolist(), band.tolist()))
    positions = values.copy()
    positions[off] = [mapping[v] for v in values[off]]
    return positions, mapping
```

- [ ] **Step 2: Replace `bar_figure`**

```python
def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None) -> None:
    """One bar per condition in CONDITIONS, coloured by method, at the mean over the (domain,
    semester) tasks of the metric ("mse", "delta" or "z"), with a capless 95% t-interval over
    those tasks (none below 3); one open circle per domain at the domain's pooled value
    (semester "all", or the given semester's value), the domains staggered across the bar
    and each domain's circles joined across the conditions by a solid line in its colour; the
    easy-item prediction as a dashed grey line (its mean raw MSE, or 0 for delta and z).
    Off-axis values (beyond 3 IQR) are drawn in a compressed band above the in-range values
    with ticks at their true values (see compress); the break glyph is left to the user."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[m.domain != "all"]
    tasks = m[m.semester != "all"]
    if semester:
        tasks = tasks[tasks.semester == semester]
    per_domain = tasks if semester else m[m.semester == "all"]
    domains = [d for d in DOMAIN_COLORS if d in set(per_domain.domain)]
    # collect everything to draw first, so the off-axis compression sees every value
    means, halves = {}, {}
    for p in CONDITIONS:
        values = tasks[tasks.protocol == p][metric].dropna().to_numpy()
        if len(values):
            means[p], halves[p] = float(values.mean()), ci95(values)
    points = {d: per_domain[per_domain.domain == d].set_index("protocol")[metric].reindex(CONDITIONS) for d in domains}
    baseline = float(tasks[tasks.protocol == BASELINE].mse.mean()) if metric == "mse" else 0.0
    raw = np.array(list(means.values()) + [v for s in points.values() for v in s.dropna()] + [baseline])
    positions, mapping = compress(raw)
    place = dict(zip(raw.tolist(), positions.tolist()))
    fig, ax = plt.subplots(figsize=(3.2, 2.4), layout="constrained")
    x = {p: i for i, p in enumerate(CONDITIONS)}
    for p, mean in means.items():
        ax.bar(x[p], place[mean], color=COLORS[p], width=0.72, zorder=1)
        ax.hlines(place[mean], x[p] - 0.36, x[p] + 0.36, color="black", lw=1.0, zorder=4)
        if np.isfinite(halves[p]) and mean not in mapping:  # an interval around a compressed mean is meaningless
            ax.errorbar(x[p], mean, yerr=halves[p], color="black", lw=0.8, capsize=0, zorder=4)
    offsets = np.linspace(-STAGGER, STAGGER, len(domains)) if len(domains) > 1 else np.zeros(1)
    for d, dx in zip(domains, offsets):
        s = points[d]
        keep = [p for p in CONDITIONS if np.isfinite(s[p])]
        if not keep:
            continue
        xs = [x[p] + dx for p in keep]
        ys = [place[float(s[p])] for p in keep]
        ax.plot(xs, ys, color=DOMAIN_COLORS[d], lw=0.8, zorder=3)
        ax.scatter(xs, ys, s=18, facecolor="white", edgecolor=DOMAIN_COLORS[d], linewidth=0.9, zorder=5)
    ax.axhline(place[baseline], color="grey", lw=0.8, ls="--", zorder=0)
    if mapping:  # true-value ticks for the compressed band, after the in-range ticks
        inside = [t for t in ax.get_yticks() if t <= max(v for v in raw if v not in mapping)]
        ax.set_yticks(inside + list(mapping.values()), [f"{t:g}" for t in inside] + [f"{v:.0f}" for v in mapping])
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    handles = [plt.Line2D([], [], color=DOMAIN_COLORS[d], lw=0.8, marker="o", ms=4, markerfacecolor="white", label=d) for d in domains]
    handles += [plt.Line2D([], [], color="black", lw=0.8, label="mean ± 95% CI over domain × semester"),
                plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handlelength=2.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Render, look, test, commit**

`PYTHONPATH=src uv run python -c "import pandas as pd; from analysis import plots; m = pd.read_csv('analysis/results/metrics.csv'); v = {'pairwise': 'calibrated', 'direct': 'raw', 'embedding': 'full'}; [plots.bar_figure(m, k, v, f'analysis/results/bars_{k}.png') for k in ('mse', 'delta', 'z')]; plots.bar_figure(m, 'mse', v, 'analysis/results/bars_mse_2026-I.png', semester='2026-I')"`. Open the four PNGs: on `mse` the in-range axis runs about 0–2.5 with the length+regex bar mean and the math/science regex circles in a band above it, ticks there reading their true values (about 8, 45, 116), a visible gap below the band, no glyph; four solid coloured domain lines with open circles staggered inside each bar; black mean lines and intervals; dashed easy-item line; legend outside; no overlap. `uv run pytest tests/test_analysis.py -q` — 8 passed.

```bash
git add src/analysis/plots.py
git commit -m "analysis: bar plot with a compressed off-axis band, coloured solid domain lines, staggered circles"
```

---

## Self-review

**Spec coverage.** Conditions: intercept and oracle (Task 4), pairwise linear/calibrated and direct raw/item_z (Tasks 6–7), glm_regex via `PROTOCOLS` (Task 7), embedding full/answer from the `.npz` files (Task 5). Reference-only statistics: `normalise` uses references' ratings; every LOO drops the held-out reference from the calibration. Judge selection by reference LOO error and mean over judges (Task 8). Core condition rule (Task 8). Metrics raw/delta/z with paired bootstrap (Task 8 `paired_metrics`, `task_metrics`). The three requested outputs: forecast dot plot for the best condition with a one-semester version, bars with jittered domain×semester points and the intercept line, semester tables for every metric × (pooled, each domain) (Tasks 9–10). `missing` column (Task 7). Data locations via env (Global Constraints, Tasks 5 and 10). Docstrings on every function. Tests (Tasks 4, 6, 8).

**Placeholders.** None.

**Type consistency.** `judges.PAIRWISE`/`DIRECT` used by Task 7; `COLUMNS` in Task 7 includes `variant` and `judge_loo_mse`, which Task 8's `KEYS`, `judge_summaries` and tests rely on; `stats.BASELINE, BEST, JUDGED, METRICS, one_per_target` are imported by Task 9 with those names; `plots.chosen`, `plots.CONDITIONS`, `plots.semester_table(metrics, metric, variants, domain)`, `plots.forecast_figure(forecasts, protocol, variant, path, semesters)` and `plots.bar_figure(metrics, metric, variants, path)` match Task 10; `JudgeRun.harness.domain` exists (`JudgeBT.__init__` stores `domain`).
