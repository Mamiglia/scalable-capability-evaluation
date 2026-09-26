"""What drives the pairwise judges' verdicts: capability, length or vendor family, by dropping each from a full GLM.

    PYTHONPATH=src uv run python scripts/analyses/judge_verdict_glm.py build           # verdicts -> design.parquet
    PYTHONPATH=src uv run python scripts/analyses/judge_verdict_glm.py fit             # pred_<fold>.parquet, coefficients.csv
    PYTHONPATH=src uv run python scripts/analyses/judge_verdict_glm.py summarise       # runs.csv, heldout.csv, figure

Rows: the verdicts of figure 3's pairwise judge (per domain and semester the roster judge with the lowest reference
leave-one-out MSE, linear conversion; its core run, `calibration.core_runs`: reasoning shown, full train set) on the
frozen train items, 2025-II and 2026-I only (every domain present), one row per judged (item, pair, shown order).
Group g = (judge, domain, semester) = one run: 2 per domain.

    logit P(pick_left) = b_g^pos + b_g^cap dcap + b_g^len dlen + b_g^fam fam + sum_f b_g^f dfeature_f
    dcap = theta*_left - theta*_right (HARD items);  dlen = log chars_left - log chars_right (characters the judge
    saw);  fam = +1 left shares the judge's vendor and right does not, -1 mirrored, 0 otherwise;  dfeature_f = left minus right of
    each deterministic feature (FEATURES: the glm_regex set without its length terms) of the transcripts shown.  Covariates are
    divided by their sd over all verdicts, so a coefficient is logits per sd.  One maximum-likelihood fit per run: with
    two runs per domain there is no spread across runs to pool over (a hierarchical NUTS fit diverged for that reason).

Model comparison: the full model, the full model without each block (capability, length, family, the deterministic features
together), and position only,
each fitted on 4/5 of the items and scored on the held-out fifth (5 folds by item): deviance explained beyond position alone (1 - log-loss / position-only log-loss), and accuracy
(P > 0.5 on the picked side) for reference. A covariate's drop = the
full model's value minus the value without it. The claim is about the judge protocol used (figure 3's judge), so the unit
is the item: 95 % intervals and two-sided p from a paired bootstrap over items.

Writes analysis/judge/judge_verdict_glm/{design.parquet, pred_<fold>.parquet, heldout.csv, coefficients.csv,
judge_verdict_glm.png|svg|pdf (held-out deviance explained per model and domain)}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("analysis/judge/judge_verdict_glm")
DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = ["2025-II", "2026-I"]
# the paper's deterministic features (glm_regex) without its length terms, which the length covariate stands for
FEATURES = ["personal_pronouns", "hedging", "backtracking", "uncertainty", "confidence", "questions", "verification",
            "connectives", "contrast", "log_tool_calls"]
BLOCKS = {"capability": ["capability"], "length": ["length"], "family": ["family"],
          "deterministic features": [f"feature {f}" for f in FEATURES]}  # each dropped from the full model as one
COVARIATES = [c for cols in BLOCKS.values() for c in cols]
MODELS = {"full": COVARIATES, **{f"without {b}": [c for c in COVARIATES if c not in cols] for b, cols in BLOCKS.items()},
          "position only": []}
FOLDS = 5
FAMILY = {"deepseek": "deepseek", "gemma": "google", "glm": "zhipu", "gpt": "openai", "o4": "openai", "hermes": "nous",
          "nous": "nous", "kimi": "moonshot", "llama": "meta", "meta-llama": "meta", "minimax": "minimax",
          "mistral": "mistral", "mixtral": "mistral", "open-mistral": "mistral", "mpt": "mosaic", "nvidia": "nvidia",
          "phi": "microsoft", "wizardlm": "microsoft", "qwen": "alibaba", "ling": "ant"}  # as family_bias.py on main


def family(m: str) -> str:
    return next(v for k, v in FAMILY.items() if m.startswith(k))


def best_judges() -> dict[tuple[str, str], str]:
    """(domain, semester) -> figure 3's pairwise judge: the lowest reference leave-one-out MSE under the linear conversion,
    from analysis/results/forecasts.csv (written by scripts/analyses/results.py)."""
    f = pd.read_csv("analysis/results/forecasts.csv")
    f = f[(f.protocol == "pairwise") & (f.variant == "linear") & (f.condition == "reasoning-shown")]
    g = f.groupby(["domain", "semester", "judge"]).judge_loo_mse.first().reset_index()
    return g.loc[g.groupby(["domain", "semester"]).judge_loo_mse.idxmin()].set_index(["domain", "semester"]).judge.to_dict()


def features(task) -> pd.DataFrame:
    """FEATURES of the transcript a judge saw per (model, bench, item): the first epoch of the model's path_<bench> log,
    correct, reasoning visible (glm_features/, as transcript_features.reader.feature_table before averaging)."""
    from protocols.transcript_features.reader import features_path, read_features

    rows = []
    for cohort in (task.references, task.test):
        for bench, ids in cohort.items.groupby("bench").item:
            for model in cohort.models.index:
                path = cohort.models.loc[model, f"path_{bench}"]
                if isinstance(path, str):
                    f = read_features(features_path(path), set(ids))[FEATURES]
                    rows.append(f.reset_index().assign(model=model, bench=bench))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["model", "bench", "item"]).set_index(["model", "bench", "item"])


def build():
    from analysis import loading
    from analysis.calibration import core_runs

    best = best_judges()
    parts = []
    for name in DOMAINS:
        dom = loading.domain(name)
        for sem in SEMESTERS:
            task = loading.tasks(dom, [sem])[0]
            theta = pd.concat([task.references.theta, task.truth])
            feats = features(task)
            for run in core_runs(loading.pairwise_runs(dom, task)):
                if run.judge != best[(name, sem)]:
                    continue
                # as analysis.matching.match_pairwise: train items only, the run's own verdict over an imported one
                v = run.table.merge(task.references.items[["bench", "item"]]).sort_values("imported", kind="stable")
                v = v.drop_duplicates(["bench", "item", "model_left", "model_right"])
                print(f"{name} {sem} {run.judge} {run.condition}: {len(v)} verdicts, {v.chars_left.isna().sum()} without lengths")
                v = v.dropna(subset=["chars_left", "chars_right"])
                fam = lambda m: (v[m].map(family) == family(run.judge)).astype(float)  # noqa: E731
                side = lambda m: feats.reindex(pd.MultiIndex.from_arrays([v[m], v.bench, v.item.astype(str)])).to_numpy()  # noqa: E731
                dfeat = side("model_left") - side("model_right")
                print(f"  {np.isnan(dfeat[:, :-1]).any(1).sum()} verdicts without a transcript's features (difference := 0)")
                parts.append(pd.DataFrame({
                    "y": v.pick_left.to_numpy(float),
                    "capability": (v.model_left.map(theta) - v.model_right.map(theta)).to_numpy(),
                    "length": np.log(v.chars_left.astype(float)).to_numpy() - np.log(v.chars_right.astype(float)).to_numpy(),
                    "family": (fam("model_left") - fam("model_right")).to_numpy(),
                    # tool calls are NaN outside agentic logs: no tool-call difference there
                    **{f"feature {f}": np.nan_to_num(dfeat[:, i]) for i, f in enumerate(FEATURES)},
                    "judge": run.judge, "condition": run.condition, "domain": name, "semester": sem,
                    "bench": v.bench.to_numpy(), "item": v.item.astype(str).to_numpy(),
                }))
    d = pd.concat(parts, ignore_index=True)
    assert d.capability.notna().all(), "a model without theta*"
    items = d[["domain", "bench", "item"]].drop_duplicates()
    fold = pd.Series(np.random.default_rng(0).permutation(len(items)) % FOLDS, index=pd.MultiIndex.from_frame(items))
    d["fold"] = fold.reindex(pd.MultiIndex.from_frame(d[["domain", "bench", "item"]])).to_numpy()
    d["group"] = d.judge + " | " + d.domain + " | " + d.semester
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "design.parquet")
    print(f"{len(d)} verdicts, {d.group.nunique()} runs;", d.groupby("domain").size().to_dict())


def standardise(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(**{c: d[c] / d[c].std() for c in COVARIATES})


def logit(d: pd.DataFrame, cols: list[str]):
    """Maximum-likelihood logistic regression of y on position (intercept) + the cols that vary in d (tool calls are
    always 0 in maths and science). Returns (fit, the cols used)."""
    import statsmodels.api as sm
    cols = [c for c in cols if d[c].std() > 0]
    return sm.Logit(d.y.to_numpy(), np.column_stack([np.ones(len(d)), d[cols].to_numpy()])).fit(disp=0, maxiter=200), cols


def fit():
    """One logistic regression per judge run and model. pred_<fold>.parquet: the held-out fold's rows with every
    model's P(pick_left), each fitted on the run's other folds. coefficients.csv: the full model on all of a run's
    verdicts, estimate and 95 % Wald interval (logits per sd)."""
    raw = pd.read_parquet(OUT / "design.parquet")
    sd, d = raw[COVARIATES].std(), standardise(raw)
    for k in range(FOLDS):
        parts = []
        for _, g in d.groupby("group"):
            train, test = g[g.fold != k], g[g.fold == k].copy()
            for name, cols in MODELS.items():
                res, used = logit(train, cols)
                test[name] = res.predict(np.column_stack([np.ones(len(test)), test[used].to_numpy()]))
            parts.append(test)
        pd.concat(parts).to_parquet(OUT / f"pred_{k}.parquet")
        print(f"fold {k} done", flush=True)
    rows = []
    for group, g in d.groupby("group"):
        res, used = logit(g, COVARIATES)
        ci = res.conf_int()
        for j, c in enumerate(["position", *used]):
            rows.append(dict(group=group, domain=g.domain.iloc[0], coefficient=c, sd=1.0 if j == 0 else sd[c],
                             estimate=res.params[j], lo=ci[j, 0], hi=ci[j, 1], p=res.pvalues[j]))
    pd.DataFrame(rows).to_csv(OUT / "coefficients.csv", index=False, float_format="%.4f")


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def measures(ll: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """ll, correct: (..., models) mean log-loss and accuracy in MODELS order. Per model the accuracy and the deviance
    explained beyond position alone; per covariate the drop in each when it is removed from the full model."""
    m = list(MODELS)
    out = {}
    for i, name in enumerate(m):
        out[f"accuracy {name}"] = correct[..., i]
        out[f"deviance explained {name}"] = 1 - ll[..., i] / ll[..., m.index("position only")]
    out["accuracy random picker"] = np.full_like(correct[..., 0], 0.5)  # P = 0.5 on every verdict: log-loss ln 2
    out["deviance explained random picker"] = 1 - np.log(2) / ll[..., m.index("position only")]
    for c in BLOCKS:
        out[f"accuracy drop without {c}"] = out["accuracy full"] - out[f"accuracy without {c}"]
        out[f"deviance drop without {c}"] = out["deviance explained full"] - out[f"deviance explained without {c}"]
    return out


def summarise(n_boot: int = 10000):
    """heldout.csv: per domain and pooled, each measure over the held-out verdicts of the domain's judge runs, with its
    95 % bootstrap interval over items and, for the drops, the two-sided bootstrap p that the drop is 0 (paired: every
    resample scores all models on the same items). The claim is about the judge protocol used, so the unit is the item."""
    p = pd.concat([pd.read_parquet(OUT / f"pred_{k}.parquet") for k in range(FOLDS)], ignore_index=True)
    y = p.y.to_numpy()
    ll = p.assign(**{m: log_loss(y, p[m].to_numpy()) for m in MODELS})
    ok = p.assign(**{m: ((p[m].to_numpy() > 0.5) == (y > 0.5)).astype(float) for m in MODELS})
    rows = []
    rng = np.random.default_rng(0)
    key = ["domain", "bench", "item"]
    for dom in [*DOMAINS, "all"]:
        sel = (p.domain == dom).to_numpy() if dom != "all" else np.ones(len(p), bool)
        # per-item sums over both runs, so a bootstrap over items is a reweighting of these rows
        ll_item = ll[sel].groupby(key)[list(MODELS)].sum().to_numpy()
        ok_item = ok[sel].groupby(key)[list(MODELS)].sum().to_numpy()
        n = p[sel].groupby(key).size().to_numpy()
        est = measures(ll_item.sum(0) / n.sum(), ok_item.sum(0) / n.sum())
        w = rng.multinomial(len(n), np.full(len(n), 1 / len(n)), size=n_boot)  # (n_boot, items) resample counts
        boot = measures((w @ ll_item) / (w @ n)[:, None], (w @ ok_item) / (w @ n)[:, None])
        for q, val in est.items():
            b = boot[q]
            pval = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())) if "drop" in q else np.nan
            rows.append(dict(domain=dom, quantity=q, n_items=len(n), value=float(val), lo=np.quantile(b, 0.025),
                             hi=np.quantile(b, 0.975), p=pval))
    heldout = pd.DataFrame(rows)
    heldout.to_csv(OUT / "heldout.csv", index=False, float_format="%.5f")
    print(heldout[heldout.quantity.str.startswith("deviance")].round(4).to_string(index=False))
    figure(heldout)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def figure(heldout: pd.DataFrame):
    """Figures 4/5 style: one panel per domain, one row per model. Dot = held-out deviance explained beyond position
    alone (%) over the domain's verdicts, box = 95 % bootstrap interval over items; dashed line = the full model; grey
    point = a random picker (P = 0.5 on every verdict); marks = two-sided paired bootstrap test of the drop from the
    full model (*** p<.001, ** p<.01, * p<.05, n.s.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.challenges.contributions import DOMAIN_TITLES
    from analysis.challenges.signal_glm import DOMAIN_COLORS
    from analysis.plots import unclip_markers

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial"], "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    rows = [m for m in MODELS if m != "position only"]
    h = heldout.set_index(["domain", "quantity"])
    cols = [f"deviance explained {m}" for m in [*rows, "random picker"]]
    acc = heldout[heldout.quantity.isin(cols)]
    lo, hi = 100 * min(0, acc.lo.min()), 100 * acc.hi.max()
    for step in (5, 10, 20):  # the finest step with at most 5 ticks, through 0, covering every interval
        ticks = list(np.arange(np.floor(lo / step), np.ceil(hi / step) + 1) * step)
        if len(ticks) <= 5:
            break
    span = ticks[-1] - ticks[0]
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(5.2, 1.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, dom in zip(axes, DOMAINS):
        colour = DOMAIN_COLORS[dom]
        full = 100 * h.loc[(dom, "deviance explained full"), "value"]
        ax.vlines(full, -0.5, len(rows) + 0.6, colors="0.5", lw=0.8, linestyles="--", zorder=1.5)
        edge = 100 * max(h.loc[(dom, f"deviance explained {m}"), "hi"] for m in rows) + 0.04 * span  # marks just past the widest box
        for y, m in enumerate(rows):
            r = 100 * h.loc[(dom, f"deviance explained {m}"), ["value", "lo", "hi"]]
            without = m != "full"
            ax.add_patch(plt.Rectangle((r.lo, y - 0.25), r.hi - r.lo, 0.5, color=colour, alpha=0.6 if without else 0.35, linewidth=0, zorder=1))
            ax.plot(r.value, y, "o", ms=3.5, mfc="white" if without else colour, mec="black", mew=0.4, zorder=2)
            if without:
                ax.text(edge, y, stars(h.loc[(dom, f"deviance drop {m}"), "p"]), ha="left", va="center", fontsize=6)
        # a last row for scale: the random picker (P = 0.5 on every verdict), a grey point, no interval, no test
        ax.plot(100 * h.loc[(dom, "deviance explained random picker"), "value"], len(rows), "o", ms=3.5, mfc="0.6", mec="black", mew=0.4, zorder=2)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlim(ticks[0], ticks[-1] + 0.2 * span)  # room for the marks, and so neighbouring tick labels do not touch
        ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows) + 0.6, -0.6)
        ax.set_title(DOMAIN_TITLES[dom], loc="left", fontsize=7, fontweight="bold")
    axes[0].set_yticks(range(len(rows) + 1), [m[0].upper() + m[1:] + (" model" if m == "full" else "") for m in rows] + ["Random picker"])
    fig.supxlabel("Held-out deviance explained (%)", fontsize=8, y=-0.12)
    unclip_markers(fig)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(OUT / f"judge_verdict_glm{suffix}", dpi=400, bbox_inches="tight")
    print(OUT / "judge_verdict_glm.png")


if __name__ == "__main__":
    {"build": lambda: build(), "fit": lambda: fit(), "summarise": lambda: summarise(), "figure": lambda: figure(pd.read_csv(OUT / "heldout.csv"))}[sys.argv[1]]()
