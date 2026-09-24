"""One model, every domain, every protocol: its capability theta* and what each protocol forecast for it.

    PYTHONPATH=src uv run python scripts/analyses/model_report.py kimi-k2.6            # substring of a data/ shortname
    PYTHONPATH=src uv run python scripts/analyses/model_report.py kimi-k2.6 --pairs    # + per-reference win rates (reads judge logs)

Reads data/meta/<domain>/{models,items,responses}.csv and analysis/results/forecasts.csv (scripts/analyses/results.py):
core condition, pairwise "linear", direct "raw", embedding "full". Per forecast the prediction score
S = -log2(e / e_base) against the reference-mean constant (e = squared error in theta). Writes to
analysis/models/<model>/:

  capability.csv       per domain: task, theta*, sd, global theta, rank among the targets, references' mean/max, accuracy
                       per bench on all / easy / hard items (repeat 0), and the cohort's mean accuracy
  forecasts.csv        per domain x protocol x judge: theta_hat, error, score, judge reference-LOO mse, missing judgments
  capability.{png,svg} theta* of the model against its cohort (references open, targets grey) per domain; accuracy per bench
  forecasts.{png,svg}  per domain: theta_hat of every protocol (one dot per judge, best-on-references ringed) vs theta*
  scores.{png,svg}     heatmap protocol x domain of the prediction score (judged protocols: best judge on references)
  pairs.{png,svg}      (--pairs) per domain x judge: the model's win rate against each reference vs the reference's theta*
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plots import HEAT, INK, INK2, MUTED, PALETTE, WIDE, legend, save, style  # noqa: E402  (sets the Agg backend)

import matplotlib.pyplot as plt  # noqa: E402

from analysis.stats import BASELINE, ORACLE, core_condition  # noqa: E402

DOMAINS = ["science", "math", "coding", "cyber"]
FORECASTS = Path("analysis/results/forecasts.csv")
VARIANT = {"pairwise": "linear", "pairwise_ensemble": "linear", "direct": "raw", "direct_ensemble": "raw", "embedding": "full"}
LABEL = {BASELINE: "Easy items", ORACLE: "Oracle (hard acc.)", "glm_length": "Length ridge", "glm_regex": "Regex ridge",
         "embedding": "Embeddings", "pairwise": "Pairwise judge", "direct": "Direct judge",
         "pairwise_ensemble": "Pairwise ensemble", "direct_ensemble": "Direct ensemble"}
ORDER = list(LABEL)
RANK_COLOR = {1: "#1f63b6", 2: "#7f9cc4", 3: "#b8b5ab"}  # judge rank by theta* among the references
ACCENT = PALETTE[1]


def resolve(query: str) -> str:
    names = set()
    for d in DOMAINS:
        names |= set(pd.read_csv(f"data/meta/{d}/models.csv").model)
    hits = sorted(n for n in names if query.lower() in n.lower())
    if len(hits) != 1:
        sys.exit(f"{query!r} matches {hits or 'nothing'} in data/meta/*/models.csv")
    return hits[0]


def task_of(models: pd.DataFrame, model: str) -> str | None:
    """The semester task in which the model is a target (else a reference), None if never rostered."""
    roles = models.loc[model, [c for c in models.columns if c.endswith("_role")]]
    for role in ("target", "reference"):
        hit = roles[roles == role]
        if len(hit):
            return hit.index[0].removesuffix("_role")
    return None


def capability(model: str) -> tuple[pd.DataFrame, dict]:
    """One row per domain (capability.csv) and the per-domain context the figure draws (cohort theta*, accuracy)."""
    rows, ctx = [], {}
    for d in DOMAINS:
        models = pd.read_csv(f"data/meta/{d}/models.csv", index_col="model")
        if model not in models.index or (sem := task_of(models, model)) is None:
            continue
        theta, role = models[f"{sem}_theta"], models[f"{sem}_role"]
        refs, targets = theta[role == "reference"].dropna(), theta[role == "target"].dropna()
        items = pd.read_csv(f"data/meta/{d}/items.csv")
        resp = pd.read_csv(f"data/meta/{d}/responses.csv").query("repeat == 0").merge(items[["bench", "item", f"{sem}_set"]])
        cohort = resp[resp.model.isin(targets.index)]
        acc = {}
        for bench, g in resp[resp.model == model].groupby("bench"):
            acc[f"acc_{bench}"] = g.correct.mean()
            acc[f"acc_{bench}_easy"] = g[g[f"{sem}_set"] == "easy"].correct.mean()
            acc[f"acc_{bench}_hard"] = g[g[f"{sem}_set"] == "hard"].correct.mean()
            c = cohort[cohort.bench == bench]
            acc[f"cohort_acc_{bench}"] = c.groupby("model").correct.mean().mean()
        rows.append(dict(domain=d, task=sem, role=role[model], theta=theta[model], theta_sd=models.loc[model, f"{sem}_theta_sd"],
                         global_theta=models.loc[model, "global_theta"], rank=int((targets > theta[model]).sum() + 1), n_targets=len(targets),
                         ref_mean=refs.mean(), ref_max=refs.max(), **acc))
        ctx[d] = dict(sem=sem, refs=refs, targets=targets, acc=acc)
    return pd.DataFrame(rows), ctx


def forecasts(model: str, cap: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_csv(FORECASTS)
    f = core_condition(f[f.model == model])
    f = f[f.variant == f.protocol.map(VARIANT).fillna("-")].copy()
    base = cap.set_index("domain").ref_mean
    f["e"] = (f.theta_hat - f.theta) ** 2
    f["e_base"] = (f.domain.map(base) - f.theta) ** 2
    f["score"] = -np.log2(f.e / f.e_base)
    f["error"] = f.theta_hat - f.theta
    return f


def judge_rank(f: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Judge rank by theta* among the references (1 = strongest); best = lowest reference-LOO mse within the cell."""
    f = f.copy()
    f["judge_theta"] = [ctx[d]["refs"].get(j, np.nan) for d, j in zip(f.domain, f.judge)]
    judged = f.judge.ne("-") & f.judge.ne("ensemble")
    f.loc[judged, "rank"] = f[judged].groupby(["domain", "protocol"]).judge_theta.rank(ascending=False, method="dense")
    best = f[judged].groupby(["domain", "protocol"]).judge_loo_mse.transform("min").reindex(f.index)
    f["best"] = judged & (f.judge_loo_mse == best)
    return f


def capability_figure(model: str, cap: pd.DataFrame, ctx: dict, out: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.6), gridspec_kw={"width_ratios": [1, 1.4]})
    rng = np.random.default_rng(0)
    for i, row in cap.iterrows():
        c = ctx[row.domain]
        a.plot(rng.uniform(-0.15, 0.15, len(c["refs"])) + i, c["refs"], "o", ms=3.5, mfc="none", mec=MUTED, mew=0.8, label="references" if i == 0 else None)
        others = c["targets"].drop(model)
        a.plot(rng.uniform(-0.15, 0.15, len(others)) + i, others, "o", ms=3.5, color=MUTED, label="targets" if i == 0 else None)
        a.errorbar(i, row.theta, yerr=row.theta_sd, fmt="o", ms=5, color=ACCENT, lw=1.2, zorder=5, label=model if i == 0 else None)
        a.annotate(f"#{row['rank']}/{row.n_targets}", (i, row.theta), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.5, color=ACCENT)
    a.set_xticks(range(len(cap)), [f"{r.domain}\n{r.task}" for r in cap.itertuples()])
    a.set_ylabel("θ* on the task")
    style(a)
    legend(a, fontsize=6.5)
    ticks, labels, y = [], [], 0
    for row in cap.itertuples():
        r = row._asdict()
        benches = [k.removeprefix("acc_") for k, v in r.items() if k.startswith("acc_") and not k.endswith(("_easy", "_hard")) and v == v]
        for bench in benches:
            b.barh(y, r[f"acc_{bench}"], color=ACCENT, height=0.7)
            b.plot(r[f"cohort_acc_{bench}"], y, "|", color=INK, ms=9, mew=1.2)
            b.plot(r[f"acc_{bench}_hard"], y, "o", ms=3, mfc="none", mec=INK2, mew=0.8)
            ticks.append(y)
            labels.append(f"{row.domain}: {bench}")
            y += 1
        y += 0.6
    b.set_yticks(ticks, labels)
    b.invert_yaxis()
    b.set_xlim(0, 1)
    b.set_xlabel("accuracy, repeat 0  (bar: all items · ○: hard items · |: cohort mean)")
    style(b, grid_axis="x")
    fig.suptitle(model, fontsize=9, x=0.02, ha="left")
    save(fig, out)


def forecast_figure(model: str, cap: pd.DataFrame, f: pd.DataFrame, out: Path):
    protocols = [p for p in ORDER if p in set(f.protocol)]
    fig, axes = plt.subplots(1, len(cap), figsize=(WIDE, 0.35 * len(protocols) + 0.9), sharey=True, sharex=True)
    for ax, row in zip(np.atleast_1d(axes), cap.itertuples()):
        g = f[f.domain == row.domain]
        ax.axvspan(row.theta - row.theta_sd, row.theta + row.theta_sd, color=ACCENT, alpha=0.15, lw=0)
        ax.axvline(row.theta, color=ACCENT, lw=1.2, zorder=1)
        ax.axvline(row.ref_mean, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for y, p in enumerate(protocols):
            rows = g[g.protocol == p]
            for r in rows.itertuples():
                if np.isnan(r.theta_hat):
                    continue
                if r.judge == "-":
                    ax.plot(r.theta_hat, y, "o", ms=4.5, color=INK2, zorder=4)
                elif r.judge == "ensemble":
                    ax.plot(r.theta_hat, y, "D", ms=4, color=INK2, zorder=4)
                else:
                    color = RANK_COLOR.get(int(r.rank), MUTED) if not np.isnan(r.rank) else MUTED
                    ax.plot(r.theta_hat, y, "o", ms=4.5, color=color, mec=INK if r.best else color, mew=1.2 if r.best else 0, zorder=5 if r.best else 4)
        ax.set_yticks(range(len(protocols)), [LABEL[p] for p in protocols])
        ax.invert_yaxis()
        ax.set_title(f"{row.domain} {row.task}\nθ* = {row.theta:.2f} ± {row.theta_sd:.2f}", fontsize=8, loc="left")
        ax.set_xlabel("θ̂")
        style(ax, grid_axis="x")
    for rank, color in RANK_COLOR.items():
        axes[-1].plot([], [], "o", ms=4.5, color=color, label=f"judge #{rank} by θ*")
    axes[-1].plot([], [], "o", ms=4.5, mfc="none", mec=INK, mew=1.2, label="best judge on refs")
    axes[-1].plot([], [], "D", ms=4, color=INK2, label="judge ensemble")
    axes[-1].plot([], [], "-", color=ACCENT, label="θ* ± sd")
    axes[-1].plot([], [], ls=(0, (4, 3)), color=MUTED, label="reference mean")
    legend(axes[-1], loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6.5)
    fig.suptitle(model, fontsize=9, x=0.02, ha="left")
    save(fig, out)



def score_table(f: pd.DataFrame) -> pd.DataFrame:
    """protocol x domain prediction score; judged protocols by the best judge on references, ensembles as they are."""
    keep = f[(f.judge == "-") | (f.judge == "ensemble") | f.best]
    t = keep.pivot_table(index="protocol", columns="domain", values="score", aggfunc="first")
    t = t.reindex([p for p in ORDER if p in t.index], columns=[d for d in DOMAINS if d in t.columns])
    t["mean"] = t.mean(axis=1)
    return t


def score_figure(model: str, t: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(0.75 * t.shape[1] + 1.6, 0.3 * len(t) + 0.8))
    v = 5  # colour saturates at |S| = 5 (e = e_base / 32); the number in the cell is exact
    ax.imshow(t.to_numpy(float), cmap=HEAT, vmin=-v, vmax=v, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            x = t.iat[i, j]
            if not np.isnan(x):
                ax.text(j, i, f"{x:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(len(t)), [LABEL.get(p, p) for p in t.index])
    ax.tick_params(length=0, colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{model}: prediction score  −log2(e / e_ref-mean)", fontsize=8, loc="left")
    save(fig, out)


def pair_rates(model: str, cap: pd.DataFrame) -> pd.DataFrame:
    """Win rate of the model against each reference, per domain x judge (core pairwise run, train items only)."""
    from analysis import calibration, loading

    rows = []
    for row in cap.itertuples():
        dom = loading.domain(row.domain)
        task = loading.tasks(dom, [row.task])[0]
        for run in calibration.core_runs(loading.pairwise_runs(dom, task)):
            t = run.table.merge(task.references.items[["bench", "item"]])
            t = t[(t.model_left == model) | (t.model_right == model)]
            t["win"] = np.where(t.model_left == model, t.pick_left, ~t.pick_left.astype(bool))
            t["ref"] = np.where(t.model_left == model, t.model_right, t.model_left)
            for ref, g in t.groupby("ref"):
                rows.append(dict(domain=row.domain, judge=run.judge, condition=run.condition, ref=ref,
                                 ref_theta=task.references.theta.get(ref, np.nan), n=len(g), win=g.win.mean()))
    return pd.DataFrame(rows)


def pair_figure(model: str, cap: pd.DataFrame, f: pd.DataFrame, pairs: pd.DataFrame, out: Path):
    """Rows = judge rank by theta* within the domain (the top-3 references differ per domain), columns = domains."""
    ranked = f[(f.protocol == "pairwise") & f.judge.ne("-") & f.judge.ne("ensemble")].dropna(subset=["rank"])
    nrow = int(ranked["rank"].max())
    fig, axes = plt.subplots(nrow, len(cap), figsize=(WIDE, 1.55 * nrow + 0.5), sharex="col", sharey=True, squeeze=False)
    for i, row in enumerate(cap.itertuples()):
        cell = ranked[ranked.domain == row.domain].set_index("rank")
        for j in range(nrow):
            ax = axes[j, i]
            style(ax)
            ax.set_ylim(-0.05, 1.05)
            if j + 1 not in cell.index:
                ax.set_axis_off()
                continue
            r = cell.loc[j + 1]
            g = pairs[(pairs.domain == row.domain) & (pairs.judge == r.judge)]
            lo, hi = g.ref_theta.min() - 0.5, max(g.ref_theta.max(), row.theta, r.theta_hat) + 0.5
            x = np.linspace(lo, hi, 100)
            ax.plot(x, 1 / (1 + np.exp(-(row.theta - x))), color=MUTED, lw=0.8, ls=(0, (4, 3)))
            ax.axvline(row.theta, color=ACCENT, lw=1)
            ax.axvline(r.theta_hat, color=RANK_COLOR[1], lw=1)
            ax.scatter(g.ref_theta, g.win, s=8 + g.n / 10, color=INK2, zorder=4)
            ax.axhline(0.5, color=MUTED, lw=0.5)
            ax.set_title(f"{row.domain} · {r.judge}" + ("\nbest judge on refs" if r.best else "\n"), fontsize=7, loc="left")
            if i == 0:
                ax.set_ylabel("win rate")
            if j == nrow - 1:
                ax.set_xlabel("reference θ*")
    axes[0, -1].plot([], [], color=ACCENT, label="θ*")
    axes[0, -1].plot([], [], color=RANK_COLOR[1], label="judge θ̂")
    axes[0, -1].plot([], [], color=MUTED, ls=(0, (4, 3)), label="BT at θ*")
    legend(axes[0, -1], loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6.5)
    fig.suptitle(f"{model}: pairwise picks against each reference (dot size ~ verdicts)", fontsize=9, x=0.02, ha="left")
    save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--pairs", action="store_true", help="per-reference win rates from the pairwise judge logs (slow)")
    args = ap.parse_args()
    model = resolve(args.model)
    out = Path("analysis/models") / model
    out.mkdir(parents=True, exist_ok=True)

    cap, ctx = capability(model)
    cap.to_csv(out / "capability.csv", index=False, float_format="%.4f")
    print(f"== {model}: capability")
    print(cap[["domain", "task", "role", "theta", "theta_sd", "global_theta", "rank", "n_targets", "ref_mean", "ref_max"]].round(3).to_string(index=False))
    acc = cap.set_index("domain").filter(like="acc_").stack().rename("acc").reset_index()
    print(acc.round(3).to_string(index=False))
    capability_figure(model, cap, ctx, out / "capability")

    f = judge_rank(forecasts(model, cap), ctx)
    assert f.loc[f.protocol == BASELINE, "score"].abs().max() < 1e-3, "easy-item prediction is the reference-mean constant"
    f = f.sort_values(["domain", "protocol", "rank"], key=lambda s: s.map(ORDER.index) if s.name == "protocol" else s)
    cols = ["domain", "protocol", "judge", "condition", "theta", "theta_hat", "error", "score", "judge_loo_mse", "best", "missing"]
    f[cols].to_csv(out / "forecasts.csv", index=False, float_format="%.4f")
    print(f"\n== {model}: forecasts (core condition; pairwise linear, direct raw)")
    print(f[cols].round(3).to_string(index=False))
    forecast_figure(model, cap, f, out / "forecasts")

    t = score_table(f)
    print(f"\n== {model}: prediction score S = -log2(e / e_base), judged protocols = best judge on references")
    print(t.round(2).to_string())
    score_figure(model, t, out / "scores")

    if args.pairs:
        pairs = pair_rates(model, cap)
        pairs.to_csv(out / "pairs.csv", index=False, float_format="%.4f")
        print(f"\n== {model}: pairwise win rate against each reference")
        print(pairs.round(3).to_string(index=False))
        pair_figure(model, cap, f, pairs, out / "pairs")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
