"""Hierarchical GLM of the pairwise judges' choices: what drives pick_left (no judging is run).

    PYTHONPATH=src uv run python scripts/analyses/judge_glm.py [--fast] [--no-hidden]

Rows: the verdicts cached by family_bias.py (every run on the frozen train items, judges with > 500 verdicts,
conditions hidden / shown / shown-10pq), one row per judged (item, pair, shown order). Group g = (judge, domain).

    logit P(pick_left) = a_g + b_g^theta dtheta + b_g^len dlen + b_g^fam fam
                       + shown * (d^a + d^theta dtheta + d^len dlen + d^fam fam)
    dtheta = theta*_left - theta*_right (HARD-item theta*);  dlen = log chars_left - log chars_right, the characters
    the judge saw (reasoning only in the shown condition);  fam = +1 left shares the judge's vendor and right does not,
    -1 mirrored, 0 otherwise (the judge itself counts as its vendor);  shown = 1 for the reasoning-shown conditions.  Every covariate except position is divided by its sd over
    all rows (`sd` column of population.csv), so a coefficient is the logit change per 1 sd; per-unit = coefficient / sd.
    b_g^k ~ N(mu^k[domain(g)], sigma^k),  mu ~ N(0, 2), sigma ~ HalfNormal(1), d ~ N(0, 1).  NUTS, centred.

Explained deviance: the same model with only the intercept ("position"), then subsets of the terms up to everything
("full"), is fitted on 80 % of the items and scored by log-loss on the other 20 % (split by item, seeded).
The reported coefficients come from the full model on every verdict.

--no-hidden: reasoning-shown runs only (no shown offsets d), and the population mean gets a per-semester offset,
b_g ~ N(mu[domain(g)] + e[semester(g)], sigma), e[2026-I] = 0, e ~ N(0, 1) (every judge sits in one semester). Also fits
the length decomposition: dlen split into its (judge, ordered model pair) mean ("length between", carries the
capability gap) and the item-level deviation from it ("length within", orthogonal to dtheta by construction), plus
dcorrect = mean correct_left - correct_right over the item's repeats. Writes to analysis/judge/judge_glm/no_hidden/.

Writes analysis/judge/judge_glm/{population.csv, judges.csv, heldout.csv, recent.csv, judge_glm.png|svg (every judge by
release semester), judge_glm_recent.png|svg (judges of the last 3 semesters: per-domain and overall mean +- 95 % CI over judges)}.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import yaml
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).parent))
from family_bias import DOMAINS, MIN_VERDICTS, OUT as FB_OUT, RUN, family, relation  # noqa: E402
from plots import INK2, MUTED, PALETTE, SEQ, save, style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

numpyro.set_host_device_count(2)
OUT = Path("analysis/judge/judge_glm")
COEFS = ["position", "capability", "length", "family"]  # order of the design columns
SHOWN = ["position", "capability", "length", "family"]  # columns with a reasoning-shown offset
VARIANTS = {"position": [], "capability": ["capability"], "length": ["length"], "capability + length": ["capability", "length"],
            "full": COEFS}
DECOMP = ["position", "capability", "correct", "length between", "length within"]
REF_SEMESTER = "2026-I"  # semester offset fixed at 0 (--no-hidden)
SD: dict[str, float] = {}  # covariate -> sd used for standardisation, filled by design()


def correctness(v: pd.DataFrame, side: str) -> np.ndarray:
    """Mean correct over the repeats of (item, model_<side>), from data/meta/<domain>/responses.csv."""
    r = pd.concat([pd.read_csv(f"data/meta/{dom}/responses.csv", dtype={"item": str}).assign(domain=dom) for dom in DOMAINS])
    c = r.groupby(["domain", "bench", "item", "model"]).correct.mean()
    return c.reindex(pd.MultiIndex.from_arrays([v.domain, v.bench, v.item.astype(str), v[f"model_{side}"]])).to_numpy()


def design(no_hidden: bool = False) -> pd.DataFrame:
    """One row per verdict with y, the covariates, group (judge, domain) and domain indices."""
    v = pd.read_parquet(FB_OUT / "verdicts.parquet")
    counts = v.groupby(RUN).size()
    v = v[counts.reindex(pd.MultiIndex.from_frame(v[RUN])).to_numpy() > MIN_VERDICTS]
    if no_hidden:
        v = v[v.condition != "hidden"]
    theta = pd.read_csv(FB_OUT / "theta.csv").set_index(["domain", "semester", "model"]).theta
    n = len(v)
    v = v.dropna(subset=["chars_left", "chars_right"])
    print(f"{n - len(v)} verdicts without shown lengths dropped")
    key = lambda m: pd.MultiIndex.from_arrays([v.domain, v.semester, v[m]])  # noqa: E731
    v = v.assign(theta_l=theta.reindex(key("model_left")).to_numpy(), theta_r=theta.reindex(key("model_right")).to_numpy())
    n = len(v)
    v = v.dropna(subset=["theta_l", "theta_r"])
    print(f"{n - len(v)} verdicts without theta* on both sides dropped")
    rel_l = np.array([relation(j, m) != "neutral" for j, m in zip(v.judge_name, v.model_left)])
    rel_r = np.array([relation(j, m) != "neutral" for j, m in zip(v.judge_name, v.model_right)])
    d = pd.DataFrame({
        "y": v.pick_left.to_numpy(float),
        "position": 1.0,
        "capability": (v.theta_l - v.theta_r).to_numpy(),
        "length": np.log(v.chars_left.to_numpy(float)) - np.log(v.chars_right.to_numpy(float)),
        "family": rel_l.astype(float) - rel_r.astype(float),
        "shown": (v.condition != "hidden").to_numpy(float),
        "judge": v.judge_name.to_numpy(), "domain": v.domain.to_numpy(), "semester": v.semester.to_numpy(),
        "condition": v.condition.to_numpy(), "bench": v.bench.to_numpy(), "item": v.item.to_numpy(),
    })
    d["group"] = d.judge + " | " + d.domain
    if no_hidden:
        d["correct"] = correctness(v, "left") - correctness(v, "right")
        print(f"{d.correct.isna().sum()} verdicts without correctness (correct := 0)")
        d["correct"] = d.correct.fillna(0)
        pair = [d.judge, d.condition, d.domain, d.semester, v.model_left.to_numpy(), v.model_right.to_numpy()]
        d["length between"] = d.groupby(pair).length.transform("mean")
        d["length within"] = d.length - d["length between"]
        print("corr(dlen, dtheta*) =", round(float(np.corrcoef(d.length, d.capability)[0, 1]), 3),
              "| between:", round(float(np.corrcoef(d["length between"], d.capability)[0, 1]), 3),
              "| within:", round(float(np.corrcoef(d["length within"], d.capability)[0, 1]), 3),
              "| corr(within, dcorrect):", round(float(np.corrcoef(d["length within"], d.correct)[0, 1]), 3))
    for c in [c for c in d.columns if c in COEFS[1:] + DECOMP[1:]]:  # standardise, so coefficients are per 1 sd and the shared sigma is on one scale
        SD[c] = float(d[c].std())
        d[c] = d[c] / SD[c]
    return d


def model(X, shown, group, dom_of_group, n_dom, cols, sem_of_group=None, n_sem=0, y=None):
    """X: (n, k) design columns in `cols` order; shown: (n,); group: (n,) index; dom_of_group: (G,).
    sem_of_group given: semester offsets on the population mean (row 0 = REF_SEMESTER, fixed at 0) and no shown offsets."""
    k, G = X.shape[1], len(dom_of_group)
    mu = numpyro.sample("mu", dist.Normal(0, 2).expand([n_dom, k]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1).expand([k]))
    loc = mu[dom_of_group]
    if sem_of_group is not None:
        e = numpyro.sample("semester", dist.Normal(0, 1).expand([n_sem - 1, k]))
        loc = loc + jnp.concatenate([jnp.zeros((1, k)), e])[sem_of_group]
    # centred: with ~10k verdicts per group the data pin every beta_g, so the non-centred form is the one with a funnel
    beta = numpyro.sample("beta", dist.Normal(loc, sigma).to_event(2))
    logit = jnp.sum(beta[group] * X, axis=1)
    shown_cols = [i for i, c in enumerate(cols) if c in SHOWN]
    if shown_cols and sem_of_group is None:
        delta = numpyro.sample("delta", dist.Normal(0, 1).expand([len(shown_cols)]))
        logit = logit + shown * jnp.sum(delta * X[:, shown_cols], axis=1)
    numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


def fit(d: pd.DataFrame, cols: list[str], groups: pd.Index, doms: pd.Index, warmup: int, samples: int, seed: int = 0,
        sems: pd.Index | None = None):
    """NUTS posterior of `model` on the rows of d; returns (samples dict, mcmc). sems: semester offsets (--no-hidden)."""
    dom_of_group = jnp.array(doms.get_indexer(pd.Series(groups).str.split(" \\| ").str[1]))
    args = dict(X=jnp.array(d[cols].to_numpy(np.float32)), shown=jnp.array(d.shown.to_numpy(np.float32)),
                group=jnp.array(groups.get_indexer(d.group)), dom_of_group=dom_of_group, n_dom=len(doms), cols=cols)
    if sems is not None:
        sem = d.drop_duplicates("group").set_index("group").semester.reindex(groups)
        args |= dict(sem_of_group=jnp.array(sems.get_indexer(sem)), n_sem=len(sems))
    mcmc = MCMC(NUTS(model), num_warmup=warmup, num_samples=samples, num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), y=jnp.array(d.y.to_numpy(np.float32)), **args)
    return mcmc.get_samples(group_by_chain=True), mcmc, args


def predict(post, args, d: pd.DataFrame, cols: list[str], groups: pd.Index) -> np.ndarray:
    """Posterior-mean P(pick_left) for the rows of d (posterior mean of the logit)."""
    beta = np.asarray(post["beta"]).reshape(-1, *post["beta"].shape[2:]).mean(0)
    logit = (beta[groups.get_indexer(d.group)] * d[cols].to_numpy()).sum(1)
    if "delta" in post and "semester" not in post:
        delta = np.asarray(post["delta"]).reshape(-1, post["delta"].shape[-1]).mean(0)
        logit = logit + d.shown.to_numpy() * (delta * d[[c for c in cols if c in SHOWN]].to_numpy()).sum(1)
    return 1 / (1 + np.exp(-logit))


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def diagnostics(post, mcmc, label):
    s = summary(post, group_by_chain=True)
    rhat = max(float(np.nanmax(v["r_hat"])) for k, v in s.items() if k != "beta")
    div = int(np.sum(mcmc.get_extra_fields()["diverging"])) if "diverging" in mcmc.get_extra_fields() else -1
    print(f"[{label}] max r_hat {rhat:.3f}, divergences {div}")


def tables(post, groups: pd.Index, doms: pd.Index, cols: list[str], sems: pd.Index | None = None):
    """population.csv: mu per domain, sigma, delta; judges.csv: beta per (judge, domain)."""
    q = lambda a: np.percentile(a, [2.5, 50, 97.5], axis=0)  # noqa: E731
    flat = {k: np.asarray(v).reshape(-1, *v.shape[2:]) for k, v in post.items()}
    rows = []
    for i, dom in enumerate(doms):
        for j, c in enumerate(cols):
            lo, med, hi = q(flat["mu"][:, i, j])
            rows.append(dict(parameter="mu", domain=dom, coefficient=c, sd=SD.get(c, 1.0), mean=flat["mu"][:, i, j].mean(), lo=lo, median=med, hi=hi))
    for j, c in enumerate(cols):
        lo, med, hi = q(flat["sigma"][:, j])
        rows.append(dict(parameter="sigma", domain="", coefficient=c, mean=flat["sigma"][:, j].mean(), lo=lo, median=med, hi=hi))
    for j, c in enumerate([c for c in cols if c in SHOWN] if "delta" in flat else []):
        lo, med, hi = q(flat["delta"][:, j])
        rows.append(dict(parameter="delta (shown)", domain="", coefficient=c, mean=flat["delta"][:, j].mean(), lo=lo, median=med, hi=hi))
    for i, sem in enumerate(sems[1:] if sems is not None else []):
        for j, c in enumerate(cols):
            x = flat["semester"][:, i, j]
            lo, med, hi = q(x)
            rows.append(dict(parameter=f"semester offset vs {REF_SEMESTER}", domain=sem, coefficient=c, mean=x.mean(), lo=lo, median=med, hi=hi))
    pop = pd.DataFrame(rows)
    rows = []
    for g, name in enumerate(groups):
        judge, dom = name.split(" | ")
        for j, c in enumerate(cols):
            lo, med, hi = q(flat["beta"][:, g, j])
            rows.append(dict(judge=judge, domain=dom, coefficient=c, mean=flat["beta"][:, g, j].mean(), lo=lo, median=med, hi=hi))
    return pop, pd.DataFrame(rows)


def yticks(lo: float, hi: float) -> list[float]:
    """Both endpoints and 0, unless 0 sits too close to an endpoint to label."""
    ticks = [lo, hi]
    if lo < 0 < hi and min(-lo, hi) > 0.15 * (hi - lo):
        ticks.insert(1, 0)
    return ticks


def judge_semesters() -> tuple[dict[str, str], list[str]]:
    """model -> release semester (configs/semesters.yml), and the semesters in order."""
    roster = yaml.safe_load(Path("configs/semesters.yml").read_text())
    return {m: str(sem) for sem, models in roster.items() for m in models}, [str(k) for k in roster]


def figure(judges: pd.DataFrame, out: Path):
    """Rows = coefficient, columns = domain, x = the judge's release semester, colour = that semester:
    per-judge posterior mean and 95 % interval."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    semester, sems = judge_semesters()
    colour = dict(zip(sems, SEQ(np.linspace(0.15, 1, len(sems)))))
    judges = judges.assign(judge_sem=judges.judge.map(semester))
    fig, axes = plt.subplots(len(COEFS), len(DOMAINS), figsize=(4.4, 4.0), sharex=True)
    rng = np.random.default_rng(0)
    for r, c in enumerate(COEFS):
        jc = judges[judges.coefficient == c]
        lo, hi = np.floor(jc.lo.min() * 2) / 2, np.ceil(jc.hi.max() * 2) / 2
        for k, dom in enumerate(DOMAINS):
            ax, j = axes[r, k], jc[jc.domain == dom]
            ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
            x = pd.Index(sems).get_indexer(j.judge_sem) + rng.uniform(-0.25, 0.25, len(j))
            col = j.judge_sem.map(colour)
            ax.vlines(x, j.lo, j.hi, color=col, lw=0.6, alpha=0.7, zorder=3)
            ax.scatter(x, j["mean"], s=6, c=col, lw=0, zorder=4)
            ax.set_xticks(range(len(sems)), sems)
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_xlim(-0.6, len(sems) - 0.4)
            ax.set_ylim(lo, hi), ax.set_yticks(yticks(lo, hi))
            if r == 0:
                ax.text(0.5, 1.06, dom, transform=ax.transAxes, fontsize=7, color=MUTED, ha="center", va="bottom")
            if k == 0:
                ax.set_ylabel(c)
            style(ax)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_position(("outward", 3))
            if k > 0:  # the row shares its y axis: only the first column carries it
                ax.tick_params(labelleft=False, left=False)
                ax.spines["left"].set_visible(False)
    fig.supxlabel("Judge release semester", fontsize=8, y=0.095)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=colour[s], label=s) for s in sems]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=6, frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.3, w_pad=0.3)
    save(fig, out)


def figure_recent(judges: pd.DataFrame, out: Path, last: int = 3) -> pd.DataFrame:
    """Judges of the `last` release semesters only. One panel per coefficient, x = domain plus "all": per-judge
    posterior means (dots) and the mean +- t 95 % CI over judges (unit = judge x domain cell; "all" pools the
    domains). Returns the table of means."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    semester, sems = judge_semesters()
    keep = sems[-last:]
    j = judges[judges.judge.map(semester).isin(keep)]
    cats = [*DOMAINS, "all"]
    rows = []
    for c in COEFS:
        for cat in cats:
            x = j[(j.coefficient == c) & ((j.domain == cat) | (cat == "all"))]["mean"]
            n = len(x)
            h = student_t.ppf(0.975, n - 1) * x.std() / np.sqrt(n) if n > 1 else np.nan
            rows.append(dict(coefficient=c, domain=cat, n=n, mean=x.mean(), lo=x.mean() - h, hi=x.mean() + h))
    table = pd.DataFrame(rows)
    colour = {**dict(zip(DOMAINS, PALETTE)), "all": INK2}
    fig, axes = plt.subplots(1, len(COEFS), figsize=(4.4, 2.0))
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, COEFS):
        jc, t = j[j.coefficient == c], table[table.coefficient == c].set_index("domain")
        for i, cat in enumerate(cats):
            d = jc if cat == "all" else jc[jc.domain == cat]
            col = d.domain.map(colour)
            sig = (d.lo > 0) | (d.hi < 0)  # posterior 95 % interval excludes 0: filled, else open
            x = i + rng.uniform(-0.2, 0.2, len(d))
            ax.scatter(x[sig], d["mean"][sig], s=7, c=col[sig], lw=0, alpha=0.8, zorder=2)
            ax.scatter(x[~sig], d["mean"][~sig], s=7, facecolors="none", edgecolors=col[~sig], lw=0.6, zorder=2)
            ax.plot([i + 0.3, i + 0.3], [t.lo[cat], t.hi[cat]], color=colour[cat], lw=1, zorder=3)
            filled = t.lo[cat] > 0 or t.hi[cat] < 0
            ax.plot(i + 0.3, t["mean"][cat], "o", ms=3.5, mfc=colour[cat] if filled else "white", mec=colour[cat], mew=1, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(len(cats)), cats)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlim(-0.5, len(cats) - 0.3)
        lo, hi = np.floor(min(jc["mean"].min(), t.lo.min()) * 2) / 2, np.ceil(max(jc["mean"].max(), t.hi.max()) * 2) / 2
        ax.set_ylim(lo, hi), ax.set_yticks(yticks(lo, hi))
        ax.text(0.5, 1.04, c, transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="center", va="bottom")
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 4))
    axes[0].set_ylabel("Coefficient (logit per sd)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=INK2, label="95 % excludes 0"),
               plt.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=INK2, label="95 % covers 0")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.4)
    save(fig, out)
    return table


def main(fast: bool = False, no_hidden: bool = False):
    out = OUT / "no_hidden" if no_hidden else OUT
    out.mkdir(parents=True, exist_ok=True)
    d = design(no_hidden)
    groups, doms = pd.Index(sorted(d.group.unique())), pd.Index(DOMAINS)
    sems = pd.Index([REF_SEMESTER, *sorted(set(d.semester) - {REF_SEMESTER})]) if no_hidden else None
    print(f"{len(d)} verdicts, {len(groups)} (judge, domain) groups")
    print("covariate sd (standardisation):", {k: round(v, 3) for k, v in SD.items()})
    warmup, samples = (200, 200) if fast else (1000, 1000)

    # explained deviance on held-out items
    items = d[["domain", "bench", "item"]].drop_duplicates()
    test = set(map(tuple, items.sample(frac=0.2, random_state=0).to_numpy()))
    is_test = np.array([k in test for k in zip(d.domain, d.bench, d.item)])
    train, held = d[~is_test], d[is_test]
    rows = []
    for name, cols in VARIANTS.items():
        cols = ["position", *cols] if "position" not in cols else cols
        post, mcmc, args = fit(train, cols, groups, doms, warmup, samples, sems=sems)
        diagnostics(post, mcmc, f"{name} / train")
        p = predict(post, args, held, cols, groups)
        rows.append(dict(variant=name, columns=" + ".join(cols), n_test=len(held), log_loss=log_loss(held.y.to_numpy(), p),
                         accuracy=float(np.mean((p > 0.5) == (held.y.to_numpy() > 0.5)))))
    heldout = pd.DataFrame(rows)
    heldout["deviance_explained_vs_position"] = 1 - heldout.log_loss / heldout.log_loss.iloc[0]
    heldout.to_csv(out / "heldout.csv", index=False, float_format="%.4f")
    print(heldout.round(4).to_string(index=False))

    # the reported fit: every verdict
    post, mcmc, _ = fit(d, COEFS, groups, doms, warmup, samples, sems=sems)
    diagnostics(post, mcmc, "full / all")
    pop, judges = tables(post, groups, doms, COEFS, sems)
    pop.to_csv(out / "population.csv", index=False, float_format="%.3f")
    judges.to_csv(out / "judges.csv", index=False, float_format="%.3f")
    print(pop.round(2).to_string(index=False))
    figure(judges, out / "judge_glm")
    recent = figure_recent(judges, out / "judge_glm_recent")
    recent.to_csv(out / "recent.csv", index=False, float_format="%.3f")
    print(recent.round(2).to_string(index=False))
    if no_hidden:  # is the length slope a capability proxy? split it into between-pair and within-pair parts
        post, mcmc, _ = fit(d, DECOMP, groups, doms, warmup, samples, sems=sems)
        diagnostics(post, mcmc, "decomposition / all")
        pop, judges = tables(post, groups, doms, DECOMP, sems)
        pop.to_csv(out / "decomposition_population.csv", index=False, float_format="%.3f")
        judges.to_csv(out / "decomposition_judges.csv", index=False, float_format="%.3f")
        print(pop.round(2).to_string(index=False))


if __name__ == "__main__":
    main("--fast" in sys.argv, "--no-hidden" in sys.argv)
