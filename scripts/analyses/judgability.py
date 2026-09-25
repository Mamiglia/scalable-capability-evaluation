"""Which items are easiest to judge: per-item discrimination of pooled reference-vs-reference
pairwise verdicts, against the item's difficulty, discrimination and model accuracy.

Cell script (conventions as scripts/analyses/results.py). Reads logs/judge only; no transcripts.

    uv run python scripts/analyses/judgability.py
"""

# %% settings
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis import judgability, loading  # noqa: E402

DOMAINS = ["science", "math", "coding", "cyber"]
OUT = Path("analysis/results/judgability")
OUT.mkdir(parents=True, exist_ok=True)

# %% per-item table
items = pd.concat([judgability.table(loading.domain(d)) for d in DOMAINS], ignore_index=True)
items.to_csv(OUT / "items.csv", index=False, float_format="%.4f")
print(items.groupby("domain")[["n", "accuracy", "a_judge"]].describe().round(2).T)

# %% what predicts judgability: Spearman of accuracy with item properties, per domain
COVARIATES = ["b", "a", "guess", "model_acc", "mean_gap", "n"]
NAMES = {"b": "difficulty b", "a": "discrimination a", "guess": "guess", "model_acc": "model accuracy",
         "mean_gap": "mean |theta gap|", "n": "n verdicts"}
rho = items.groupby("domain").apply(lambda g: g[COVARIATES].corrwith(g.accuracy, method="spearman")).T.rename(index=NAMES)
rho["pooled"] = items[COVARIATES].corrwith(items.accuracy, method="spearman").rename(NAMES)
rho.to_csv(OUT / "spearman_accuracy.csv", float_format="%.3f")
print("Spearman rho with judge accuracy\n", rho.round(2))

# %% by bench
bench = items.groupby(["domain", "bench"]).agg(items=("item", "size"), accuracy=("accuracy", "median"),
                                                a_judge=("a_judge", "median"), b=("b", "median"), model_acc=("model_acc", "median"))
bench.to_csv(OUT / "bench.csv", float_format="%.3f")
print(bench.round(2))

# %% difficulty quartiles within domain: is judge accuracy flat across b?
items["b_quartile"] = items.groupby("domain").b.transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 easiest", "Q2", "Q3", "Q4 hardest"]))
print(items.pivot_table(index="domain", columns="b_quartile", values="accuracy", aggfunc="median", observed=True).round(3))

# %% is "easy to judge" an item property or noise? split-half over judges: per-item accuracy from
# each judge separately, Spearman between judges within a domain (pairs of judges on the same items)
verdicts = pd.concat([judgability.with_feature_gaps(judgability.reference_verdicts(dom), dom).assign(domain=dom.name)
                      for dom in map(loading.domain, DOMAINS)], ignore_index=True)
verdicts.to_csv(OUT / "verdicts.csv", index=False, float_format="%.4f")
per_judge = verdicts[verdicts.gap != 0].assign(hit=lambda v: (v.gap > 0) == (v.pick_left > 0.5)).groupby(
    ["domain", "bench", "item", "judge"]).hit.agg(["mean", "size"]).reset_index()
per_judge = per_judge[per_judge["size"] >= 20]
wide = per_judge.pivot_table(index=["domain", "bench", "item"], columns="judge", values="mean")
rel = []
for d, g in wide.groupby(level="domain"):
    g = g.dropna(axis=1, how="all")
    for i, j1 in enumerate(g.columns):
        for j2 in g.columns[i + 1:]:
            both = g[[j1, j2]].dropna()
            if len(both) >= 20:
                rel.append(dict(domain=d, judge_1=j1, judge_2=j2, items=len(both), rho=both[j1].corr(both[j2], method="spearman")))
rel = pd.DataFrame(rel)
rel.to_csv(OUT / "judge_agreement.csv", index=False, float_format="%.3f")
print(rel.round(2).to_string())
print(rel.groupby("domain").rho.median().round(2))

# %% does the judge follow transcript length, and is that what makes an item "easy"?
# verdict level: pick_left ~ theta gap + length gap (left - right), per domain
import statsmodels.api as sm  # noqa: E402

GAPS = ["gap", "d_log_chars"]
fits = []
for d, g in verdicts.dropna(subset=GAPS).groupby("domain"):
    m = sm.Logit(g.pick_left, sm.add_constant(g[GAPS])).fit(disp=0)
    fits.append(pd.Series(m.params, name=d))
print("verdict-level logit coefficients\n", pd.DataFrame(fits).round(2))
# item level: an item is "easy" when the stronger reference also wrote the longer transcript
per_item = verdicts.dropna(subset=GAPS).assign(
    hit=lambda v: (v.gap > 0) == (v.pick_left > 0.5),
    length_aligned=lambda v: np.sign(v.gap) * np.sign(v.d_log_chars)).groupby(["domain", "bench", "item"]).agg(
    accuracy=("hit", "mean"), length_aligned=("length_aligned", "mean"))
print("Spearman(item accuracy, share of pairs where the stronger model is longer)\n",
      per_item.groupby("domain").apply(lambda g: g.accuracy.corr(g.length_aligned, method="spearman")).round(2))

# %% figure: what the pairwise judge follows, and whether "easy to judge" is an item property
import matplotlib.pyplot as plt  # noqa: E402

from analysis.stats import DOMAIN_LABELS
from plots.results import DOMAIN_COLORS  # noqa: E402  (sets the rcParams)

FIG = Path("analysis/results/figures/judgability")
FIG.mkdir(parents=True, exist_ok=True)
MARKERS = {"science": "o", "math": "s", "coding": "^", "cyber": "D"}
PREDICTORS = {"gap": ("True ability gap", "#444444"), "d_log_chars": ("Length gap", "#bbbbbb")}

z = verdicts.dropna(subset=GAPS).copy()
z[GAPS] = z[GAPS] / z[GAPS].std()  # one sd of each gap, so the two slopes are comparable
def logit(g): return sm.Logit(g.pick_left, sm.add_constant(g[GAPS])).fit(disp=0).params[GAPS]
pooled = z.groupby("domain").apply(logit)
by_judge = z.groupby(["domain", "judge"]).filter(lambda g: len(g) >= 200).groupby(["domain", "judge"]).apply(logit)
same = rel_split = None  # filled below from the per-judge item accuracies
pj = verdicts[verdicts.gap != 0].assign(hit=lambda v: (v.gap > 0) == (v.pick_left > 0.5)).groupby(
    ["domain", "bench", "item", "judge", "semester"]).hit.agg(["mean", "size"]).reset_index().query("size >= 10")
rows = []
for d, g in pj.groupby("domain"):
    units = {k: u.set_index(["bench", "item"])["mean"] for k, u in g.groupby(["judge", "semester"])}
    keys = list(units)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            both = pd.concat([units[k1], units[k2]], axis=1, join="inner")
            if len(both) >= 20:
                rows.append(dict(domain=d, same=k1[1] == k2[1], rho=both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")))
agreement = pd.DataFrame(rows)
agreement.to_csv(OUT / "judge_agreement_by_semester.csv", index=False, float_format="%.3f")

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), gridspec_kw=dict(width_ratios=[1.2, 1, 0.9], wspace=0.7))
# A: standardised logit slopes, pooled bar + one circle per judge
ax = axes[0]
x = np.arange(len(DOMAINS))
for k, (p, (label, color)) in enumerate(PREDICTORS.items()):
    off = (k - 0.5) * 0.36
    ax.bar(x + off, pooled.loc[DOMAINS, p], width=0.34, color=color, label=label)
    for i, d in enumerate(DOMAINS):
        pts = by_judge.loc[d, p] if d in by_judge.index.get_level_values(0) else []
        ax.plot(np.full(len(pts), i + off), pts, "o", ms=2.5, mfc="none", mec="black", mew=0.6)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks(x, [DOMAIN_LABELS[d] for d in DOMAINS], rotation=90)
ax.set_ylabel("Logit slope per s.d.")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1)
ax.spines["bottom"].set_bounds(0, len(DOMAINS) - 1)
ax.set_yticks([0, 2, 4]); ax.spines["left"].set_bounds(0, 5)
# B: item accuracy vs share of pairs where the stronger reference is longer
ax = axes[1]
for d in DOMAINS:
    g = per_item.loc[d]
    ax.plot(g.length_aligned, g.accuracy, MARKERS[d], ms=2.5, mfc="none", mec=DOMAIN_COLORS[d], mew=0.6, label=DOMAIN_LABELS[d])
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Stronger model longer\n(share of pairs, −1 to 1)")
ax.set_ylabel("Judge accuracy on item")
ax.set_xticks([-1, 0, 1]); ax.set_yticks([0.2, 0.5, 0.8]); ax.set_ylim(0.1, 0.95)
ax.spines["bottom"].set_bounds(-1, 1); ax.spines["left"].set_bounds(0.2, 0.8)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
# C: judges agree on which items are easy only when they judged the same references
ax = axes[2]
rng = np.random.default_rng(0)
for d in DOMAINS:
    for s, xpos in ((True, 0), (False, 1)):
        vals = agreement.query("domain == @d and same == @s").rho
        ax.plot(xpos + rng.uniform(-0.15, 0.15, len(vals)), vals, MARKERS[d], ms=2.5, mfc="none", mec=DOMAIN_COLORS[d], mew=0.6)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks([0, 1], ["Same\nreferences", "Different\nreferences"])
ax.set_ylabel("Agreement between\njudges (ρ)")
ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_ylim(-0.6, 1.0)
ax.spines["bottom"].set_bounds(0, 1); ax.spines["left"].set_bounds(-0.5, 1)
for ext in ("png", "svg"):
    fig.savefig(FIG / f"judgability.{ext}", bbox_inches="tight")
print(FIG / "judgability.png")

# %% items ordered by judgeability, and what the order correlates with
# per-item transcript features: mean over the reference transcripts the judge saw on that item
item_len = verdicts.groupby(["domain", "bench", "item"]).agg(
    abs_length_gap=("d_log_chars", lambda s: s.abs().mean()), abs_theta_gap=("gap", lambda s: s.abs().mean()))
lengths = []
for dom in map(loading.domain, DOMAINS):
    for task in loading.tasks(dom):
        f = judgability.transcript_features(task.references, "visible")
        lengths.append(f.assign(domain=dom.name)[["domain", "bench", "item", "log_chars", "log_reasoning_chars", "log_tool_calls"]])
item_len = item_len.join(pd.concat(lengths).groupby(["domain", "bench", "item"]).mean())
ranked = items.merge(item_len.reset_index()).sort_values("accuracy", ascending=False)
ranked["rank"] = np.arange(1, len(ranked) + 1)
ranked.to_csv(OUT / "items_ranked.csv", index=False, float_format="%.4f")

FEATURES_OF_INTEREST = {"b": "item difficulty b", "a": "item discrimination a", "model_acc": "model accuracy on item",
                        "log_chars": "transcript length (log chars)", "log_reasoning_chars": "reasoning length (log chars)",
                        "log_tool_calls": "tool calls (log)", "abs_length_gap": "|length gap| between the pair",
                        "abs_theta_gap": "|theta gap| between the pair", "n": "verdicts on item"}
def spearman(g):
    return g[list(FEATURES_OF_INTEREST)].corrwith(g.accuracy, method="spearman")
corr = ranked.groupby("domain").apply(spearman).T
corr["pooled"] = spearman(ranked)
corr.index = corr.index.map(FEATURES_OF_INTEREST)
corr.to_csv(OUT / "rank_correlations.csv", float_format="%.3f")
print("Spearman of judge accuracy with item features\n", corr.round(2))
print("domain: Kruskal-Wallis on accuracy", [round(x, 3) for x in __import__("scipy").stats.kruskal(*[g.accuracy for _, g in ranked.groupby("domain")])])
# all features at once, domain as dummies, standardised: which features carry independent signal
import statsmodels.formula.api as smf  # noqa: E402
fit = smf.ols("accuracy ~ C(domain) + b + a + model_acc + log_chars + log_reasoning_chars + abs_length_gap + abs_theta_gap",
              data=ranked.assign(**{c: (ranked[c] - ranked[c].mean()) / ranked[c].std() for c in FEATURES_OF_INTEREST if c != "n"})).fit()
print(fit.summary().tables[1], "\nR2", round(fit.rsquared, 3))
print("\ntop 15\n", ranked.head(15)[["domain", "bench", "item", "accuracy", "a_judge", "b", "log_chars", "n"]].round(2).to_string(index=False))
print("\nbottom 15\n", ranked.tail(15)[["domain", "bench", "item", "accuracy", "a_judge", "b", "log_chars", "n"]].round(2).to_string(index=False))
