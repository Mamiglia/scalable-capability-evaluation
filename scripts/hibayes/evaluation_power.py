"""Answer the evaluation-method questions that the existing data can already settle.

The feature analysis measures per-comparison behaviour. That is the wrong unit for
judging the method as an evaluation tool: nobody ranks two models from one
comparison. The unit that matters is the pooled verdict over many comparisons.
This script computes that and several checks that the current data supports:

  A  pair-level accuracy   pool every comparison for a model pair, take the
                           majority, and ask whether the pair is ranked correctly
  B  power curve           how many comparisons a pair needs before the pooled
                           verdict is reliable, by capability gap
  C  level vs gap          within a fixed gap band, does absolute capability
                           matter? The saturation and small-gap explanations are
                           confounded in the raw data; holding gap fixed is the
                           only separation available without new runs
  D  length control        the length effect under equal-length transcripts,
                           where the length cue has been removed by construction
  E  judge profiles        pair-balanced accuracy at narrow and wide gaps
  F  panel checks          matched-pair comparisons of one judge and judge panels
  G  bias diversity        optional panel diagnostics from existing feature fits

The resampling unit is a question item. Repeated rows for the same item across
judges, presentation orders, and transcript conditions are pooled before the
bootstrap, so they are not treated as independent evidence.

Writes evaluation_power.json and the evaluation figure set next to the other figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting
from plotting import DARK, HIB, LIGHT, OUT, finish, style, write_json

CSV = HIB / "gpqa_correctness.csv"
SEED = 0

COLS = ["judge", "item", "model_pair", "left_model", "right_model",
        "left_capability", "right_capability", "transcript_condition",
        "capability_difference_major", "is_correct", "self_comparison",
        "major_is_left"]

GAP_BANDS = [(0.00, 0.05, "under 0.05"), (0.05, 0.10, "0.05 to 0.10"),
             (0.10, 0.20, "0.10 to 0.20"), (0.20, 1.00, "0.20 and above")]

out: dict[str, object] = {}


def band_of(gap):
    for lo, hi, name in GAP_BANDS:
        if lo <= gap < hi:
            return name
    return GAP_BANDS[-1][2]


def item_votes(frame: pd.DataFrame) -> np.ndarray:
    """One vote per question item after pooling repeated comparison rows."""
    share = frame.groupby("item", sort=False)["is_correct"].mean().to_numpy(float)
    return np.where(share > 0.5, 1.0, np.where(share < 0.5, 0.0, 0.5))


def majority_trials(values: np.ndarray, k: int, draws: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Bootstrap majority outcomes for `k` question-item votes."""
    sample = rng.choice(values, size=(draws, k), replace=True)
    means = sample.mean(axis=1)
    return np.where(means > 0.5, 1.0, np.where(means < 0.5, 0.0, 0.5))


def curve_point(trials: np.ndarray, k: int, max_observed: int) -> dict:
    lo, hi = np.quantile(trials, [0.025, 0.975])
    return {
        "k": int(k),
        "accuracy": float(trials.mean()),
        "interval_2.5%": float(lo),
        "interval_97.5%": float(hi),
        "extrapolated": bool(k > max_observed),
    }


def load():
    df = pd.read_csv(CSV, usecols=COLS, low_memory=False)
    df = df[df["self_comparison"] == 0].copy()
    # Both orders of a pair were run deliberately (include_reverse), so pooling
    # over a pair cancels the position bias that inflates one order and deflates
    # the other. Pair identity is therefore the unordered pair.
    df["gap"] = df["capability_difference_major"].astype(float)
    df["band"] = df["gap"].map(band_of)
    df["both_high"] = ((df["left_capability"] >= 0.80) &
                       (df["right_capability"] >= 0.80)).astype(int)
    df["level"] = (df[["left_capability", "right_capability"]].mean(axis=1))
    return df


# ---------------------------------------------------------------- A, B
def pair_level(df):
    """Pool repeated rows within items, then pool item votes for each pair."""
    g = df.groupby("model_pair")
    rows = []
    for pair, d in g:
        votes = item_votes(d)
        share = votes.mean()
        rows.append(dict(pair=pair, n_items=int(len(votes)), n_rows=int(len(d)),
                         share=float(share),
                         gap=float(d["gap"].iloc[0]),
                         level=float(d["level"].mean()),
                         band=d["band"].iloc[0],
                         correct=bool(share > 0.5)))
    p = pd.DataFrame(rows).sort_values("gap")
    out["pair_level"] = dict(
        n_pairs=int(len(p)),
        n_correct=int(p["correct"].sum()),
        accuracy=float(p["correct"].mean()),
        by_band={b: dict(pairs=int((p["band"] == b).sum()),
                         correct=int(p[p["band"] == b]["correct"].sum()),
                         accuracy=float(p[p["band"] == b]["correct"].mean()))
                 for b in p["band"].unique()},
        pairs=p.to_dict("records"))
    return p


def power_curve(df, sizes=(1, 3, 7, 15, 31, 63, 127, 255, 511, 1023), B=400):
    """Majority vote over `k` bootstrapped question items per model pair."""
    rng = np.random.default_rng(SEED)
    res = {}
    for lo, hi, band in GAP_BANDS:
        d = df[(df["gap"] >= lo) & (df["gap"] < hi)]
        if d.empty:
            continue
        pairs = [item_votes(group) for _, group in d.groupby("model_pair")]
        pairs = [values for values in pairs if len(values) >= 8]
        if not pairs:
            continue
        curve = []
        for k in sizes:
            trials = np.mean(
                [majority_trials(values, k, B, rng) for values in pairs], axis=0
            )
            curve.append(curve_point(trials, k, min(map(len, pairs))))
        res[band] = dict(n_pairs=len(pairs),
                         observed_items=[int(len(values)) for values in pairs],
                         curve=curve)
    out["power"] = res
    return res


# ---------------------------------------------------------------- C
def level_vs_gap(df):
    """Hold the gap fixed and vary absolute capability.

    In the raw data every near-saturated pair also has a small gap, so the two
    explanations for the failure at the top cannot be separated. Within a single
    gap band they can be compared directly, which is as far as this data goes.
    """
    d = df[df["gap"] < 0.05].copy()
    if d.empty:
        return {}
    med = d["level"].median()
    d["level_group"] = np.where(d["level"] >= med, "higher-capability half",
                                "lower-capability half")
    pp_rows = []
    for (pair, group), frame in d.groupby(["model_pair", "level_group"]):
        votes = item_votes(frame)
        pp_rows.append({
            "model_pair": pair, "level_group": group,
            "n_items": int(len(votes)), "share": float(votes.mean()),
            "gap": float(frame["gap"].iloc[0]),
            "level": float(frame["level"].mean()),
        })
    pp = pd.DataFrame(pp_rows)
    tab = pp.groupby("level_group").agg(
        pairs=("model_pair", "size"), accuracy=("share", "mean"),
        mean_gap=("gap", "mean"), mean_level=("level", "mean")).reset_index()
    res = dict(median_level=float(med),
               groups=tab.to_dict("records"),
               pairs=pp.sort_values("level").to_dict("records"))
    out["level_vs_gap"] = res
    return res


# ---------------------------------------------------------------- D
def length_control(df):
    """Compare conditions on the same pair, judge, item, and presentation order."""
    key = ["model_pair", "judge", "item", "major_is_left"]
    cells = (df.groupby(key + ["transcript_condition"], as_index=False)
               ["is_correct"].mean())
    wide = cells.pivot(index=key, columns="transcript_condition", values="is_correct")
    conditions = sorted(df["transcript_condition"].unique())
    matched = wide.dropna(subset=conditions)
    matched_rows = [
        {"condition": condition, "n_cells": int(len(matched)),
         "accuracy": float(matched[condition].mean())}
        for condition in conditions
    ]
    res = dict(
        matched_cells=matched_rows,
        n_matched_cells=int(len(matched)),
        n_matched_pairs=int(matched.reset_index()["model_pair"].nunique()),
        match_key=key,
    )
    out["length_control"] = res
    return res


# ---------------------------------------------------------------- figures
def fig_power(power, c):
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    cols = ["s4", "s3", "s1", "s2"]
    for (lo, hi, band), col in zip(GAP_BANDS, cols):
        if band not in power:
            continue
        cur = power[band]["curve"]
        ks = [p["k"] for p in cur]
        acc = [p["accuracy"] for p in cur]
        lo_q = [p["interval_2.5%"] for p in cur]
        hi_q = [p["interval_97.5%"] for p in cur]
        ax.fill_between(ks, lo_q, hi_q, color=c[col], alpha=0.12, linewidth=0)
        ax.plot(ks, acc, marker="o", ms=5, lw=2, color=c[col],
                label=f"gap {band}  ({power[band]['n_pairs']} pairs)")
        first_extrapolated = next((p["k"] for p in cur if p["extrapolated"]), None)
        if first_extrapolated is not None:
            start = max(0, ks.index(first_extrapolated) - 1)
            ax.plot(ks[start:], acc[start:], lw=2, ls=(0, (4, 3)), color=c[col])
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (1.05, 0.505), color=c["crit"], fontsize=8.8, va="bottom")
    ax.axhline(0.95, color=c["axis"], lw=1.2)
    ax.annotate("95% reliable", (1.05, 0.955), color=c["muted"], fontsize=8.8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled per model pair", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("How many question items a pair needs before the verdict is trustworthy",
                 color=c["ink"], loc="left")
    ax.annotate("Bands are 95% item-bootstrap intervals. Dashed segments exceed the\n"
                "smallest observed item count in that capability-gap group.",
                (0.015, 0.02), xycoords="axes fraction", fontsize=8.4,
                color=c["muted"], linespacing=1.5)
    ax.set_ylim(0, 1.02)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig14_power", c)


def fig_pairs(p, c):
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ok = p[p["correct"]]
    bad = p[~p["correct"]]
    ax.scatter(ok["gap"], ok["share"], s=46, color=c["s1"], zorder=3,
               edgecolor=c["surface"], linewidth=1, label="pair ranked correctly")
    ax.scatter(bad["gap"], bad["share"], s=52, color=c["crit"], zorder=3,
               marker="X", edgecolor=c["surface"], linewidth=1,
               label="pair ranked WRONG")
    ax.axhline(0.5, color=c["axis"], lw=1.4)
    ax.set_xlabel("capability gap between the two models", color=c["ink2"])
    ax.set_ylabel("share of question-item votes favouring the stronger model", color=c["ink2"])
    ax.set_title("Pooling every comparison for a pair: where the method holds and where it breaks",
                 color=c["ink"], loc="left")
    ax.annotate("above this line the pooled verdict is right",
                (0.995, 0.52), xycoords=("axes fraction", "data"), ha="right",
                fontsize=8.8, color=c["muted"])
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig15_pairs", c)


def fig_length(lc, c):
    order = ["truncated_10", "truncated_50", "truncated_90", "raw"]
    lab = {"truncated_10": "10% of the\nshorter side", "truncated_50": "50%",
           "truncated_90": "90%", "raw": "full transcripts\n(unequal length)"}
    rows = {r["condition"]: r for r in lc["matched_cells"]}
    xs, ys = [], []
    for o in order:
        if o in rows:
            xs.append(lab[o]); ys.append(rows[o]["accuracy"])
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    cols = [c["s1"]] * len(xs)
    if xs:
        cols[-1] = c["s2"]
    ax.bar(range(len(xs)), ys, color=cols, width=.6)
    for i, v in enumerate(ys):
        ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9.6, color=c["ink"], family="monospace")
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (len(xs) - .45, 0.505), color=c["crit"], fontsize=8.8, va="bottom")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=9.5, color=c["ink2"])
    ax.set_ylim(0, max(ys + [0.7]) * 1.18)
    ax.set_ylabel("accuracy at ranking the pair", color=c["ink2"])
    ax.set_title("Accuracy across transcript-length conditions on matched cells",
                 color=c["ink"], loc="left")
    ax.annotate(f"Same {lc['n_matched_cells']:,} pair/judge/item/order cells "
                f"from {lc['n_matched_pairs']} model pairs in every condition. "
                "The truncated conditions also remove the final answer,\n"
                "so this is length-equalisation and answer-removal together, not length alone.",
                (0, -0.20), xycoords="axes fraction", fontsize=8.6, color=c["muted"],
                linespacing=1.6)
    ax.grid(axis="y", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "fig16_length_control", c)


def noise_vs_bias_by_gap(df, sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300):
    """Repeat the three pooling levels at narrow and wide capability gaps."""
    res = {}
    for label, sub in [("narrow gaps (under 0.10)", df[df["gap"] < 0.10]),
                       ("wide gaps (0.10 and above)", df[df["gap"] >= 0.10])]:
        levels = _three_levels(sub, sizes, B)
        n_pairs = sub["model_pair"].nunique()
        res[label] = dict(n_pairs=int(n_pairs), levels=levels)
    out["noise_vs_bias_by_gap"] = res
    return res


def fig_noise_by_gap(res, c):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)
    style_map = [("one judge, one order", "s4", "o"),
                 ("one judge, both orders", "s3", "s"),
                 ("all judges, both orders", "s1", "D")]
    for ax, (label, d) in zip(axes, res.items()):
        for name, col, mk in style_map:
            cur = d["levels"][name]
            ax.plot([p["k"] for p in cur], [p["accuracy"] for p in cur], marker=mk,
                    ms=5.5, lw=2, color=c[col], label=name)
        ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
        ax.set_xscale("log")
        ax.set_title(f"{label}  ({d['n_pairs']} pairs)", color=c["ink"], loc="left")
        ax.set_xlabel("question items pooled", color=c["ink2"])
        ax.grid(lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    axes[0].annotate("chance", (1.05, 0.505), color=c["crit"], fontsize=8.6, va="bottom")
    axes[0].set_ylim(0.45, 1.02)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.suptitle("Does a panel of judges help where single judges disagree most?",
                 color=c["ink"], fontsize=13.5, fontweight="semibold",
                 x=0.005, ha="left", y=1.03)
    fig.tight_layout()
    finish(fig, "fig19_panel_by_gap", c)


def panel_on_covered_pairs(df, min_judges=5, sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300):
    """Repeat the three-level decomposition where a panel actually exists."""
    per_pair = df.groupby("model_pair")["judge"].nunique()
    keep = set(per_pair[per_pair >= min_judges].index)
    sub = df[df["model_pair"].isin(keep)]
    levels = _three_levels(sub, sizes, B)
    out["panel_covered"] = dict(min_judges=int(min_judges),
                                n_pairs=int(len(keep)),
                                judges_per_pair=(int(per_pair[per_pair >= min_judges].min())
                                                 if keep else 0),
                                levels=levels)
    return levels


def solo_judges_vs_panel(df, sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300):
    """Each judge on its own against the panel, on the pairs that judge covers.

    Per-pair selection was unusable: a judge's average score is dominated by
    which pairs it was assigned, and the selected judge often had no data on the
    held-out pair. Fixing one judge for every pair removes both problems. The
    panel is then recomputed on exactly the pairs that judge covers, so the two
    curves describe the same comparisons.

    The best judge here is still a maximum over ten candidates, so it is
    optimistic. It is one maximum over ten, not a fresh maximum per pair, which
    is far milder than the oracle.
    """
    rng = np.random.default_rng(SEED)

    def majority(arr, k):
        return float(majority_trials(arr, k, B, rng).mean())

    per = {(p, j): item_votes(g)
           for (p, j), g in df.groupby(["model_pair", "judge"])
           if g["item"].nunique() >= 8}
    allrows = {p: item_votes(g) for p, g in df.groupby("model_pair")}

    res = {}
    for j in sorted(df["judge"].unique()):
        pairs = [p for (p, jj) in per if jj == j]
        if len(pairs) < 4:
            continue
        solo, panel = [], []
        for k in sizes:
            solo.append(dict(k=int(k),
                             accuracy=float(np.mean([majority(per[(p, j)], k) for p in pairs]))))
            panel.append(dict(k=int(k),
                              accuracy=float(np.mean([majority(allrows[p], k) for p in pairs]))))
        res[j] = dict(n_pairs=len(pairs), solo=solo, panel_same_pairs=panel)
    best = max(res, key=lambda j: res[j]["solo"][-1]["accuracy"]) if res else None
    out["solo_vs_panel"] = dict(best=best, judges=res)
    return res, best


def fig_solo_vs_panel(res, best, c):
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for j, d in res.items():
        if j == best:
            continue
        ax.plot([p["k"] for p in d["solo"]], [p["accuracy"] for p in d["solo"]],
                lw=1.3, color=c["muted"], alpha=.55, zorder=2)
    d = res[best]
    ax.plot([p["k"] for p in d["solo"]], [p["accuracy"] for p in d["solo"]],
            marker="s", ms=6, lw=2.4, color=c["s3"], zorder=4,
            label=f"best single judge: {best}  ({d['n_pairs']} pairs)")
    ax.plot([p["k"] for p in d["panel_same_pairs"]],
            [p["accuracy"] for p in d["panel_same_pairs"]],
            marker="D", ms=6, lw=2.4, color=c["s1"], zorder=4,
            label="all judges pooled, same pairs")
    ax.plot([], [], lw=1.3, color=c["muted"], alpha=.55,
            label=f"the other {max(0, len(res) - 1)} judges, alone")
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (1.05, 0.505), color=c["crit"], fontsize=8.8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled per model pair", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("One fixed judge against the whole panel, on the same pairs",
                 color=c["ink"], loc="left")
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig20_solo_vs_panel", c)


def feature_diversity(M):
    """Which features do the judges agree on, and which split them?

    M is judges x features. For each feature the mean across judges is the
    shared pull, and the spread across judges is how much they differ. A feature
    with a large mean and a small spread is a bias every judge has, which no
    panel can cancel. A feature with a small mean and a large spread is where
    ensembling has something to work with.
    """
    # Spread alone is the wrong measure: a coefficient with a large mean has a
    # large standard deviation by scale, so ranking on spread just re-ranks the
    # biggest effects. What decides whether pooling can cancel a bias is whether
    # judges disagree about its DIRECTION. `disagree` is the share of judges on
    # the minority side of zero: 0 means unanimous and nothing cancels, 0.5 means
    # an even split and the feature averages away entirely.
    mean = M.mean(axis=1)
    sd = M.std(axis=1)
    pos = (M > 0).sum(axis=1)
    n = M.notna().sum(axis=1)
    disagree = np.minimum(pos, n - pos) / n
    t = pd.DataFrame(dict(feature=M.index, shared=mean.values, spread=sd.values,
                          disagree=disagree.values))
    t["family"] = [str(f).split("__")[1] if "__" in str(f) else "" for f in t["feature"]]
    t["ratio"] = t["shared"].abs() / t["spread"].replace(0, np.nan)
    res = dict(n_features=int(len(t)),
               mean_shared_abs=float(t["shared"].abs().mean()),
               mean_spread=float(t["spread"].mean()),
               most_shared=t.reindex(t["shared"].abs().sort_values().index).tail(12)
                            .iloc[::-1].to_dict("records"),
               most_divergent=t.reindex(t["disagree"].sort_values().index).tail(12)
                               .iloc[::-1].to_dict("records"),
               by_family=t.groupby("family").agg(
                   n=("feature", "size"), shared=("shared", lambda s: float(s.abs().mean())),
                   spread=("spread", "mean")).reset_index().to_dict("records"))
    out["feature_diversity"] = res
    return t


def fig_feature_diversity(t, c):
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    x = t["shared"].abs(); y = t["disagree"]
    ax.scatter(x, y, s=26, color=c["muted"], alpha=.5, edgecolor="none", zorder=2)
    hi_shared = t.reindex(x.sort_values().index).tail(8)
    hi_spread = t.reindex(y.sort_values().index).tail(8)  # most sign-disagreement
    ax.scatter(hi_shared["shared"].abs(), hi_shared["disagree"], s=52, color=c["s2"],
               zorder=4, edgecolor=c["surface"], linewidth=1, label="strongest shared pull")
    ax.scatter(hi_spread["shared"].abs(), hi_spread["disagree"], s=52, color=c["s1"],
               zorder=4, edgecolor=c["surface"], linewidth=1, label="judges differ most")
    for _, r in hi_spread.iterrows():
        ax.annotate(str(r["feature"]).split("__")[-1][:24], (abs(r["shared"]), r["disagree"]),
                    xytext=(6, 3), textcoords="offset points", fontsize=7.4, color=c["ink2"])
    ax.set_xlabel("shared pull  |mean coefficient across judges|", color=c["ink2"])
    ax.set_ylabel("share of judges on the minority side of zero", color=c["ink2"])
    ax.set_title("Which features every judge shares, and which split them",
                 color=c["ink"], loc="left")
    ax.annotate("high here, low left:\nensembling can help",
                (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8.6,
                color=c["s1"], linespacing=1.5)
    ax.annotate("right and low:\nbias no panel can cancel",
                (0.98, 0.03), xycoords="axes fraction", ha="right", fontsize=8.6,
                color=c["s2"], linespacing=1.5)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig26_feature_diversity", c)


def fig_bias_sources(t, res, c):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax = axes[0]
    groups = [("different model-name\nfamily", t[t["same_family"] == 0]["corr"].values, "s1"),
              ("same model-name\nfamily", t[t["same_family"] == 1]["corr"].values, "s2")]
    for i, (lab, vals, col) in enumerate(groups):
        if not len(vals):
            continue
        jit = (np.arange(len(vals)) % 7 - 3) * 0.028
        ax.scatter(np.full(len(vals), i) + jit, vals, s=42, color=c[col], alpha=.75,
                   edgecolor=c["surface"], linewidth=.8, zorder=3)
        ax.plot([i - .22, i + .22], [vals.mean()] * 2, lw=2.4, color=c["ink"], zorder=4)
        ax.annotate(f"mean {vals.mean():.3f}", (i, vals.mean()), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9, color=c["ink"])
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups], color=c["ink2"])
    ax.set_ylabel("bias correlation between two judges", color=c["ink2"])
    ax.set_title("Does sharing a model-name family explain it?", color=c["ink"], loc="left")
    ax.grid(axis="y", lw=0.8); ax.set_axisbelow(True)
    same_mean = res["same_family"]["mean"]
    cross_mean = res["cross_family"]["mean"]
    if same_mean is not None and cross_mean is not None:
        note = f"difference {same_mean - cross_mean:+.3f}"
    else:
        note = "insufficient coverage for this comparison"
    ax.annotate(note, (0.5, 0.03), xycoords="axes fraction", ha="center",
                fontsize=8.8, color=c["muted"])

    ax = axes[1]
    h = t.dropna(subset=["cap_diff"])
    ax.scatter(h["cap_diff"], h["corr"], s=46, color=c["s3"], alpha=.8,
               edgecolor=c["surface"], linewidth=.8)
    ax.set_xlabel("difference in the two judges' own benchmark accuracy", color=c["ink2"])
    ax.set_ylabel("bias correlation", color=c["ink2"])
    r = res["capability"]["corr_with_capdiff"]
    label = f"r = {r:+.3f}" if r is not None else "capability data unavailable"
    ax.set_title(f"Does capability explain it?  {label}", color=c["ink"], loc="left")
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    fig.suptitle("Observed judge-bias similarity by model-name family and capability",
                 color=c["ink"], fontsize=13, fontweight="semibold", x=0.005, ha="left", y=1.03)
    fig.tight_layout()
    finish(fig, "fig24_bias_sources", c)


def fig_diverse_ensemble(e, c):
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    spec = [("diverse", "s1", "D", "panel picked to disagree"),
            ("similar", "s2", "o", "panel picked to agree")]
    for kind, col, mk, lab in spec:
        v = e.get(kind, {})
        if "ensemble" not in v:
            continue
        ax.plot([p["k"] for p in v["ensemble"]], [p["accuracy"] for p in v["ensemble"]],
                marker=mk, ms=6, lw=2.2, color=c[col],
                label=f"{lab} (internal corr {v['mean_corr']:.2f}, {v['pairs']} pairs)")
        ax.plot([p["k"] for p in v["all_judges"]], [p["accuracy"] for p in v["all_judges"]],
                lw=1.6, ls=(0, (4, 3)), color=c[col], alpha=.6,
                label=f"all {e['n_available_judges']} judges on those same pairs")
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled per model pair", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("A panel chosen to disagree, against one chosen to agree",
                 color=c["ink"], loc="left")
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.6, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig25_diverse_ensemble", c)


def _family(name):
    n = str(name).lower()
    for f in ("gemini", "gemma", "gpt", "qwen", "claude", "llama", "mistral"):
        if n.startswith(f):
            return f
    return n.split("-")[0]


def bias_similarity_sources(corr, df):
    """Is judge bias similarity explained by model family, or by capability?

    Two candidate explanations for judges being biased alike: they come from the
    same model-name family, or they sit at a similar capability level. Both are testable
    against the correlation matrix.
    """
    caps = {}
    for side in ("left", "right"):
        for m, c in zip(df[f"{side}_model"], df[f"{side}_capability"]):
            caps.setdefault(m, float(c))
    js = [j for j in corr.columns]
    rows = []
    for i, a in enumerate(js):
        for b in js[i + 1:]:
            r = float(corr.loc[a, b])
            if not np.isfinite(r):
                continue
            rows.append(dict(a=a, b=b, corr=r,
                             same_family=int(_family(a) == _family(b)),
                             cap_a=caps.get(a), cap_b=caps.get(b),
                             cap_diff=(abs(caps[a] - caps[b])
                                       if a in caps and b in caps else None)))
    t = pd.DataFrame(rows)
    same = t[t["same_family"] == 1]["corr"]
    diff = t[t["same_family"] == 0]["corr"]
    have = t.dropna(subset=["cap_diff"])
    res = dict(
        n_pairs=int(len(t)),
        same_family=dict(n=int(len(same)), mean=float(same.mean()) if len(same) else None),
        cross_family=dict(n=int(len(diff)), mean=float(diff.mean()) if len(diff) else None),
        capability=dict(
            n=int(len(have)),
            corr_with_capdiff=(float(np.corrcoef(have["cap_diff"], have["corr"])[0, 1])
                               if len(have) > 3 else None)),
        pairs=t.to_dict("records"))
    out["bias_similarity_sources"] = res
    return t, res


def diverse_ensemble(df, corr, sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300,
                     n_members=4):
    """Does a deliberately anti-correlated panel beat pooling everyone?

    Members are chosen greedily to minimise mean pairwise bias correlation. Two
    controls: the most-similar panel of the same size, and the full pool. All
    three are scored on the pairs every member of the diverse panel covers, so
    they describe the same comparisons.
    """
    rng = np.random.default_rng(SEED)
    js = list(corr.columns)

    def pick(minimise):
        chosen = [max(js, key=lambda j: corr[j].mean()) if not minimise
                  else min(js, key=lambda j: corr[j].mean())]
        while len(chosen) < n_members:
            rest = [j for j in js if j not in chosen]
            score = {j: np.mean([corr.loc[j, c] for c in chosen]) for j in rest}
            chosen.append(min(score, key=score.get) if minimise else max(score, key=score.get))
        return chosen

    diverse, similar = pick(True), pick(False)

    def curves(members):
        sub = df[df["judge"].isin(members)]
        cov = sub.groupby("model_pair")["judge"].nunique()
        pairs = [p for p in cov[cov == len(members)].index]
        if len(pairs) < 3:
            pairs = [p for p in cov[cov >= max(2, len(members) - 1)].index]
        if len(pairs) < 3:
            return None
        ens = {p: item_votes(sub[sub["model_pair"] == p]) for p in pairs}
        full = {p: item_votes(df[df["model_pair"] == p]) for p in pairs}

        def majority(arr, k):
            return float(majority_trials(arr, k, B, rng).mean())
        return dict(
            pairs=len(pairs),
            ensemble=[dict(k=int(k), accuracy=float(np.mean([majority(ens[p], k) for p in pairs])))
                      for k in sizes],
            all_judges=[dict(k=int(k), accuracy=float(np.mean([majority(full[p], k) for p in pairs])))
                        for k in sizes])

    res = dict(n_members=n_members, n_available_judges=len(js),
               diverse=dict(members=diverse,
                            mean_corr=float(np.mean([corr.loc[a, b] for i, a in enumerate(diverse)
                                                     for b in diverse[i + 1:]])),
                            **(curves(diverse) or {})),
               similar=dict(members=similar,
                            mean_corr=float(np.mean([corr.loc[a, b] for i, a in enumerate(similar)
                                                     for b in similar[i + 1:]])),
                            **(curves(similar) or {})))
    out["diverse_ensemble"] = res
    return res


def judge_bias_similarity():
    """How alike are the judges' biases?

    Pooling cancels an error only when the things pooled err in different
    directions. That predicts the panel result directly: if the judges' feature
    biases are strongly correlated, a panel cannot cancel them and will land on
    the average judge, which is what figure 18 shows. This measures the
    correlation rather than inferring it.

    Each judge becomes a vector of its coefficients across the 136 one-feature
    fits in features_within, then every pair of judges is correlated.
    """
    rows = {}
    base = HIB / "features_within"
    for d in sorted(base.glob("marginal__*/communicate/judge_feature_effects.csv")):
        feature = d.parent.parent.name
        t = pd.read_csv(d)
        required = {"judge", "feature", "mean_log_odds"}
        if not required <= set(t.columns):
            raise ValueError(f"{d} is missing columns {sorted(required - set(t.columns))}")
        # Marginal reports also repeat the intercept, position control, and
        # condition contrasts in every fit. The judge-bias vector must contain
        # only the transcript feature fitted by that directory.
        t = t[t["feature"].astype(str).str.startswith("d_")]
        for _, r in t.iterrows():
            rows.setdefault(r["judge"], {})[f"{feature}::{r['feature']}"] = float(r["mean_log_odds"])
    if len(rows) < 2:
        return None
    M = pd.DataFrame(rows).dropna()
    if M.empty:
        return None
    corr = M.corr()
    off = corr.where(~np.eye(len(corr), dtype=bool))
    vals = off.stack().values
    if not len(vals):
        return None
    res = dict(n_judges=int(len(corr)), n_features=int(len(M)),
               mean_corr=float(np.mean(vals)), min_corr=float(np.min(vals)),
               max_corr=float(np.max(vals)),
               share_above_half=float(np.mean(vals > 0.5)),
               matrix=corr.round(3).to_dict())
    out["judge_bias_similarity"] = res
    return corr, res, M


def fig_bias_similarity(corr, res, c):
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    order = corr.mean(axis=1).sort_values(ascending=False).index
    M = corr.loc[order, order].values
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8.6, color=c["ink2"])
    ax.set_yticklabels(order, fontsize=8.6, color=c["ink2"])
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if M[i, j] > 0.55 else c["ink"])
    ax.set_title("How alike are the judges' biases?", color=c["ink"], loc="left")
    ax.annotate(f"Correlation between judges across {res['n_features']} one-feature coefficients. "
                f"Mean {res['mean_corr']:.2f}, range {res['min_corr']:.2f} to {res['max_corr']:.2f}.\n"
                "High values mean the judges are pulled the same way, so pooling them cannot cancel it.",
                (0, -0.30), xycoords="axes fraction", fontsize=8.6, color=c["muted"],
                linespacing=1.7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=.72, label="correlation")
    fig.tight_layout()
    finish(fig, "fig23_bias_similarity", c)


def _solo_best(df, sizes, B, rng, min_pairs=3):
    """Best single judge and the panel scored on that judge's pairs."""
    def majority(arr, k):
        return float(majority_trials(arr, k, B, rng).mean())

    per = {(p, j): item_votes(g)
           for (p, j), g in df.groupby(["model_pair", "judge"])
           if g["item"].nunique() >= 8}
    allrows = {p: item_votes(g) for p, g in df.groupby("model_pair")}
    cand = {}
    for j in sorted(df["judge"].unique()):
        pairs = [p for (p, jj) in per if jj == j]
        if len(pairs) < min_pairs:
            continue
        solo = [dict(k=int(k), accuracy=float(np.mean([majority(per[(p, j)], k) for p in pairs])))
                for k in sizes]
        panel = [dict(k=int(k), accuracy=float(np.mean([majority(allrows[p], k) for p in pairs])))
                 for k in sizes]
        cand[j] = dict(n_pairs=len(pairs), solo=solo, panel=panel)
    if not cand:
        return None
    best = max(cand, key=lambda j: cand[j]["solo"][-1]["accuracy"])
    return dict(best=best, n_pairs=cand[best]["n_pairs"],
                solo=cand[best]["solo"], panel=cand[best]["panel"],
                n_candidates=len(cand))


def solo_best_by_subset(df, sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300):
    """Best single judge against the whole panel, split by condition and by gap.

    The panel is always rescored on the pairs the winning judge covers, so the
    two curves in a panel describe the same comparisons. The winner is chosen
    inside each subset, so it is a maximum over the judges available there and
    is optimistic by the usual amount; the winner's name is reported so the
    reader can see whether it is stable across subsets.
    """
    rng = np.random.default_rng(SEED)
    # Only pairs that appear in every condition, otherwise the four panels are
    # drawn from different pair sets of different difficulty and cannot be read
    # against each other.
    ncond = df["transcript_condition"].nunique()
    have = df.groupby("model_pair")["transcript_condition"].nunique()
    matched = set(have[have == ncond].index)
    m = df[df["model_pair"].isin(matched)]
    conds = [("full transcripts (unequal length)", m[m["transcript_condition"] == "raw"]),
             ("cut to 10% of the shorter", m[m["transcript_condition"] == "truncated_10"]),
             ("cut to 50%", m[m["transcript_condition"] == "truncated_50"]),
             ("cut to 90%", m[m["transcript_condition"] == "truncated_90"])]
    gaps = [(f"gap {lab}", df[(df["gap"] >= lo) & (df["gap"] < hi)])
            for lo, hi, lab in GAP_BANDS]
    res = {"by_length": {}, "by_gap": {}}
    for key, items in (("by_length", conds), ("by_gap", gaps)):
        for label, sub in items:
            if sub.empty:
                continue
            r = _solo_best(sub, sizes, B, rng)
            if r:
                res[key][label] = r
    out["solo_best_by_subset"] = res
    return res


def fig_solo_facets(res, key, title, fname, c):
    items = list(res[key].items())
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(3.35 * len(items), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, r) in zip(axes, items):
        ax.plot([p["k"] for p in r["panel"]], [p["accuracy"] for p in r["panel"]],
                marker="D", ms=5, lw=2, color=c["s1"], label="all judges pooled")
        ax.plot([p["k"] for p in r["solo"]], [p["accuracy"] for p in r["solo"]],
                marker="s", ms=5, lw=2, color=c["s3"], label="best single judge")
        ax.axhline(0.5, color=c["crit"], lw=1.3, ls=(0, (5, 3)))
        ax.set_xscale("log")
        ax.set_title(label, color=c["ink"], loc="left", fontsize=10.5)
        ax.annotate(f"{r['best']}\n{r['n_pairs']} pairs", (0.04, 0.955),
                    xycoords="axes fraction", va="top", fontsize=8.2,
                    color=c["muted"], linespacing=1.5)
        ax.set_xlabel("question items pooled", color=c["ink2"], fontsize=9.5)
        ax.grid(lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("chance the pooled verdict is right", color=c["ink2"], fontsize=9.5)
    axes[0].set_ylim(0.42, 1.03)
    axes[-1].legend(frameon=False, fontsize=8.8, loc="lower right", labelcolor=c["ink2"])
    fig.suptitle(title, color=c["ink"], fontsize=13, fontweight="semibold",
                 x=0.005, ha="left", y=1.04)
    fig.tight_layout()
    finish(fig, fname, c)


def _three_levels(df, sizes, B):
    """Shared machinery for the noise-versus-bias decomposition."""
    rng = np.random.default_rng(SEED)
    levels = {"one judge, one order": [], "one judge, both orders": [],
              "all judges, both orders": []}
    cells = {}
    for pair, d in df.groupby("model_pair"):
        by_judge_order, by_judge = {}, {}
        for (j, mil), g in d.groupby(["judge", "major_is_left"]):
            if g["item"].nunique() >= 8:
                by_judge_order[(j, mil)] = item_votes(g)
        for j, g in d.groupby("judge"):
            if g["item"].nunique() >= 8:
                by_judge[j] = item_votes(g)
        if by_judge_order and by_judge and d["item"].nunique() >= 8:
            cells[pair] = (list(by_judge_order.values()), list(by_judge.values()),
                           item_votes(d))

    def majority(arr, k):
        return majority_trials(arr, k, B, rng).sum()

    for k in sizes:
        acc = {name: [] for name in levels}
        for pair, (jo, jj, allrows) in cells.items():
            acc["one judge, one order"].append(float(np.mean([majority(a, k) / B for a in jo])))
            acc["one judge, both orders"].append(float(np.mean([majority(a, k) / B for a in jj])))
            acc["all judges, both orders"].append(majority(allrows, k) / B)
        for name in levels:
            levels[name].append(dict(k=int(k), accuracy=float(np.mean(acc[name]))))
    return levels


def noise_vs_bias(df, sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300):
    """Does pooling cancel noise, or does it cancel bias?

    Pooling more comparisons always averages away random error. It only removes a
    systematic error if the things being pooled are biased in different
    directions. Three nested levels separate the two:

      one judge, one order    only item-to-item noise cancels
      one judge, both orders  that judge's position bias also cancels
      all judges, both orders the judges' differing biases also cancel

    Every level draws the same k comparisons, so sample size is held fixed and
    any gap between the curves is cancelled bias rather than reduced noise. This
    is the panel-of-judges question: it asks what a panel buys beyond running one
    judge k times.
    """
    levels = _three_levels(df, sizes, B)
    n_pairs = int(df["model_pair"].nunique())
    out["noise_vs_bias"] = dict(n_pairs=n_pairs, levels=levels)
    return levels


def fig_noise_vs_bias(levels, c):
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    style_map = [("one judge, one order", "s4", "o", "-"),
                 ("one judge, both orders", "s3", "s", "-"),
                 ("all judges, both orders", "s1", "D", "-")]
    for name, col, mk, ls in style_map:
        cur = levels[name]
        ax.plot([p["k"] for p in cur], [p["accuracy"] for p in cur], marker=mk,
                ms=6, lw=2, ls=ls, color=c[col], label=name)
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (1.05, 0.505), color=c["crit"], fontsize=8.8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled (same number at every level)", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("What a panel of judges buys that one judge cannot",
                 color=c["ink"], loc="left")
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.2, loc="lower right", labelcolor=c["ink2"])
    ax.annotate("Sample size is identical on all three curves, so the vertical gaps are\n"
                "systematic error being cancelled, not random error being averaged.",
                (0.015, 0.97), xycoords="axes fraction", va="top", fontsize=8.8,
                color=c["muted"], linespacing=1.6)
    fig.tight_layout()
    finish(fig, "fig18_noise_vs_bias", c)


def judge_by_gap(df):
    """Per judge, accuracy at narrow and wide gaps.

    Judge choice barely matters when the gap is wide and matters enormously when
    it is narrow, so a single overall number per judge hides the useful part.
    """
    rows = []

    def pair_mean(frame):
        return float(np.mean([
            item_votes(group).mean() for _, group in frame.groupby("model_pair")
        ]))

    for j, d in df.groupby("judge"):
        narrow = d[d["gap"] < 0.05]
        wide = d[d["gap"] >= 0.10]
        if narrow.empty or wide.empty:
            continue
        rows.append(dict(judge=j,
                         overall=pair_mean(d), n_pairs=int(d["model_pair"].nunique()),
                         narrow=pair_mean(narrow),
                         n_narrow_pairs=int(narrow["model_pair"].nunique()),
                         wide=pair_mean(wide),
                         n_wide_pairs=int(wide["model_pair"].nunique())))
    t = pd.DataFrame(rows).sort_values("narrow")
    out["judge_by_gap"] = t.to_dict("records")
    return t


def fig_judge_gap(t, c):
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        ax.plot([r["narrow"], r["wide"]], [i, i], color=c["grid"], lw=2.4, zorder=1)
    ax.scatter(t["narrow"], y, s=76, color=c["s2"], zorder=3,
               edgecolor=c["surface"], linewidth=1.2, label="capability gap under 0.05")
    ax.scatter(t["wide"], y, s=76, color=c["s1"], zorder=3,
               edgecolor=c["surface"], linewidth=1.2, label="capability gap 0.10 and above")
    ax.axvline(0.5, color=c["crit"], lw=1.5, ls=(0, (5, 3)), zorder=2)
    ax.annotate("chance", (0.5, len(t) - 0.4), color=c["crit"], fontsize=8.8,
                ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(t["judge"], fontsize=10, color=c["ink2"])
    # Annotate above the marker, not beside it: the leftmost judge sits hard
    # against the axis and a left-placed label lands on its own tick text.
    for i, (_, r) in enumerate(t.iterrows()):
        if r["narrow"] < 0.5:
            ax.annotate("below chance", (r["narrow"], i), xytext=(0, 11),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.2, color=c["crit"])
    ax.set_xlabel("accuracy at identifying the stronger model", color=c["ink2"])
    ax.set_title("Judge choice hardly matters at wide gaps and matters enormously at narrow ones",
                 color=c["ink"], loc="left", fontsize=12)
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=c["ink2"],
              bbox_to_anchor=(0.0, -0.19), ncol=2)
    fig.tight_layout()
    finish(fig, "fig17_judge_by_gap", c)


def main(argv: list[str] | None = None):
    global CSV, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--power-draws", type=int, default=400)
    parser.add_argument("--panel-draws", type=int, default=300)
    parser.add_argument("--skip-feature-ensembles", action="store_true",
                        help="skip analyses that read existing feature-fit artifacts")
    args = parser.parse_args(argv)
    CSV = args.csv
    OUT = args.out_dir
    plotting.OUT = args.out_dir
    out.clear()
    df = load()
    print(f"rows {len(df)}  pairs {df['model_pair'].nunique()}  judges {df['judge'].nunique()}")
    p = pair_level(df)
    print(f"\nA. pair-level: {out['pair_level']['n_correct']}/{out['pair_level']['n_pairs']} "
          f"pairs ranked correctly ({out['pair_level']['accuracy']:.3f})")
    for b, v in out["pair_level"]["by_band"].items():
        print(f"     gap {b:<16} {v['correct']}/{v['pairs']} correct")
    pw = power_curve(df, B=args.power_draws)
    print("\nB. power curve (pooled verdict accuracy):")
    for band, v in pw.items():
        pts = {q["k"]: round(q["accuracy"], 3) for q in v["curve"]}
        print(f"     gap {band:<16} k=1:{pts.get(1)}  k=31:{pts.get(31)}  k=255:{pts.get(255)}  k=1023:{pts.get(1023)}")
    lv = level_vs_gap(df)
    print("\nC. level vs gap, within gaps under 0.05:")
    for g in lv.get("groups", []):
        print(f"     {g['level_group']:<26} pairs={g['pairs']:<3} acc={g['accuracy']:.4f} "
              f"mean gap={g['mean_gap']:.4f} mean level={g['mean_level']:.3f}")
    lc = length_control(df)
    print("\nD. condition comparison (matched pair/judge/item/order cells):")
    for r in sorted(lc["matched_cells"], key=lambda r: r["condition"]):
        print(f"     {r['condition']:<14} n={r['n_cells']:<6} acc={r['accuracy']:.4f}")
    nb = noise_vs_bias(df, B=args.panel_draws)
    nbg = noise_vs_bias_by_gap(df, B=args.panel_draws)
    panel_on_covered_pairs(df, min_judges=5, B=args.panel_draws)
    solo, best = solo_judges_vs_panel(df, B=args.panel_draws)
    sbs = solo_best_by_subset(df, B=args.panel_draws)
    sim = None if args.skip_feature_ensembles else judge_bias_similarity()
    if sim:
        srcs = bias_similarity_sources(sim[0], df)
        dive = diverse_ensemble(df, sim[0], B=args.panel_draws)
        fdiv = feature_diversity(sim[2])
        sfe = split_feature_ensembles(
            df, sim[2], B=args.panel_draws
        )
    print(f"\nF. noise vs bias, {out['noise_vs_bias']['n_pairs']} pairs, same k at every level:")
    ks = [p["k"] for p in nb["one judge, one order"]]
    print("     %-26s %s" % ("level", "  ".join(f"k={k}" for k in ks)))
    for name, cur in nb.items():
        print("     %-26s %s" % (name, "  ".join(f"{p['accuracy']:.3f}" for p in cur)))
    jg = judge_by_gap(df)
    print("\nE. per judge, narrow vs wide gap:")
    for _, r in jg.iterrows():
        flag = "  <- below chance" if r["narrow"] < 0.5 else ""
        print(f"     {r['judge']:<24} narrow={r['narrow']:.3f}  wide={r['wide']:.3f}{flag}")
    for c in (LIGHT, DARK):
        style(c)
        fig_power(pw, c)
        fig_pairs(p, c)
        fig_length(lc, c)
        fig_judge_gap(jg, c)
        fig_noise_vs_bias(nb, c)
        fig_noise_by_gap(nbg, c)
        fig_solo_vs_panel(solo, best, c)
        fig_solo_facets(sbs, "by_length", "Best single judge against the panel, by how much of the transcript was shown", "fig21_solo_by_length", c)
        fig_solo_facets(sbs, "by_gap", "Best single judge against the panel, by capability gap", "fig22_solo_by_gap", c)
        if sim:
            fig_bias_similarity(sim[0], sim[1], c)
            fig_bias_sources(srcs[0], srcs[1], c)
            fig_diverse_ensemble(dive, c)
            fig_feature_diversity(fdiv, c)
            if sfe: fig_split_ensembles(sfe, c)
    write_json(OUT / "evaluation_power.json", out)
    print(f"\nwrote {OUT / 'evaluation_power.json'} and the evaluation figure set")


def split_feature_ensembles(df, M, n_members=4, min_disagree=0.30,
                            sizes=(1, 3, 7, 15, 31, 63, 127, 255), B=300):
    """Rebuild the panels using only the features judges actually split on.

    Correlating judges across all 816 coefficients is dominated by the handful
    of large, unanimous biases, which makes every judge look alike and hides
    whatever structure exists elsewhere. Restricting to features where at least
    `min_disagree` of judges sit on the minority side of zero measures similarity
    only where there is disagreement to measure.

    All three panels are then scored on the same pairs, so the curves are
    directly comparable rather than each having its own baseline.
    """
    pos = (M > 0).sum(axis=1)
    n = M.notna().sum(axis=1)
    disagree = np.minimum(pos, n - pos) / n
    keep = M.index[disagree >= min_disagree]
    if len(keep) < 10:
        return None
    Msub = M.loc[keep]
    corr = Msub.corr()
    js = list(corr.columns)

    def pick(minimise):
        chosen = [min(js, key=lambda j: corr[j].mean()) if minimise
                  else max(js, key=lambda j: corr[j].mean())]
        while len(chosen) < n_members:
            rest = [j for j in js if j not in chosen]
            sc = {j: np.mean([corr.loc[j, c] for c in chosen]) for j in rest}
            chosen.append(min(sc, key=sc.get) if minimise else max(sc, key=sc.get))
        return chosen

    diverse, similar = pick(True), pick(False)
    def covered(members, need):
        s = df[df["judge"].isin(members)]
        cc = s.groupby("model_pair")["judge"].nunique()
        return set(cc[cc >= need].index)
    pairs = sorted(covered(diverse, n_members) & covered(similar, n_members))
    need = n_members
    while len(pairs) < 4 and need > 2:
        need -= 1
        pairs = sorted(covered(diverse, need) & covered(similar, need))
    if len(pairs) < 3:
        return None

    rng = np.random.default_rng(SEED)
    def majority(arr, k):
        return float(majority_trials(arr, k, B, rng).mean())

    def curve(members):
        sub = df if members is None else df[df["judge"].isin(members)]
        arrs = {p: item_votes(sub[sub["model_pair"] == p]) for p in pairs}
        arrs = {p: a for p, a in arrs.items() if len(a) >= 8}
        return [dict(k=int(k), accuracy=float(np.mean([majority(a, k) for a in arrs.values()])))
                for k in sizes], len(arrs)

    c_div, n_div = curve(diverse)
    c_sim, n_sim = curve(similar)
    c_all, n_all = curve(None)
    mc = lambda g: float(np.mean([corr.loc[a, b] for i, a in enumerate(g) for b in g[i + 1:]]))
    res = dict(n_features=int(len(keep)), n_judges=len(js), min_disagree=min_disagree,
               judges_required=need, n_pairs=len(pairs),
               diverse=dict(members=diverse, mean_corr=mc(diverse), curve=c_div, pairs=n_div),
               similar=dict(members=similar, mean_corr=mc(similar), curve=c_sim, pairs=n_sim),
               all_judges=dict(curve=c_all, pairs=n_all))
    out["split_feature_ensembles"] = res
    return res


def fig_split_ensembles(r, c):
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    spec = [("all_judges", "s1", "D", f"all {r['n_judges']} judges"),
            ("similar", "s2", "o", f"similar panel (corr {r['similar']['mean_corr']:+.2f})"),
            ("diverse", "s3", "s", f"dissimilar panel (corr {r['diverse']['mean_corr']:+.2f})")]
    for key, col, mk, lab in spec:
        cur = r[key]["curve"]
        ax.plot([p["k"] for p in cur], [p["accuracy"] for p in cur], marker=mk, ms=6,
                lw=2.2, color=c[col], label=lab)
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (1.05, 0.505), color=c["crit"], fontsize=8.8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled per model pair", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("Panels chosen on the features judges actually split on",
                 color=c["ink"], loc="left")
    ax.annotate(f"Similarity measured over the {r['n_features']} features where at least "
                f"{int(100*r['min_disagree'])}% of judges sit on the\nminority side of zero. "
                f"All three curves use the same {r['n_pairs']} model pairs.",
                (0.015, 0.975), xycoords="axes fraction", va="top", fontsize=8.6,
                color=c["muted"], linespacing=1.6)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig27_split_feature_ensembles", c)


if __name__ == "__main__":
    main()
