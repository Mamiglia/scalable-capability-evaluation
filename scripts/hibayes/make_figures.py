"""Build the figure set for the judge feature analysis.

Reads the finished fits under analysis/hibayes and writes one PNG per figure to
analysis/hibayes/figures, in a light and a dark variant, plus figures.json with
the numbers each figure was drawn from.

Every figure answers one question. The four verdicts get one figure each rather
than sharing a scatter, because the scatter collapses four different findings
into one picture and none of them reads clearly.

Verdicts, as bias_map.csv assigns them:
  JUDGE BIAS        the judge follows it without a detected capability signal
  justified signal  the judge follows it and it tracks held-out accuracy
  missed signal     it tracks held-out accuracy and the judge does not follow it
  correctly ignored it has no detected signal and the judge ignores it
  INVERTED          the judge follows it in the opposite direction to the signal
  component         a principal component with no single feature to match against
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

SETS = ["features_within", "features_between", "features_crossfam", "features_samefam"]

VERDICT_COLOR = {"JUDGE BIAS": "s1", "INVERTED": "s2", "correctly ignored": "s3",
                 "missed signal": "s4", "justified signal": "s5",
                 "component (no counterpart)": "s6"}

manifest: dict[str, object] = {}


def nice(s):
    return str(s).replace("_", " ")


def load_bias():
    frames = []
    for s in SETS:
        df = pd.read_csv(HIB / s / "bias_map.csv")
        df["set"] = s.replace("features_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_accuracy():
    """Weighted LOO accuracy per fit: how much of the judge's choice it recovers."""
    rows = []
    for s in SETS:
        for d in sorted((HIB / s).glob("*/communicate/judge_loo_accuracy.csv")):
            a = pd.read_csv(d)
            required = {"n", "loo_accuracy", "pareto_k_over_0.7"}
            if not required <= set(a.columns):
                raise ValueError(f"{d} is missing columns {sorted(required - set(a.columns))}")
            if a["n"].sum() == 0:
                raise ValueError(f"{d} contains no evaluated rows")
            run = d.parent.parent.name
            parts = run.split("__")
            rows.append(dict(
                set=s.replace("features_", ""), run=run, spec=parts[0],
                family=parts[1] if len(parts) > 1 else "",
                feature=parts[2] if len(parts) > 2 else "",
                n=int(a["n"].sum()),
                accuracy=float(np.average(a["loo_accuracy"], weights=a["n"])),
                worst_judge=float(a["loo_accuracy"].min()),
                best_judge=float(a["loo_accuracy"].max()),
                pareto_k_over_0_7=int(a["pareto_k_over_0.7"].sum()),
            ))
    return pd.DataFrame(rows)


def forest(ax, labels, mean, lo, hi, c, color="s1", show_values=True):
    y = np.arange(len(labels))[::-1]
    cred = (lo > 0) | (hi < 0)
    col = c[color]
    ax.hlines(y, lo, hi, color=col, lw=2.0, alpha=0.95, zorder=2,
              capstyle="round")
    ax.scatter(mean[cred], y[cred], s=42, color=col, zorder=3,
               edgecolor=c["surface"], linewidth=1.0)
    ax.scatter(mean[~cred], y[~cred], s=42, facecolor=c["surface"], zorder=3,
               edgecolor=col, linewidth=1.8)
    ax.axvline(0, color=c["axis"], lw=1.3, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=c["ink2"])
    ax.grid(axis="x", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    if show_values:
        span = (hi.max() - lo.min()) or 1
        for yy, m in zip(y, mean):
            ax.annotate(f"{m:+.2f}", (m, yy), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5,
                        color=c["ink"], family="monospace")
    return cred


# ----------------------------------------------------------------- figures
def fig_headline(bias, acc, c):
    pop = {k: pd.read_csv(HIB / k / "communicate" / "population_effects.csv")
           for k in ("run", "run-correctness")}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9))
    titles = ["Which side the judge picked", "Whether the judge was right"]
    for ax, (k, df), t in zip(axes, pop.items(), titles):
        df = df.reindex(df["mean_log_odds"].abs().sort_values().index)
        forest(ax, [nice(e) for e in df["effect"]], df["mean_log_odds"].values,
               df["hdi_2.5%"].values, df["hdi_97.5%"].values, c)
        ax.set_title(t, color=c["ink"], loc="left")
        ax.set_xlabel("effect on the odds (log-odds)", color=c["ink2"])
    fig.suptitle("The two headline models", color=c["ink"], fontsize=13.5,
                 fontweight="semibold", x=0.005, ha="left", y=1.06)
    fig.tight_layout()
    finish(fig, "fig01_headline", c)
    if c is LIGHT:
        manifest["headline"] = {k: v.to_dict("records") for k, v in pop.items()}


def fig_position(bias, acc, c):
    """Position bias runs both ways. The population mean hides it."""
    j = pd.read_csv(HIB / "run" / "communicate" / "judge_feature_effects.csv")
    p = j[j["feature"] == "position_bias_left"].sort_values("mean_log_odds")
    y = np.arange(len(p))
    left = p["mean_log_odds"].values > 0
    cred = p["credible_nonzero"].astype(str).str.lower().eq("true").values
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    for i, (m, lo, hi, isleft, ok) in enumerate(zip(
            p["mean_log_odds"], p["hdi_2.5%"], p["hdi_97.5%"], left, cred)):
        col = c["s1"] if isleft else c["s2"]
        ax.hlines(i, lo, hi, color=col, lw=2.2, alpha=1.0 if ok else 0.45, zorder=2)
        ax.scatter([m], [i], s=52, zorder=3, linewidth=1.8,
                   color=col if ok else c["surface"], edgecolor=col)
    ax.axvline(0, color=c["axis"], lw=1.4, zorder=1)
    population = pd.read_csv(HIB / "run" / "communicate" / "population_effects.csv")
    pop = float(population.loc[
        population["effect"] == "position_bias_left", "mean_log_odds"
    ].iloc[0])
    ax.axvline(pop, color=c["crit"], lw=1.5, ls=(0, (5, 3)), zorder=1)
    ax.annotate(f"population mean {pop:+.2f}:\nthe opposing biases cancel\nin the aggregate",
                (pop, 4.5), fontsize=8.6, color=c["crit"], ha="center", va="center",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=c["surface"],
                          edgecolor=c["crit"], linewidth=0.9))
    ax.set_yticks(y)
    ax.set_yticklabels(p["judge"], fontsize=10, color=c["ink2"])
    ax.grid(axis="x", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("favours the answer shown on the RIGHT  ←     →  favours the answer shown on the LEFT",
                  color=c["ink2"], fontsize=9.5)
    n_credible = int(cred.sum())
    ax.set_title(f"{n_credible} of {len(p)} judge intervals exclude zero, in opposing directions",
                 color=c["ink"], loc="left")
    lim = max(abs(p["hdi_2.5%"].min()), abs(p["hdi_97.5%"].max())) * 1.12
    ax.set_xlim(-lim, lim)
    handles = [plt.Line2D([], [], marker="o", ls="none", color=c["s2"], label="favours right"),
               plt.Line2D([], [], marker="o", ls="none", color=c["s1"], label="favours left"),
               plt.Line2D([], [], marker="o", ls="none", markerfacecolor=c["surface"],
                          markeredgecolor=c["muted"], color=c["muted"], label="interval includes zero")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right",
              labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig02_position", c)
    if c is LIGHT:
        manifest["position"] = p[["judge", "mean_log_odds", "hdi_2.5%", "hdi_97.5%",
                                  "credible_nonzero"]].to_dict("records")


def fig_judge_consistency(bias, acc, c):
    j = pd.read_csv(HIB / "run" / "communicate" / "judge_feature_effects.csv")
    keep = ["capability_difference_sd", "length_difference_sd"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, eff in zip(axes, keep):
        d = j[j["feature"] == eff].sort_values("mean_log_odds")
        forest(ax, list(d["judge"]), d["mean_log_odds"].values,
               d["hdi_2.5%"].values, d["hdi_97.5%"].values, c, show_values=False)
        ax.set_title(nice(eff), color=c["ink"], loc="left")
        ax.set_xlabel("log-odds", color=c["ink2"])
    axes[0].set_xlim(left=0)
    fig.suptitle("Every judge responds to both capability and length",
                 color=c["ink"], fontsize=13.5, fontweight="semibold",
                 x=0.005, ha="left", y=1.04)
    fig.tight_layout()
    finish(fig, "fig03_judge_consistency", c)
    if c is LIGHT:
        manifest["judge_consistency"] = {
            effect: j[j["feature"] == effect][
                ["judge", "mean_log_odds", "hdi_2.5%", "hdi_97.5%"]
            ].to_dict("records")
            for effect in keep
        }


def fig_accuracy(bias, acc, c):
    """How much of the judge's choice does each single feature recover?"""
    all_marginal = acc[(acc["set"] == "within") & (acc["spec"] == "marginal")].copy()
    m = all_marginal[all_marginal["pareto_k_over_0_7"] == 0].copy()
    m = m.sort_values("accuracy")
    head = pd.read_csv(HIB / "run" / "communicate" / "judge_loo_accuracy.csv")
    head_acc = float(np.average(head["loo_accuracy"], weights=head["n"]))
    floor = float(m["accuracy"].min())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8),
                             gridspec_kw=dict(width_ratios=[1, 1.3]))
    # Every value sits between 0.65 and 0.69, so an axis anchored at chance (0.50)
    # would render the whole distribution as one invisible spike. The axis is
    # zoomed and the distance to chance is stated instead of drawn.
    ax = axes[0]
    ax.hist(m["accuracy"], bins=30, color=c["s1"], edgecolor=c["surface"], linewidth=0.6)
    top_y = ax.get_ylim()[1]
    ax.axvline(head_acc, color=c["s2"], lw=1.8)
    ax.annotate(f"all four headline\npredictors together: {head_acc:.3f}",
                (head_acc, top_y * 0.97), color=c["s2"], fontsize=8.8,
                ha="right", va="top", xytext=(-7, 0), textcoords="offset points")
    ax.set_xlim(floor - 0.002, max(m["accuracy"].max(), head_acc) + 0.004)
    ax.set_xlabel("leave-one-out accuracy of a one-feature model", color=c["ink2"])
    ax.set_ylabel("number of features", color=c["ink2"])
    ax.set_title("Every feature lands in the same narrow band",
                 color=c["ink"], loc="left")
    ax.annotate(f"Chance is 0.500, far to the left of this axis.\n"
                f"The whole spread here is only {m['accuracy'].max()-floor:.3f} wide.",
                (0.97, 0.62), xycoords="axes fraction", va="top", ha="right",
                fontsize=8.8, color=c["muted"], linespacing=1.6)
    ax.grid(axis="y", lw=0.8); ax.set_axisbelow(True)

    # A dot plot, not bars: bars from a non-zero baseline overstate the differences,
    # and a zero baseline would flatten them to nothing.
    top = m.tail(16)
    ax2 = axes[1]
    y = np.arange(len(top))
    ax2.hlines(y, floor, top["accuracy"], color=c["s1"], lw=1.4, alpha=.45, zorder=2)
    ax2.scatter(top["accuracy"], y, s=54, color=c["s1"], zorder=3,
                edgecolor=c["surface"], linewidth=1.0)
    ax2.axvline(head_acc, color=c["s2"], lw=1.8, zorder=1)
    ax2.annotate(f"four headline predictors {head_acc:.3f}", (head_acc, len(top) - 0.4),
                 color=c["s2"], fontsize=8.6, ha="right", va="bottom",
                 xytext=(-6, 0), textcoords="offset points", annotation_clip=False)
    ax2.axvline(floor, color=c["axis"], lw=1.2, zorder=1)
    ax2.annotate(f"weakest of the {len(m)} features, {floor:.3f}",
                 (floor, len(top) - 0.4), color=c["muted"], fontsize=8.6,
                 ha="left", va="bottom", xytext=(6, 0), textcoords="offset points",
                 annotation_clip=False)
    ax2.set_ylim(-0.8, len(top) + 0.3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([nice(f) for f in top["feature"]], fontsize=9, color=c["ink2"])
    for i, v in enumerate(top["accuracy"]):
        ax2.annotate(f"{v:.3f}", (v, i), xytext=(8, 0), textcoords="offset points",
                     va="center", fontsize=8.6, color=c["ink"], family="monospace")
    ax2.set_xlim(floor - 0.0025, max(top["accuracy"].max(), head_acc) + 0.006)
    ax2.set_xlabel("leave-one-out accuracy", color=c["ink2"])
    ax2.set_title("The 16 that recover the most", color=c["ink"], loc="left")
    ax2.grid(axis="x", lw=0.8); ax2.set_axisbelow(True)
    fig.suptitle("How much of the judge's choice each feature explains on its own",
                 color=c["ink"], fontsize=13.5, fontweight="semibold",
                 x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    finish(fig, "fig04_accuracy", c)
    if c is LIGHT:
        manifest["accuracy"] = dict(
            headline=head_acc,
            median=float(m["accuracy"].median()),
            floor=floor,
            spread=float(m["accuracy"].max() - floor),
            best=top.tail(1)[["feature", "accuracy"]].to_dict("records")[0],
            top=top[["feature", "family", "accuracy", "n"]].iloc[::-1].to_dict("records"),
            n_features=int(len(m)),
            excluded_unreliable_loo=int(len(all_marginal) - len(m)),
        )


def _verdict_panel(ax, d, c, color, xlabel, note=None, n=14):
    d = d.reindex(d["mean_log_odds"].abs().sort_values().index).tail(n)
    forest(ax, [nice(f) for f in d["feature"]], d["mean_log_odds"].values,
           d["hdi_2.5%"].values, d["hdi_97.5%"].values, c, color=color,
           show_values=False)
    ax.set_xlabel(xlabel, color=c["ink2"])
    if note:
        ax.annotate(note, (0.995, 0.02), xycoords="axes fraction", ha="right",
                    fontsize=8.6, color=c["muted"])
    return d


def fig_verdicts(bias, acc, c):
    """One figure per verdict. The scatter collapses these into an unreadable blob."""
    # Marginal fits only: one feature per model. Across all three specifications a
    # feature appears three times, which would list the same name repeatedly.
    w = bias[(bias["set"] == "within") & (bias["spec"] == "marginal")]
    specs = [
        ("JUDGE BIAS", "s1", "fig05_judge_bias",
         "Judge bias: the judge follows these, but they do not track held-out accuracy",
         "judge's response (log-odds)"),
        ("justified signal", "s5", "fig06_justified",
         "Working as intended: the judge follows features that track held-out accuracy",
         "judge's response (log-odds)"),
        ("missed signal", "s4", "fig07_missed",
         "Missed signal: these track held-out accuracy and the judge does not use them",
         "judge's response (log-odds)"),
        ("INVERTED", "s2", "fig08_inverted",
         "Inverted: the judge follows these in the wrong direction",
         "judge's response (log-odds)"),
    ]
    counts = {}
    for verdict, color, name, title, xlabel in specs:
        d = w[w["quadrant"] == verdict]
        counts[verdict] = int(len(d))
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(10.6, 5.4))
        shown = _verdict_panel(ax, d, c, color, xlabel,
                               note=f"{len(d)} marginal fits in this verdict · showing the strongest {min(14, len(d))}")
        ax.set_title(title, color=c["ink"], loc="left", fontsize=12)
        fig.tight_layout()
        finish(fig, name, c)
        if c is LIGHT:
            manifest.setdefault("verdicts", {})[verdict] = dict(
                n=int(len(d)),
                top=shown.iloc[::-1][["feature", "family", "mean_log_odds",
                                      "hdi_2.5%", "hdi_97.5%", "beta_capability"]]
                .to_dict("records"))
    if c is LIGHT:
        manifest["verdict_counts"] = counts


def fig_correctly_ignored(bias, acc, c):
    w = bias[(bias["set"] == "within") & (bias["spec"] == "marginal")]
    d = w[w["quadrant"] == "correctly ignored"]
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.hist(d["mean_log_odds"], bins=34, color=c["s3"], edgecolor=c["surface"],
            linewidth=0.6)
    ax.axvline(0, color=c["axis"], lw=1.4)
    ax.set_xlabel("judge's response (log-odds)", color=c["ink2"])
    ax.set_ylabel("number of features", color=c["ink2"])
    ax.set_title(f"Correctly ignored: {len(d)} features carry no signal and the judge does not follow them",
                 color=c["ink"], loc="left", fontsize=12)
    ax.annotate("clustered on zero, which is the desired result",
                (0.99, 0.93), xycoords="axes fraction", ha="right", fontsize=8.8,
                color=c["muted"])
    ax.grid(axis="y", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "fig09_correctly_ignored", c)
    if c is LIGHT:
        manifest["correctly_ignored"] = {
            "n": int(len(d)),
            "mean_log_odds": d["mean_log_odds"].tolist(),
        }


def fig_matrix(bias, acc, c):
    """The decision structure itself.

    The verdict is NOT read off the two coefficient magnitudes. It comes from two
    yes/no tests: does the judge follow the feature (multiplicity-corrected
    q < 0.05), and does the feature track held-out accuracy (the
    capability analysis flags it as a discovery). A 2x2 of those two tests is the
    honest picture; a scatter of the two magnitudes is not, because a large
    capability coefficient that fails its own significance test still counts as
    "no real signal".
    """
    w = bias[(bias["set"] == "within") & (bias["spec"] == "marginal") &
             (~bias["is_component"].astype(bool))].copy()
    follows = w["judge_follows"].astype(str).str.lower().eq("true")
    tracks = w["tracks_capability"].astype(str).str.lower().eq("true")
    cells = [
        # (row, col, count, label, sublabel, colour key)
        (0, 1, int((follows & tracks).sum()), "Working as intended",
         "the judge follows it\nand it tracks held-out accuracy", "s5"),
        (0, 0, int((~follows & tracks).sum()), "Missed signal",
         "it tracks held-out accuracy\nand the judge ignores it", "s4"),
        (1, 1, int((follows & ~tracks).sum()), "Judge bias",
         "the judge follows it\nwithout a detected capability signal", "s1"),
        (1, 0, int((~follows & ~tracks).sum()), "Correctly ignored",
         "no signal, and the judge\ndoes not respond", "s3"),
    ]
    total = sum(x[2] for x in cells)
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    for r, col, n, lab, sub, key in cells:
        x, y = col, 1 - r
        ax.add_patch(plt.Rectangle((x + .02, y + .02), .96, .96,
                                   facecolor=c[key], alpha=.13, edgecolor=c[key],
                                   linewidth=1.6, zorder=1))
        ax.text(x + .5, y + .70, f"{n}", ha="center", va="center",
                fontsize=34, color=c[key], fontweight="bold", zorder=2)
        ax.text(x + .5, y + .49, f"{100*n/total:.0f}% of {total}", ha="center",
                va="center", fontsize=9.5, color=c["muted"], zorder=2)
        ax.text(x + .5, y + .34, lab, ha="center", va="center", fontsize=12.5,
                color=c["ink"], fontweight="semibold", zorder=2)
        ax.text(x + .5, y + .17, sub, ha="center", va="center", fontsize=9,
                color=c["ink2"], zorder=2, linespacing=1.5)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([.5, 1.5]); ax.set_xticklabels(
        ["judge does NOT follow it", "judge FOLLOWS it"], fontsize=11, color=c["ink2"])
    ax.set_yticks([.5, 1.5]); ax.set_yticklabels(
        ["NO detected\ncapability signal", "TRACKS held-out\naccuracy"], fontsize=11,
        color=c["ink2"], linespacing=1.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Every feature sorted by two yes/no tests",
                 color=c["ink"], loc="left", fontsize=13)
    ax.annotate("“follows” = multiplicity-corrected q < 0.05 on the judge's coefficient.  "
                "“predicts” = the capability analysis flags it as a real discovery.",
                (0, -0.11), xycoords="axes fraction", fontsize=8.8, color=c["muted"])
    fig.tight_layout()
    finish(fig, "fig10_matrix", c)
    if c is LIGHT:
        manifest["matrix"] = dict(total=total,
                                  cells=[dict(n=n, label=lab) for _, _, n, lab, _, _ in cells])


def fig_map(bias, acc, c):
    """Magnitudes, coloured by verdict. Deliberately carries no quadrant labels."""
    w = bias[(bias["set"] == "within") & (bias["spec"] == "marginal") &
             bias["beta_capability"].notna()]
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    xl = float(np.abs(w["mean_log_odds"]).max()) * 1.15
    yl = float(np.abs(w["beta_capability"]).max()) * 1.15
    for q, col in [("correctly ignored", "muted"), ("missed signal", "s4"),
                   ("justified signal", "s5"), ("INVERTED", "s2"),
                   ("JUDGE BIAS", "s1")]:
        d = w[w["quadrant"] == q]
        hi = q in ("JUDGE BIAS", "INVERTED")
        ax.scatter(d["mean_log_odds"], d["beta_capability"], s=44 if hi else 26,
                   label=f"{q} ({len(d)})",
                   color=c[col] if hi else "none",
                   edgecolor=c["surface"] if hi else c[col],
                   linewidth=1.0 if hi else 1.3, alpha=1.0 if hi else 0.55,
                   zorder=3 if hi else 2)
    ax.axhline(0, color=c["axis"], lw=1.4, zorder=1)
    ax.axvline(0, color=c["axis"], lw=1.4, zorder=1)
    ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)
    ax.set_xlabel("how strongly the judge responds  (log-odds)", color=c["ink2"])
    ax.set_ylabel("estimated association with a correct answer", color=c["ink2"])
    ax.set_title("The same features as coefficient magnitudes",
                 color=c["ink"], loc="left")
    ax.annotate("Colour is the verdict from the two tests above, not the position here.\n"
                "A point can sit high and still count as “no real signal” when its own\n"
                "capability coefficient fails its significance test.",
                (0.985, 0.975), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.8, color=c["muted"], linespacing=1.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(0.0, -0.10), ncol=3, labelcolor=c["ink2"])
    ax.grid(lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "fig10b_map", c)
    if c is LIGHT:
        manifest["coefficient_map"] = w[
            ["feature", "mean_log_odds", "beta_capability", "quadrant"]
        ].to_dict("records")


def fig_families(bias, acc, c):
    w = bias[(bias["set"] == "within") & (bias["spec"] == "marginal") &
             (~bias["is_component"].astype(bool))]
    order = ["JUDGE BIAS", "INVERTED", "missed signal", "justified signal",
             "correctly ignored", "component (no counterpart)"]
    tab = (w.groupby(["family", "quadrant"]).size().unstack(fill_value=0)
           .reindex(columns=order, fill_value=0))
    share = tab.div(tab.sum(axis=1), axis=0)
    share = share.loc[share["JUDGE BIAS"].sort_values().index]
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    left = np.zeros(len(share))
    for q in order:
        vals = share[q].values
        ax.barh(np.arange(len(share)), vals, left=left, height=0.66,
                color=c[VERDICT_COLOR[q]], label=q, edgecolor=c["surface"], linewidth=1.4)
        left += vals
    ax.set_yticks(np.arange(len(share)))
    ax.set_yticklabels([nice(f) for f in share.index], fontsize=10, color=c["ink2"])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the family's features", color=c["ink2"])
    ax.set_title("Which feature families the judge is most biased by",
                 color=c["ink"], loc="left")
    for i, fam in enumerate(share.index):
        ax.annotate(f"{tab.loc[fam,'JUDGE BIAS']}/{int(tab.loc[fam].sum())}",
                    (1.005, i), va="center", fontsize=8.8, color=c["muted"],
                    annotation_clip=False)
    ax.legend(frameon=False, fontsize=8.8, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.13), labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig11_families", c)
    if c is LIGHT:
        manifest["families"] = tab.to_dict("index")


def fig_sets(bias, acc, c):
    order = ["JUDGE BIAS", "INVERTED", "missed signal", "justified signal",
             "correctly ignored", "component (no counterpart)"]
    marginal = bias[(bias["spec"] == "marginal") &
                    (~bias["is_component"].astype(bool))]
    tab = (marginal.groupby(["set", "quadrant"]).size().unstack(fill_value=0)
           .reindex(columns=order, fill_value=0)
           .reindex(["within", "crossfam", "samefam", "between"]))
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    left = np.zeros(len(tab))
    for q in order:
        vals = tab[q].values
        ax.barh(np.arange(len(tab)), vals, left=left, height=0.6,
                color=c[VERDICT_COLOR[q]], label=q, edgecolor=c["surface"], linewidth=1.6)
        left += vals
    ax.set_yticks(np.arange(len(tab)))
    ax.set_yticklabels(tab.index, fontsize=10, color=c["ink2"])
    ax.set_xlabel("number of feature fits", color=c["ink2"])
    ax.set_title("The same picture holds across every comparison set",
                 color=c["ink"], loc="left")
    ax.legend(frameon=False, fontsize=8.8, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.20), labelcolor=c["ink2"])
    fig.tight_layout()
    finish(fig, "fig12_sets", c)
    if c is LIGHT:
        manifest["sets"] = tab.to_dict("index")


def fig_two_axes(bias, acc, c):
    """The two questions, side by side: what the judge follows vs what actually predicts."""
    w = bias[(bias["set"] == "within") & bias["beta_capability"].notna()].copy()
    # Left panel: what the judge responds to, restricted to responses that survive
    # the multiplicity correction. Right panel: what tracks held-out accuracy,
    # restricted to features the capability analysis actually flags as real.
    # Ranking raw magnitudes on either side would put unsupported estimates in
    # both lists and overstate the overlap.
    # One row per feature. A feature appears once per specification (marginal,
    # joint, adjusted) and its capability coefficient is identical across all
    # three, so without this the right-hand list is the same feature repeated.
    # The marginal fits are the clean per-feature estimate: one feature per model.
    w = w[w["spec"] == "marginal"].drop_duplicates(subset="feature")
    follows = w["judge_follows"].astype(str).str.lower().eq("true")
    tracks = w["tracks_capability"].astype(str).str.lower().eq("true")
    jw = w[follows]
    cw = w[tracks]
    top_judge = jw.reindex(jw["mean_log_odds"].abs().sort_values().index).tail(12)
    top_cap = cw.reindex(cw["beta_capability"].abs().sort_values().index).tail(12)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    ax = axes[0]
    y = np.arange(len(top_judge))
    ax.barh(y, top_judge["mean_log_odds"], height=0.66,
            color=[c["s1"] if v > 0 else c["s2"] for v in top_judge["mean_log_odds"]])
    ax.axvline(0, color=c["axis"], lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels([nice(f) for f in top_judge["feature"]],
                                         fontsize=9, color=c["ink2"])
    ax.set_xlabel("judge's response (log-odds)", color=c["ink2"])
    ax.set_title("What the judge responds to most", color=c["ink"], loc="left")
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    ax = axes[1]
    y = np.arange(len(top_cap))
    ax.barh(y, top_cap["beta_capability"], height=0.66,
            color=[c["s3"] if v > 0 else c["s4"] for v in top_cap["beta_capability"]])
    ax.axvline(0, color=c["axis"], lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels([nice(f) for f in top_cap["feature"]],
                                         fontsize=9, color=c["ink2"])
    ax.set_xlabel("association with held-out GPQA accuracy", color=c["ink2"])
    ax.set_title("What tracks held-out accuracy", color=c["ink"], loc="left")
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    overlap = set(top_judge["feature"]) & set(top_cap["feature"])
    fig.suptitle(f"The two lists share only {len(overlap)} of 12 features",
                 color=c["ink"], fontsize=13.5, fontweight="semibold",
                 x=0.005, ha="left", y=1.03)
    fig.tight_layout()
    finish(fig, "fig13_two_axes", c)
    if c is LIGHT:
        manifest["two_axes"] = dict(
            overlap=sorted(overlap),
            top_judge=top_judge.iloc[::-1][["feature", "mean_log_odds", "beta_capability"]].to_dict("records"),
            top_capability=top_cap.iloc[::-1][["feature", "mean_log_odds", "beta_capability"]].to_dict("records"))


FIGURES = [fig_headline, fig_position, fig_judge_consistency, fig_accuracy,
           fig_verdicts, fig_correctly_ignored, fig_matrix, fig_map, fig_families,
           fig_sets, fig_two_axes]


def main(argv: list[str] | None = None):
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    OUT = args.out_dir
    plotting.OUT = args.out_dir
    manifest.clear()
    bias = load_bias()
    acc = load_accuracy()
    print(f"bias rows {len(bias)}   accuracy rows {len(acc)}")
    for c in (LIGHT, DARK):
        style(c)
        for fn in FIGURES:
            fn(bias, acc, c)
        print(f"{'light' if c is LIGHT else 'dark'} variants written")
    manifest["counts"] = dict(bias_rows=int(len(bias)), fits_with_accuracy=int(len(acc)))
    write_json(OUT / "figures.json", manifest)
    files = sorted(p.name for p in OUT.iterdir())
    print(f"{len(files)} files in {OUT}")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
