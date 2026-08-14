"""How confidently is the capability ordering itself known?

The pair-ranking analysis treats the model with higher aggregate GPQA Diamond
accuracy as the more capable one. Those accuracies are estimates from a finite
item set, so for a close pair the ordering may not be established at all. A pair
the judge "fails" may be a pair whose true ordering is unknown.

This estimates a Beta posterior on each model's accuracy from its per-item
results, computes P(theta_major > theta_minor) for every judged pair, and asks
whether the failures concentrate where that probability is near a coin flip.

If they do, the close-pair failure is partly a property of the label rather than
of the judge, and the claim must be stated accordingly.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting import DARK, HIB, LIGHT, style, write_json

ROOT = HIB.parent.parent
RESPONSES = ROOT / "external_data" / "epoch-gpqa" / "responses.parquet"
OUT = HIB / "readme_figures"
DRAWS = 40000
SEED = 0


LOGS = pathlib.Path("/fast/jyeon/judge_logs/w1_gpqa")
ROLLOUTS = pathlib.Path("/fast/jyeon/judge_logs/rollouts")


def model_posteriors():
    """Per-item GPQA counts for each judged model, from both capability sources.

    The judge logs record which file each model's capability came from. That
    stem is either an Epoch response-set model name or one of our own collection
    logs, so both sources must be read to cover every judged model.
    """
    from inspect_ai.log import read_eval_log
    from prepare_gpqa import resolve_sources

    mapping = resolve_sources(LOGS)
    d = pd.read_parquet(RESPONSES, columns=["model", "item", "correct"])
    # One record per model and item, so repeated epochs are not counted as
    # independent evidence about the same question.
    d = d.groupby(["model", "item"])["correct"].mean().reset_index()
    epoch = d.groupby("model")["correct"].agg(["sum", "size"])
    epoch.columns = ["correct", "n"]

    rows = {}
    for short, stem in mapping.items():
        if stem in epoch.index:
            rows[short] = dict(correct=float(epoch.loc[stem, "correct"]),
                               n=int(epoch.loc[stem, "n"]), source="epoch")
            continue
        hit = sorted(ROLLOUTS.glob(f"*{stem}*.eval"))
        if not hit:
            continue
        log = read_eval_log(str(hit[0]))
        vals = []
        for smp in (log.samples or []):
            sc = list((smp.scores or {}).values())
            if sc:
                v = sc[0].value
                vals.append(1.0 if v in (1, 1.0, True, "C") else 0.0)
        if vals:
            rows[short] = dict(correct=float(np.sum(vals)), n=len(vals),
                               source="rollout")
    g = pd.DataFrame(rows).T
    g["accuracy"] = g["correct"] / g["n"]
    return g


def item_matrix():
    """Per-item correctness for each judged model, aligned on question id.

    Both models in a pair answer the same GPQA items, so their correctness is
    paired through question difficulty. Independent Beta posteriors ignore that
    structure. This returns the aligned matrix a paired bootstrap needs.
    """
    from inspect_ai.log import read_eval_log
    from prepare_gpqa import resolve_sources

    mapping = resolve_sources(LOGS)
    d = pd.read_parquet(RESPONSES, columns=["model", "item", "correct"])
    d = d.groupby(["model", "item"])["correct"].mean().reset_index()
    cols = {}
    for short, stem in mapping.items():
        sub = d[d["model"] == stem]
        if len(sub):
            cols[short] = sub.set_index("item")["correct"]
            continue
        hit = sorted(ROLLOUTS.glob(f"*{stem}*.eval"))
        if not hit:
            continue
        log = read_eval_log(str(hit[0]))
        vals = {}
        for smp in (log.samples or []):
            sc = list((smp.scores or {}).values())
            if sc:
                v = sc[0].value
                vals[str(smp.id)] = 1.0 if v in (1, 1.0, True, "C") else 0.0
        if vals:
            cols[short] = pd.Series(vals)
    return pd.DataFrame(cols)


def paired_bootstrap(M, pairs, rng, draws=20000):
    """Resample questions, not models, and recompute both accuracies together."""
    rows = []
    for pair, gap, ranked in pairs:
        a, b = pair.split("|")
        if a not in M.columns or b not in M.columns:
            rows.append(dict(pair=pair, p_paired=None, n_items=0))
            continue
        both = M[[a, b]].dropna()
        if len(both) < 20:
            rows.append(dict(pair=pair, p_paired=None, n_items=int(len(both))))
            continue
        va, vb = both[a].to_numpy(), both[b].to_numpy()
        hi, lo = (va, vb) if va.mean() >= vb.mean() else (vb, va)
        idx = rng.integers(0, len(both), size=(draws, len(both)))
        diff = hi[idx].mean(axis=1) - lo[idx].mean(axis=1)
        rows.append(dict(pair=pair, p_paired=float((diff > 0).mean()),
                         n_items=int(len(both))))
    return pd.DataFrame(rows)


def pair_probabilities(post, pairs, rng):
    """P(theta_major > theta_minor) for each judged pair."""
    rows = []
    for pair, gap, ranked in pairs:
        a, b = pair.split("|")
        if a not in post.index or b not in post.index:
            rows.append(dict(pair=pair, gap=gap, ranked_correctly=ranked,
                             p_order=None, note="no per-item results"))
            continue
        ra, rb = post.loc[a], post.loc[b]
        # major is whichever the pipeline treated as more capable
        hi, lo = (ra, rb) if ra["accuracy"] >= rb["accuracy"] else (rb, ra)
        da = rng.beta(1 + hi["correct"], 1 + hi["n"] - hi["correct"], DRAWS)
        db = rng.beta(1 + lo["correct"], 1 + lo["n"] - lo["correct"], DRAWS)
        rows.append(dict(pair=pair, gap=gap, ranked_correctly=ranked,
                         p_order=float((da > db).mean()),
                         n_hi=int(hi["n"]), n_lo=int(lo["n"]), note=""))
    return pd.DataFrame(rows)


def fig(t, c, n=10):
    d = t.dropna(subset=["p_order"])
    fig_, (axl, ax) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                   gridspec_kw=dict(width_ratios=[1, 1.25]))
    # left: do the two estimators agree?
    a = d.dropna(subset=["p_paired"])
    axl.scatter(a["p_order"], a["p_paired"], s=46, color=c["s1"], zorder=3,
                edgecolor=c["surface"], linewidth=.9)
    bad_a = a[~a["ranked_correctly"]]
    axl.scatter(bad_a["p_order"], bad_a["p_paired"], s=64, color=c["crit"],
                marker="X", zorder=4, edgecolor=c["surface"], linewidth=.9)
    axl.plot([0, 1], [0, 1], color=c["muted"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    axl.set_xlabel("independent Beta posterior", color=c["ink2"])
    axl.set_ylabel("paired bootstrap over questions", color=c["ink2"])
    axl.set_title("The two estimators disagree", color=c["ink"], loc="left",
                  fontsize=10.5)
    axl.annotate("both models answer the same items,\nso the pairing matters",
                 (0.04, 0.95), xycoords="axes fraction", va="top", fontsize=8.4,
                 color=c["muted"], linespacing=1.5)
    axl.grid(lw=0.8); axl.set_axisbelow(True)
    d = d.assign(p_order=d["p_paired"].fillna(d["p_order"]))
    ok = d[d["ranked_correctly"]]
    bad = d[~d["ranked_correctly"]]
    ax.scatter(ok["p_order"], ok["gap"], s=52, color=c["s1"], zorder=3,
               edgecolor=c["surface"], linewidth=1, label="judge ranked the pair correctly")
    ax.scatter(bad["p_order"], bad["gap"], s=64, color=c["crit"], marker="X", zorder=4,
               edgecolor=c["surface"], linewidth=1, label="judge ranked the pair wrong")
    ax.axvline(0.95, color=c["axis"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("0.95 support\nthreshold", (0.95, ax.get_ylim()[1]),
                xytext=(-8, -6), textcoords="offset points", ha="right", va="top",
                fontsize=8.6, color=c["muted"], linespacing=1.5)
    ax.set_xlabel("paired-bootstrap probability the labelled stronger model is ahead on GPQA",
                  color=c["ink2"])
    ax.set_ylabel("labelled capability gap", color=c["ink2"])
    ax.set_title("Only one failure sits where the ordering is uncertain",
                 color=c["ink"], loc="left", fontsize=11.5)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig_.suptitle("Is the capability ordering itself established?", color=c["ink"],
                  fontsize=13, fontweight="semibold", x=0.005, ha="left", y=1.02)
    fig_.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_.savefig(OUT / f"{n}{'' if c is LIGHT else '_dark'}.png", dpi=150,
                 bbox_inches="tight", facecolor=c["surface"])
    if c is LIGHT:
        fig_.savefig(OUT / f"{n}.svg", bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig_)


def main():
    rng = np.random.default_rng(SEED)
    post = model_posteriors()
    pl = json.loads((HIB / "figures" / "pooling_by_question.json").read_text())["pair_level"]
    pairs = [(r["pair"], r["gap"], r["correct"]) for r in pl]
    t = pair_probabilities(post, pairs, rng)
    M = item_matrix()
    t = t.merge(paired_bootstrap(M, pairs, rng), on="pair", how="left")

    have = t.dropna(subset=["p_order"])
    print(f"pairs: {len(t)}   with per-item GPQA results for both models: {len(have)}")
    if len(have):
        counts = post["n"].astype(int)
        print(f"items per model: median {int(counts.median())}, "
              f"min {int(counts.min())}   "
              f"sources: {post['source'].value_counts().to_dict()}")
        est = have[have["p_order"] >= 0.95]
        unc = have[have["p_order"] < 0.95]
        print(f"\nordering established at 95%: {len(est)} pairs, "
              f"judge correct on {int(est['ranked_correctly'].sum())}")
        print(f"ordering NOT established:    {len(unc)} pairs, "
              f"judge correct on {int(unc['ranked_correctly'].sum())}")
        print("\nfailed pairs:")
        for _, r in have[~have["ranked_correctly"]].sort_values("p_order").iterrows():
            pb = "n/a" if pd.isna(r.get("p_paired")) else f"{r['p_paired']:.3f}"
            print(f"  {r['pair']:<46} gap {r['gap']:.4f}  "
                  f"beta {r['p_order']:.3f}  paired {pb}")
        ag = have.dropna(subset=["p_paired"])
        if len(ag):
            worst = float((ag["p_order"] - ag["p_paired"]).abs().max())
            print(f"\nBeta posterior vs paired bootstrap: largest difference "
                  f"{worst:.4f} across {len(ag)} pairs")
    write_json(OUT / "capability_uncertainty.json",
               dict(pairs=t.to_dict("records"),
                    median_items_per_model=int(post["n"].astype(int).median())))
    for c in (LIGHT, DARK):
        style(c)
        fig(t, c)
    print(f"\nwrote {OUT / 'capability_uncertainty.json'} and figure 10")


if __name__ == "__main__":
    main()
