"""Per-method decomposition: does the judge-bias picture depend on the
feature extraction method?

The primary analysis classifies each of gpqa_tfa's hand-built features by
whether the judge follows it and whether it carries capability signal (fig 7).
gpqa_tfa's capability evidence also has 240 TF-IDF topic components and 32 SAE
latents. This reads the three bias maps, which were built with identical fits
and identical multiplicity control, and draws the same stacked bar once per
method, so the reader can see whether "most followed features carry no
capability signal" is a property of hand-built features or of the judge.

Writes readme figure 15 and method_decomposition.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plotting import DARK, HIB, LIGHT, style, write_json

OUT = HIB / "readme_figures"
METHODS = [
    ("hand-built", HIB / "features_within" / "bias_map.csv"),
    ("TF-IDF", HIB / "features_tfidf" / "bias_map.csv"),
    ("SAE", HIB / "features_sae_judge" / "bias_map.csv"),
]
PARTS = [
    ("no association with judge choices", "none", "muted"),
    ("judge bias: followed, no capability signal", "JUDGE BIAS", "s1"),
    ("opposite direction to capability", "INVERTED", "s2"),
    ("same direction as capability", "justified signal", "s5"),
]


def marginal(path):
    b = pd.read_csv(path)
    b = b[~b["is_component"].astype(bool)]
    return b[b["spec"] == "marginal"].drop_duplicates(subset="feature")


def tally(m):
    follows = m["judge_follows"].astype(str).str.lower().eq("true")
    q = m["quadrant"]
    t = {
        "none": int((~follows).sum()),
        "JUDGE BIAS": int((q == "JUDGE BIAS").sum()),
        "INVERTED": int((q == "INVERTED").sum()),
        "justified signal": int((q == "justified signal").sum()),
        "missed signal": int((q == "missed signal").sum()),
        "n": int(len(m)),
    }
    # Every capability-tracking feature must land in exactly one of the three
    # quadrants that can hold one; drift here means the evidence join broke.
    tracks = int(m["tracks_capability"].astype(str).str.lower().eq("true").sum())
    assert (t["INVERTED"] + t["justified signal"] + t["missed signal"]) == tracks, (
        f"quadrants disagree with tracks_capability ({tracks})")
    return t


def fig(rows, c, n=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_, ax = plt.subplots(figsize=(10.4, 3.9))
    ys = np.arange(len(rows))[::-1]
    for y, (method, t) in zip(ys, rows):
        left = 0.0
        for label, key, col in PARTS:
            share = t[key] / t["n"]
            ax.barh([y], [share], left=left, height=.56, color=c[col],
                    edgecolor=c["surface"], linewidth=2)
            if share > 0.04:
                ax.text(left + share / 2, y, str(t[key]), ha="center",
                        va="center", fontsize=10.5,
                        color="#ffffff" if col != "muted" else c["ink"],
                        fontweight="bold")
            left += share
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{m}  (n = {t['n']})" for m, t in rows],
                       fontsize=10.5, color=c["ink2"])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the method's features in the within-pair marginal "
                  "analysis", color=c["ink2"])
    ax.set_title("The decomposition is not an artefact of hand-built features",
                 color=c["ink"], loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[col]) for _, _, col in PARTS]
    ax.legend(handles, [p[0] for p in PARTS], frameon=False, fontsize=8.8,
              loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2,
              labelcolor=c["ink2"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig_.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_.savefig(OUT / f"{n}{'' if c is LIGHT else '_dark'}.png", dpi=150,
                 bbox_inches="tight", facecolor=c["surface"])
    if c is LIGHT:
        fig_.savefig(OUT / f"{n}.svg", bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig_)


def main():
    rows = []
    for method, path in METHODS:
        if not path.exists():
            print(f"skipping {method}: {path} does not exist yet")
            continue
        t = tally(marginal(path))
        rows.append((method, t))
        followed = t["JUDGE BIAS"] + t["INVERTED"] + t["justified signal"]
        print(f"{method:<11} n {t['n']:>3}  followed {followed:>3}  "
              f"bias {t['JUDGE BIAS']:>3}  inverted {t['INVERTED']:>2}  "
              f"justified {t['justified signal']:>3}  missed {t['missed signal']:>2}")
    if len(rows) < 2:
        raise SystemExit("need at least two methods' bias maps for the figure")
    write_json(OUT / "method_decomposition.json",
               {m: t for m, t in rows})
    for c in (LIGHT, DARK):
        style(c)
        fig(rows, c)
    print(f"wrote {OUT / 'method_decomposition.json'} and figure 15")


if __name__ == "__main__":
    main()
