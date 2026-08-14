"""Cross-domain transfer analysis. Reads the verdict jsonls (base + each cell
adapter on each test domain) and renders, per (label, size):
  - a 3x3-ish grid: train-set (rows) x test-domain (cols), P(pick more capable)
  - transfer (off-diagonal vs in-domain ceiling / base floor) and mixing (LOO vs
    best single-source) deltas
  - ECI-vs-in-domain label disagreement: where the two labels flip, which wins.

Run locally after pulling verdicts to judge_results/judge_ft/cross/<size>/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import _picked   # noqa: E402

SINGLE = ["gpqa", "aime", "chess"]
LOO = {"gpqa": "loo_gpqa", "aime": "loo_aime", "chess": "loo_chess"}


def cell_accuracy(df):
    """Counter-balanced P(pick more-capable): mean over each pair's two orders."""
    df = df.assign(hit=(_picked(df) == df.label).astype(float))
    return float(df.groupby("pair_uid").hit.mean().mean())


def _read(path):
    return pd.DataFrame([json.loads(l) for l in open(path)]) if path.exists() else None


def collect(root, cfg):
    """Long table: label, train_cell, test_dom, n, accuracy (+ base floor rows)."""
    rows = []
    cells = list(cfg["cells"])
    for label in cfg["labels"]:
        for dom in SINGLE:
            base = _read(root / "base" / f"eval_{dom}_{label}.jsonl")
            if base is not None:
                rows.append(dict(label=label, train="base", test=dom,
                                 n=base.pair_uid.nunique(), acc=cell_accuracy(base)))
            for cell in cells:
                v = _read(root / f"{cell}_{label}" / f"eval_{dom}.jsonl")
                if v is not None:
                    rows.append(dict(label=label, train=cell, test=dom,
                                     n=v.pair_uid.nunique(), acc=cell_accuracy(v)))
    return pd.DataFrame(rows)


def grid(tab, label):
    """train-set x test-domain accuracy matrix for one label."""
    t = tab[tab.label == label]
    order = ["base"] + SINGLE + [LOO[d] for d in SINGLE]
    return t.pivot_table(index="train", columns="test", values="acc").reindex(
        index=[o for o in order if o in set(t.train)], columns=SINGLE)


def deltas(tab, label):
    """Per test-domain: base floor, in-domain ceiling, best cross-domain single,
    LOO (mix), and the transfer + mixing gaps."""
    g = grid(tab, label)
    out = []
    for d in SINGLE:
        ceiling = g.loc[d, d] if d in g.index else np.nan
        floor = g.loc["base", d] if "base" in g.index else np.nan
        singles = [s for s in SINGLE if s != d and s in g.index]
        best_single = g.loc[singles, d].max() if singles else np.nan
        loo = g.loc[LOO[d], d] if LOO[d] in g.index else np.nan
        out.append(dict(test=d, floor=round(floor, 3), ceiling=round(ceiling, 3),
                        best_single_transfer=round(best_single, 3), mix_loo=round(loo, 3),
                        transfer_vs_floor=round(best_single - floor, 3),
                        mixing_gain=round(loo - best_single, 3)))
    return pd.DataFrame(out)


def disagreement(root, pairs_dir, cfg):
    """On test pairs where in-domain and ECI labels disagree, does each LOO judge
    track the in-domain or the ECI target? Uses the held-out eval_mm pairs."""
    rows = []
    for dom in SINGLE:
        ind = pd.read_parquet(pairs_dir / f"pairs_{dom}_indomain.parquet")
        eci = pd.read_parquet(pairs_dir / f"pairs_{dom}_eci.parquet")
        ind = ind[ind.split == "eval_mm"][["pair_uid", "order", "label"]].rename(columns={"label": "lab_ind"})
        eci = eci[eci.split == "eval_mm"][["pair_uid", "order", "label"]].rename(columns={"label": "lab_eci"})
        m = ind.merge(eci, on=["pair_uid", "order"])
        dis = m[m.lab_ind != m.lab_eci]
        if dis.empty:
            continue
        for label in cfg["labels"]:
            v = _read(root / f"{LOO[dom]}_{label}" / f"eval_{dom}.jsonl")
            if v is None:
                continue
            vd = v.merge(dis, on=["pair_uid", "order"])
            pick = _picked(vd)
            rows.append(dict(test=dom, judge_label=label, n_disagree=len(dis) // 2,
                             tracks_indomain=round(float((pick == vd.lab_ind).mean()), 3),
                             tracks_eci=round(float((pick == vd.lab_eci).mean()), 3)))
    return pd.DataFrame(rows)


def figures(tab, size, out_dir):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyses"))
    from plots import MUTED, save, style
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(tab.label.unique()), figsize=(11, 4.2))
    axs = np.atleast_1d(axs)
    for ax, label in zip(axs, sorted(tab.label.unique())):
        g = grid(tab, label)
        im = ax.imshow(g.values, cmap="RdYlBu_r", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(g.columns)), g.columns)
        ax.set_yticks(range(len(g.index)), g.index)
        for i in range(len(g.index)):
            for j in range(len(g.columns)):
                v = g.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="#0b0b0b")
        ax.set_title(f"{label}  ({size})", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("test domain"); ax.set_ylabel("train set")
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(pick more capable)")
    fig.suptitle("Cross-domain transfer: train-set x test-domain accuracy", color="#0b0b0b")
    stub = str(out_dir / f"transfer_grid_{size}")
    save(fig, stub)
    return stub


def _verdicts_with_gap(root, pairs_dir, train, test, label):
    """Verdict rows for (train-set, test-domain) joined to the pair capability gap."""
    vp = (root / "base" / f"eval_{test}_{label}.jsonl") if train == "base" \
        else (root / f"{train}_{label}" / f"eval_{test}.jsonl")
    v = _read(vp)
    if v is None:
        return None
    caps = pd.read_parquet(pairs_dir / f"pairs_{test}_{label}.parquet")[
        ["pair_uid", "order", "cap_1", "cap_2"]]
    v = v.merge(caps, on=["pair_uid", "order"], how="left")
    v["gap"] = (v.cap_1 - v.cap_2).abs()
    v["hit"] = (_picked(v) == v.label).astype(float)
    return v


def gap_curves(root, pairs_dir, cfg, size, out_dir, tests=("gpqa", "aime", "chess")):
    """Accuracy vs capability gap, per label: one panel per test domain, one line
    per train set (base floor, in-domain single, cross-domain single, LOO mix).
    Points = binned pair means ± SE; lines = sigmoid fits (house style)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyses"))
    from analysis import gap_fit
    from plots import MUTED, PALETTE, chance, legend, save, style
    import matplotlib.pyplot as plt

    # fixed colour per TRAIN set (consistent across panels): gpqa=blue, aime=green,
    # chess=amber, any loo=pink, base=grey.
    dom_color = {"gpqa": PALETTE[0], "aime": PALETTE[2], "chess": PALETTE[3]}
    def color_of(tr):
        return MUTED if tr == "base" else PALETTE[4] if tr.startswith("loo_") else dom_color[tr]
    def name_of(tr, test):
        if tr == "base":
            return "base"
        if tr.startswith("loo_"):
            return "loo (other two)"
        return f"{tr} (in-domain)" if tr == test else tr

    for label in cfg["labels"]:
        fig, axs = plt.subplots(1, len(tests), figsize=(5.2 * len(tests), 4.0), squeeze=False)
        for ax, test in zip(axs[0], tests):
            others = [t for t in tests if t != test]
            trains = ["base", test] + others + [LOO[test]]        # floor, in-domain, cross, mix
            xmax = 0
            for tr in trains:
                v = _verdicts_with_gap(root, pairs_dir, tr, test, label)
                if v is None or v.empty:
                    continue
                pair = v.groupby("pair_uid").agg(gap=("gap", "first"), hit=("hit", "mean"))
                xmax = max(xmax, pair.gap.max())
                b = pd.cut(pair.gap, 6)
                g = pair.groupby(b, observed=True).agg(x=("gap", "mean"), p=("hit", "mean"),
                                                       n=("hit", "size"))
                se = np.sqrt(g.p * (1 - g.p) / g.n)
                ax.errorbar(g.x, g.p, yerr=se, fmt="o", ms=4, lw=0, elinewidth=0.8,
                            color=color_of(tr), zorder=3)
                a, d = gap_fit(v.gap, v.hit)
                xs = np.linspace(0, pair.gap.max(), 100)
                ax.plot(xs, 1 / (1 + np.exp(-(a + d * xs))), color=color_of(tr), lw=1.6,
                        label=f"{name_of(tr, test)} (n={pair.shape[0]})", zorder=2)
            chance(ax)
            ax.set_ylim(0, 1); ax.set_xlim(0, xmax * 1.03)
            ax.set_xlabel(f"capability gap ({'ECI' if label=='eci' else 'acc-logit'})")
            ax.set_ylabel("P(pick more capable)")
            ax.set_title(f"test = {test}", fontsize=10, color="#0b0b0b")
            style(ax); legend(ax, loc="lower right", fontsize=7)
        fig.suptitle(f"Cross-domain gap curves — {label} label ({size})", color="#0b0b0b")
        save(fig, str(out_dir / f"gap_curves_{label}_{size}"))
    print(f"wrote gap_curves_*_{size}.png")


def basic_gap_curve(root, pairs_dir, size, out_dir, label="indomain"):
    """The plain gap curve: each in-domain judge on its own domain (one coloured
    line per domain) vs the base floor. Accuracy vs capability gap, one panel."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyses"))
    from analysis import gap_fit
    from plots import MUTED, PALETTE, chance, legend, save, style
    import matplotlib.pyplot as plt

    dom_color = {"gpqa": PALETTE[0], "aime": PALETTE[2], "chess": PALETTE[3]}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    chance(ax)
    base_rows, xmax = [], 0
    for d in SINGLE:
        v = _verdicts_with_gap(root, pairs_dir, d, d, label)          # d-trained judge on d
        if v is None:
            continue
        pair = v.groupby("pair_uid").agg(gap=("gap", "first"), hit=("hit", "mean"))
        xmax = max(xmax, pair.gap.max())
        b = pd.cut(pair.gap, 6)
        g = pair.groupby(b, observed=True).agg(x=("gap", "mean"), p=("hit", "mean"), n=("hit", "size"))
        ax.errorbar(g.x, g.p, yerr=np.sqrt(g.p * (1 - g.p) / g.n), fmt="o", ms=4, lw=0,
                    elinewidth=0.8, color=dom_color[d], zorder=3)
        a, dd = gap_fit(v.gap, v.hit)
        xs = np.linspace(0, pair.gap.max(), 100)
        ax.plot(xs, 1 / (1 + np.exp(-(a + dd * xs))), color=dom_color[d], lw=1.8,
                label=f"{d} judge (n={pair.shape[0]})", zorder=2)
        bv = _verdicts_with_gap(root, pairs_dir, "base", d, label)
        if bv is not None:
            base_rows.append(bv)
    if base_rows:                                                    # one pooled floor line
        bv = pd.concat(base_rows, ignore_index=True)
        a, dd = gap_fit(bv.gap, bv.hit)
        xs = np.linspace(0, xmax, 100)
        ax.plot(xs, 1 / (1 + np.exp(-(a + dd * xs))), color=MUTED, lw=1.6, ls="--",
                label="base (no FT), pooled", zorder=2)
    ax.set_ylim(0, 1); ax.set_xlim(0, xmax * 1.03)
    ax.set_xlabel("capability gap (acc-logit)"); ax.set_ylabel("P(pick more capable)")
    ax.set_title(f"{size} in-domain judges — accuracy vs capability gap", fontsize=11, color="#0b0b0b")
    style(ax); legend(ax, loc="lower right")
    save(fig, str(out_dir / f"basic_gap_{size}"))
    print(f"wrote basic_gap_{size}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", required=True, help="Qwen3-0.6B / Qwen3-8B")
    ap.add_argument("--gap-curves", action="store_true", help="also render gpqa/aime/chess gap curves")
    ap.add_argument("--basic-gap", action="store_true", help="one-panel in-domain gap curve + base floor")
    ap.add_argument("--root", default="judge_results/judge_ft/cross")
    ap.add_argument("--pairs-dir", default="judge_results/judge_ft/cross_domain")
    ap.add_argument("--config", default=str(Path(__file__).parent / "cross_domain.yaml"))
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    root = Path(args.root) / args.size

    tab = collect(root, cfg)
    tab.to_csv(Path(args.root) / f"accuracy_long_{args.size}.csv", index=False)
    for label in cfg["labels"]:
        print(f"\n=== grid {label} ({args.size}) ===")
        print(grid(tab, label).round(3).to_string())
        print("-- deltas --")
        print(deltas(tab, label).to_string(index=False))
    print("\n=== label disagreement (LOO judges on flipped pairs) ===")
    print(disagreement(root, Path(args.pairs_dir), cfg).to_string(index=False))
    stub = figures(tab, args.size, Path(args.root))
    print(f"\nwrote {stub}.png")
    if args.gap_curves:
        gap_curves(root, Path(args.pairs_dir), cfg, args.size, Path(args.root))
    if args.basic_gap:
        basic_gap_curve(root, Path(args.pairs_dir), args.size, Path(args.root))


if __name__ == "__main__":
    main()
