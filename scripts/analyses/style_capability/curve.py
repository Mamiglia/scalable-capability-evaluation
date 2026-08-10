"""Capability-gap curve with spoofing overlaid. For every pair of the 4 clean
models, P(judge picks the truly-stronger model) is measured three ways:
  original            both models in their own (baseline) style
  weak_spoofs_strong  the weaker model wears the stronger's style
  strong_spoofs_weak  the stronger model wears the weaker's style
x-axis = true capability gap (pilot logit difference); default judge prompt.

  --judge   run the judging (spend); reuses judge_task
  --plot    read verdict logs, print a table, write the figure
"""

import argparse
import glob
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log
from judge.task import judge_task

EVALS = "data/capability_spoofing/evals"
LOGROOT = "logs/style_capability_curve"
FIG = "judge_results/style_capability/capability_gap_curve.png"
MODELS = ["qwen3.5-122b-a10b", "gemma-4-26b-a4b-it", "qwen3-32b", "gpt-4o"]
JUDGES = ["openrouter/openai/gpt-4.1-nano",
          "openrouter/google/gemini-3.1-flash-lite",
          "openrouter/google/gemma-4-31b-it"]
JARGS = {
    "openrouter/openai/gpt-4.1-nano": {"provider": {"data_collection": "deny", "require_parameters": True}},
    "openrouter/google/gemini-3.1-flash-lite": {"provider": {"data_collection": "deny", "require_parameters": True}},
    "openrouter/google/gemma-4-31b-it": {"provider": {"only": ["siliconflow/fp8"], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}},
}
CONDS = ["original", "weak_spoofs_strong", "strong_spoofs_weak"]
SPOOF = "copy"  # "copy" = example transcripts; "instruct" = confident/uncertain instruction


def caps():
    d = pd.read_csv("judge_results/gpqa-pilot-rd2/models.csv").set_index("model")["accuracy"]
    return {m: float(d[m]) for m in MODELS}


def cond_path(name):
    hits = sorted(glob.glob(f"{EVALS}/{name}/*.eval"))
    return hits[-1] if hits else None


def comparisons():
    """(gap, condition, major=stronger-side, minor=weaker-side) per pair."""
    L = caps()
    out = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            strong, weak = (a, b) if L[a] >= L[b] else (b, a)
            gap = abs(L[a] - L[b])
            if SPOOF == "instruct":
                up, down = f"{weak}__confident", f"{strong}__uncertain"
            else:
                up, down = f"{weak}__copy_{strong}", f"{strong}__copy_{weak}"
            out.append((gap, "original", f"{strong}__baseline", f"{weak}__baseline"))
            out.append((gap, "weak_spoofs_strong", f"{strong}__baseline", up))
            out.append((gap, "strong_spoofs_weak", down, f"{weak}__baseline"))
    return out


def pick_stronger(logpath):
    log = read_eval_log(logpath)
    by = defaultdict(list)
    for s in log.samples:
        sc = list(s.scores.values())[0]
        if sc.answer in ("1", "2"):
            by[s.id.rsplit("_", 1)[0]].append(1.0 if sc.answer == s.target else 0.0)
    return [statistics.mean(v) for v in by.values() if v]


def run_judge():
    for gap, cond, major, minor in comparisons():
        mp, np_ = cond_path(major), cond_path(minor)
        if not mp or not np_:
            print("MISSING", major, "or", minor)
            continue
        for judge in JUDGES:
            js = judge.split("/")[-1]
            task = judge_task(mp, np_, include_reasoning=False,
                              task_name=f"{js}_{cond}_{major}__vs__{minor}",
                              experiment_name="style_capability_curve")
            # timeout+retries: gemini-flash-lite occasionally hangs a streaming
            # read forever; a per-call timeout aborts it so eval_set retries.
            eval_set(tasks=[task], model=judge, model_args=JARGS.get(judge, {}),
                     log_dir=f"{LOGROOT}/{cond}/{js}/{major}__vs__{minor}",
                     max_connections=40, timeout=90, max_retries=5)
        print("done", cond, major, "vs", minor, flush=True)


def _item_means(byic, key):
    """Per-item pick rate: for each item, mean picked_major over all judges +
    both slot orders. The item is the independent unit."""
    return [statistics.mean(v) for v in byic.get(key, {}).values() if v]


def _collect(by_judge=False):
    """{(gap,cond[,judge]) -> {item -> [picks]}} over the verdict logs."""
    byic = defaultdict(lambda: defaultdict(list))
    for gap, cond, major, minor in comparisons():
        for judge in JUDGES:
            js = judge.split("/")[-1]
            for lp in glob.glob(f"{LOGROOT}/{cond}/{js}/{major}__vs__{minor}/*.eval"):
                log = read_eval_log(lp)
                for s in log.samples:
                    sc = list(s.scores.values())[0]
                    if sc.answer in ("1", "2"):
                        item = s.id.rsplit("_", 1)[0]
                        key = (js, round(gap, 4), cond) if by_judge else (round(gap, 4), cond)
                        byic[key][item].append(1.0 if sc.answer == s.target else 0.0)
    return byic


def plot():
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))  # scripts/analyses -> plots.py
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import INK2, MUTED, PALETTE, WIDE, chance, legend, save, style

    byic = _collect()
    gaps = sorted({g for g, _ in byic})
    print(f"{'gap':>6}  " + "  ".join(f"{c:>20}" for c in CONDS))
    for g in gaps:
        row = f"{g:6.3f}  "
        for c in CONDS:
            im = _item_means(byic, (g, c))
            row += f"{statistics.mean(im):>13.2f} (n{len(im):>3})" if im else f"{'-':>20}"
        print(row)

    color = {"original": INK2, "weak_spoofs_strong": PALETTE[0], "strong_spoofs_weak": PALETTE[1]}
    label = {"original": "original (both natural)",
             "weak_spoofs_strong": "weak mimics strong style" if SPOOF == "copy" else "weak told to sound confident",
             "strong_spoofs_weak": "strong mimics weak style" if SPOOF == "copy" else "strong told to sound uncertain"}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, c in enumerate(CONDS):
        dx = (i - 1) * 0.005  # jitter the 3 lines so their whiskers separate
        xs, ys, lo, hi = [], [], [], []
        for g in gaps:
            im = _item_means(byic, (g, c))
            if not im:
                continue
            a = np.array(im)
            boot = [rng.choice(a, a.size, replace=True).mean() for _ in range(2000)]
            xs.append(g + dx)
            ys.append(a.mean())
            lo.append(np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5))
        for x, l, h in zip(xs, lo, hi):
            ax.plot([x, x], [l, h], color=color[c], lw=1.0, alpha=0.7, zorder=2,
                    solid_capstyle="butt")
        ax.plot(xs, ys, "-o", color=color[c], ms=4.5, mew=1.1, mfc=color[c],
                mec=color[c], zorder=3, label=label[c])
    chance(ax, 0.5, axis="y")
    ax.set_xlabel("pair accuracy gap |d_acc|")
    ax.set_ylabel("P(judge picks the stronger model)")
    # instruct's strong-uncertain line collapses to ~0; drop the floor so it shows
    ax.set_ylim(-0.02 if SPOOF == "instruct" else 0.25, 1.03)
    ax.text(0, -0.19, "one point per model pair (x jittered per line) · whiskers: 95% CI, "
            "bootstrap over items · dashed = chance",
            transform=ax.transAxes, va="top", color=MUTED, fontsize=7)
    # copy: lower-right is empty. instruct: the uncertain line sits there, so
    # tuck the legend into the empty band above it (below the dashed chance line).
    if SPOOF == "instruct":
        legend(ax, loc="center right", bbox_to_anchor=(1.0, 0.42))
    else:
        legend(ax, loc="lower right")
    style(ax)
    save(fig, FIG[:-4] if FIG.endswith(".png") else FIG)
    print(f"\nwrote {FIG} (+ .svg)")


def plot_by_judge():
    """Same curve, one panel per judge."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import matplotlib.pyplot as plt
    from plots import INK2, MUTED, PALETTE, WIDE, chance, legend, save, style

    byic = _collect(by_judge=True)
    judges = [j.split("/")[-1] for j in JUDGES]
    gaps = sorted({g for _, g, _ in byic})
    color = {"original": INK2, "weak_spoofs_strong": PALETTE[0], "strong_spoofs_weak": PALETTE[1]}
    label = {"original": "original",
             "weak_spoofs_strong": "weak mimics strong" if SPOOF == "copy" else "weak: confident",
             "strong_spoofs_weak": "strong mimics weak" if SPOOF == "copy" else "strong: uncertain"}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(judges), figsize=(2.2 * len(judges) + 0.6, 2.7), sharey=True)
    for ax, js in zip(np.atleast_1d(axes), judges):
        for i, c in enumerate(CONDS):
            dx = (i - 1) * 0.006
            xs, ys, lo, hi = [], [], [], []
            for g in gaps:
                im = _item_means(byic, (js, g, c))
                if not im:
                    continue
                a = np.array(im)
                boot = [rng.choice(a, a.size, replace=True).mean() for _ in range(1000)]
                xs.append(g + dx); ys.append(a.mean())
                lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
            for x, l, h in zip(xs, lo, hi):
                ax.plot([x, x], [l, h], color=color[c], lw=0.9, alpha=0.5, zorder=2)
            ax.plot(xs, ys, "-o", color=color[c], ms=3, mfc=color[c], mec=color[c],
                    lw=1.2, zorder=3, label=label[c])
        chance(ax, 0.5, axis="y")
        ax.set_title(js, fontsize=8, color=INK2)
        ax.set_ylim(-0.02 if SPOOF == "instruct" else 0.25, 1.05)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("P(pick stronger)")
    fig.supxlabel("pair accuracy gap |d_acc|", fontsize=8, color=INK2)
    legend(np.atleast_1d(axes)[-1], loc="lower right")
    stub = (FIG[:-4] if FIG.endswith(".png") else FIG) + "_by_judge"
    save(fig, stub)
    print(f"wrote {stub}.png (+ .svg)")


def main():
    global EVALS, LOGROOT, FIG, MODELS, SPOOF
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--evals", default=EVALS)
    ap.add_argument("--logroot", default=LOGROOT)
    ap.add_argument("--fig", default=FIG)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--spoof", default=SPOOF, choices=["copy", "instruct"])
    args = ap.parse_args()
    EVALS, LOGROOT, FIG = args.evals, args.logroot, args.fig
    MODELS = args.models.split(",")
    SPOOF = args.spoof
    if args.judge:
        run_judge()
    if args.plot or not args.judge:
        plot()
        plot_by_judge()


if __name__ == "__main__":
    main()
