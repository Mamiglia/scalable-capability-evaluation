"""Before/after table for the judge fine-tune: pair accuracy, implied-elo
Kendall, position bias, self-preference, GPQA answering accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression


def _picked(df):
    return np.where(df.logit_1 > df.logit_2, 1, 2)


def bt_fit(df, c):
    """Elo per model + position term from verdict rows."""
    models = sorted(set(df.model_1) | set(df.model_2))
    idx = {m: i for i, m in enumerate(models)}
    X = np.zeros((len(df), len(models)))
    for r, (m1, m2) in enumerate(zip(df.model_1, df.model_2)):
        X[r, idx[m1]] += 1
        X[r, idx[m2]] -= 1
    y = (df.logit_1 > df.logit_2).astype(int)
    clf = LogisticRegression(C=c, max_iter=1000).fit(X, y)
    return pd.Series(clf.coef_[0], index=models), float(clf.intercept_[0])


def accuracy_by_split(df):
    """Verdict accuracy per split, plus order consistency."""
    df = df.assign(picked=_picked(df))
    out = df.groupby("split").apply(
        lambda g: float((g.picked == g.label).mean()), include_groups=False)
    if "model_1" in df.columns:
        picked_model = np.where(df.picked == 1, df.model_1, df.model_2)
        agree = df.assign(pm=picked_model).groupby("pair_uid").pm.nunique() == 1
        out["order_consistency"] = float(agree.mean())
    return out


def fit_gap_model(df):
    """Logistic of picked-slot-1 on capability gap, intercept = position."""
    X = (df.cap_1 - df.cap_2).values.reshape(-1, 1)
    y = (df.logit_1 > df.logit_2).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def selfpref(df_selfpref, gap_model, self_model="qwen3-32b"):
    """Observed P(pick own side) minus capability-gap-predicted."""
    coef, intercept = gap_model
    self_is_1 = df_selfpref.model_1 == self_model
    picked = _picked(df_selfpref)
    observed = float((picked == np.where(self_is_1, 1, 2)).mean())
    p1 = 1 / (1 + np.exp(-(coef * (df_selfpref.cap_1 - df_selfpref.cap_2) + intercept)))
    predicted = float(np.where(self_is_1, p1, 1 - p1).mean())
    return observed - predicted


def read_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])


def gap_fit(gap, correct):
    """Sigmoid(a + d * gap) fit, same form as the pilot judge fits."""
    clf = LogisticRegression(max_iter=1000).fit(
        np.asarray(gap).reshape(-1, 1), np.asarray(correct).astype(int))
    return float(clf.intercept_[0]), float(clf.coef_[0][0])


def gap_curve(dfs, judge_fits, refs, out_stub, selfs=None, n_bins=8):
    """Judge accuracy vs raw-accuracy capability gap, house style.
    dfs: {label: verdicts df with acc_1/acc_2 author accuracies}.
    judge_fits: pilot judge sigmoid fits; refs: judge names drawn as grey
    reference curves. selfs: {label: same-model control verdicts}, drawn as
    open markers in a broken-axis section (should sit at chance)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "analyses"))
    from plots import INK2, MUTED, PALETTE, chance, legend, save, style
    import matplotlib.pyplot as plt

    variant_style = {
        "base": dict(color=PALETTE[0], marker="o", ls="--"),
        "fine-tuned": dict(color=PALETTE[1], marker="s", ls="-"),
        "fine-tuned+hard": dict(color=PALETTE[2], marker="^", ls="-."),
    }
    pooled = np.concatenate([np.abs(d.acc_1 - d.acc_2).values for d in dfs.values()])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    # axis and fit lines stop at the highest binned point, not the raw max
    x_top = max(pd.Series(np.abs(d.acc_1 - d.acc_2)).groupby(
        pd.cut(np.abs(d.acc_1 - d.acc_2), edges, include_lowest=True),
        observed=True).mean().max() for d in dfs.values())
    xs = np.linspace(0, x_top, 200)

    if selfs:
        fig, (axs, ax) = plt.subplots(1, 2, sharey=True, figsize=(6.2, 3.4),
                                      gridspec_kw={"width_ratios": [1, 11], "wspace": 0.05})
        for i, (name, df) in enumerate(selfs.items()):
            st = variant_style[name]
            p = float((_picked(df) == df.label).mean())
            se = np.sqrt(p * (1 - p) / len(df))
            x = (i - (len(selfs) - 1) / 2) * 0.5
            axs.errorbar(x, p, yerr=1.96 * se, fmt=st["marker"], ms=4.5,
                         color=st["color"], ecolor=st["color"], elinewidth=0.9,
                         capsize=0, mfc="white", mew=1.1, zorder=3)
        chance(axs)
        axs.set_xlim(-0.7, 0.7)
        axs.set_xticks([0])
        axs.set_xticklabels(["self\npairs"])
        style(axs)
        for a, x0 in ((axs, 1.0), (ax, 0.0)):
            a.plot([x0, x0], [-0.015, 0.015], transform=a.transAxes, color=MUTED,
                   lw=0.9, clip_on=False, marker=[(-0.5, -1), (0.5, 1)], ms=5, mew=0.9)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        axs.set_ylabel("judge accuracy")
    else:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
    chance(ax)
    for name, df in dfs.items():
        st = variant_style[name]
        gap = np.abs(df.acc_1 - df.acc_2)
        correct = (_picked(df) == df.label).astype(float)
        binned = pd.DataFrame(dict(gap=gap, correct=correct))
        binned["bin"] = pd.cut(binned.gap, edges, include_lowest=True)
        g = binned.groupby("bin", observed=True).agg(
            x=("gap", "mean"), p=("correct", "mean"), n=("correct", "size"))
        se = np.sqrt(g.p * (1 - g.p) / g.n)
        a, d = gap_fit(gap, correct)
        ax.errorbar(g.x, g.p, yerr=se, fmt=st["marker"], color=st["color"],
                    ms=4.5, capsize=0, lw=0, elinewidth=0.9, zorder=3)
        ax.plot(xs, 1 / (1 + np.exp(-(a + d * xs))), color=st["color"],
                ls=st["ls"], lw=1.6, zorder=2,
                label=f"qwen3-8b {name} (n={len(df)})")

    # pilot-round zero-shot judge references, full judge prompt
    pilot_max_gap = 0.32
    px = np.linspace(0, pilot_max_gap, 100)
    fits = judge_fits.set_index("judge")
    for j, color, dash in zip(refs, [INK2, MUTED], [(0, (4, 2)), (0, (1, 2))]):
        r = fits.loc[j]
        ax.plot(px, 1 / (1 + np.exp(-(r.a + r.d * px))), color=color, ls=dash,
                lw=1.3, zorder=2, label=f"{j} (pilot zero-shot)")

    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(0, x_top * 1.02)
    ax.set_xlabel("capability gap (GPQA accuracy difference)")
    if not selfs:
        ax.set_ylabel("judge accuracy")
    style(ax)
    legend(ax, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.text(0.98, 0.04,
            "points: quantile-bin means ± SE, held-out eval pairs · lines: sigmoid fits\n"
            "pilot references: full-prompt zero-shot judges, drawn over the pilot gap range",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=6.5)
    save(fig, out_stub)


def add_caps(df, caps):
    return df.assign(cap_1=df.model_1.map(caps), cap_2=df.model_2.map(caps))


def gpqa_metrics(df, holdout_items):
    hold = df.item.isin(holdout_items)
    out = {}
    for name, sub in [("gpqa_holdout_items", df[hold]), ("gpqa_train_items_leaky", df[~hold])]:
        p, n = sub.correct.mean(), len(sub)
        out[name] = p
        out[name + "_se"] = np.sqrt(p * (1 - p) / n) if n else np.nan
    out["gpqa_unfinished"] = float((~df.finished).mean())
    return out


def metrics_for(verdicts, selfpref_df, gpqa_df, model_caps, pilot_caps, holdout_items, bt_c):
    out = accuracy_by_split(verdicts).add_prefix("acc_").to_dict()
    elo, pos = bt_fit(verdicts, c=bt_c)
    caps = elo.index.to_series().map(model_caps)
    out["kendall_elo_vs_capability"] = kendalltau(elo.values, caps.values).statistic
    out["position_term"] = pos
    out["p_pick_slot1"] = float((_picked(verdicts) == 1).mean())
    gap_model = fit_gap_model(add_caps(verdicts, model_caps))
    out["selfpref_excess"] = selfpref(add_caps(selfpref_df, pilot_caps), gap_model)
    out.update(gpqa_metrics(gpqa_df, holdout_items))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ["base-verdicts", "ft-verdicts", "base-selfpref", "ft-selfpref",
                 "base-gpqa", "ft-gpqa"]:
        ap.add_argument(f"--{flag}", required=True)
    for flag in ["hard2-verdicts", "hard2-selfpref", "hard2-gpqa",
                 "base-self", "ft-self"]:
        ap.add_argument(f"--{flag}", default=None)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["out_dir"])

    model_caps = pd.read_csv(out_dir / "models_ft.csv").set_index("model").logit
    pilot_caps = pd.read_csv(cfg["pilot_models_csv"]).set_index("model").logit
    item_split = pd.read_csv(out_dir / "item_split.csv")
    holdout_items = set(item_split.item[item_split.holdout])

    rows = {}
    tags = ["base", "ft"] + (["hard2"] if args.hard2_verdicts else [])
    for tag in tags:
        rows[tag] = metrics_for(
            read_jsonl(getattr(args, f"{tag}_verdicts")),
            read_jsonl(getattr(args, f"{tag}_selfpref")),
            read_jsonl(getattr(args, f"{tag}_gpqa")),
            model_caps, pilot_caps, holdout_items, cfg["analysis"]["bt_c"])
    selfs = None
    if args.base_self:
        selfs = {"base": read_jsonl(args.base_self), "fine-tuned": read_jsonl(args.ft_self)}
        for tag, df in zip(["base", "ft"], selfs.values()):
            p = float((_picked(df) == df.label).mean())
            rows[tag]["selfpair_pick_rate"] = p
            rows[tag]["selfpair_pick_rate_se"] = float(np.sqrt(p * (1 - p) / len(df)))
    table = pd.DataFrame(rows).round(4)
    table.index.name = "metric"
    print(table.to_string())
    table.to_csv(out_dir / "summary.csv")

    model_accs = pd.read_csv(out_dir / "models_ft.csv").set_index("model").accuracy

    def with_accs(df):
        return df.assign(acc_1=df.model_1.map(model_accs), acc_2=df.model_2.map(model_accs))

    curves = {"base": with_accs(read_jsonl(args.base_verdicts)),
              "fine-tuned": with_accs(read_jsonl(args.ft_verdicts))}
    judge_fits = pd.concat([pd.read_csv(cfg["analysis"]["judge_fits_csv"]),
                            pd.read_csv(cfg["analysis"]["judge_fits_fallback_csv"])]
                           ).drop_duplicates("judge", keep="first")
    gap_curve(curves, judge_fits, cfg["analysis"]["ref_judges"], out_dir / "gap_curve",
              selfs=selfs)
    print(f"wrote {out_dir / 'gap_curve.png'}")


if __name__ == "__main__":
    main()
