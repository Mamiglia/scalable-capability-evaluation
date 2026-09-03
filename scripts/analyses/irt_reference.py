"""IRT theta as reference for the rank pipeline, and BT ability as a theta estimate (Q_src = 1 runs).

Figures in analysis/irt_reference/:
  01_reference_equivalence  Q2-Q4 accuracy vs theta per model; judge tau under either reference
  02_bt_vs_theta            calibrated BT ability (a*BT + c, fit on Q1 responses) vs theta
  03_predicted_by_release   actual vs BT-predicted Q2-Q4 accuracy per model, by release date
Theta per model = 1PL-with-guessing MLE over all items, item b fixed from the domain fit
(data/meta/domains/<domain>/items.csv). Predicted accuracy = mean over Q2-Q4 items of
guess + (1-guess) * sigmoid(theta_hat - b).
Run:  uv run python scripts/analyses/irt_reference.py
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import kendalltau, pearsonr

from configs.model import ModelConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "irt_reference"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})
JUDGE_COLOR = {"deepseek-r1-0528": "#eb6834", "gemma-4-31b-it": "#1baf7a", "gpt-5.2": "#eda100",
               "o4-mini": "#e87ba4"}
SHORT = {"deepseek-r1-0528": "ds-r1", "gemma-4-31b-it": "gemma", "gpt-5.2": "gpt-5.2", "o4-mini": "o4-mini"}
HARD = [2, 3, 4]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 4))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def logistic(theta, b, guess):
    return guess + (1 - guess) / (1 + np.exp(-(theta - b)))


def nll(theta, sub):
    p = np.clip(logistic(theta, sub.b, sub.guess), 1e-9, 1 - 1e-9)
    return -(sub.correct * np.log(p) + (1 - sub.correct) * np.log(1 - p)).sum()


def load_domain(domain):
    items = pd.read_csv(ROOT / f"data/meta/domains/{domain}/items.csv", dtype={"item": str})
    resp = pd.read_parquet(ROOT / f"data/meta/domains/{domain}/responses.parquet")
    resp = resp.merge(items[["bench", "item", "b", "quartile"]])
    resp["correct"] = resp.correct.astype(float)
    return resp


def accuracy(sub):
    return sub.groupby(["model", "bench", "item"]).correct.mean().groupby("model").mean()


def theta_all(resp):
    return resp.groupby("model").apply(
        lambda s: minimize_scalar(lambda t: nll(t, s) + t ** 2 / 8, bounds=(-8, 8), method="bounded").x)


def predicted_accuracy(theta, sub):
    it = sub.groupby(["bench", "item"])[["b", "guess"]].first()
    return logistic(theta, it.b, it.guess).mean()


def calibrate(bt, resp):
    """theta_hat = a*bt + c, (a, c) by max likelihood on Q1 responses only."""
    q1 = resp[(resp.quartile == 1) & resp.model.isin(bt.index)]
    s = bt.loc[q1.model].values
    a, c = minimize(lambda p: nll(p[0] * s + p[1], q1), [1, 0], method="Nelder-Mead").x
    return a * bt + c


def runs():
    for d in sorted(glob.glob(str(ROOT / "logs/rank/*/*/*") + "/")):
        evals = sorted(glob.glob(d + "*.eval"))
        if not evals or not Path(d, "ranking.csv").exists():
            continue
        md = read_eval_log(evals[0], header_only=True).eval.metadata
        if md["q_src"] != 1:
            continue
        judge = Path(d).parent.name
        yield {"domain": md["domain"], "judge": judge, "cot": md["include_reasoning"],
               "bt": pd.read_csv(d + "ranking.csv").set_index("model").ability,
               "label": judge + (", CoT" if md["include_reasoning"] else ""),
               "tag": md["domain"] + " · " + SHORT[judge] + (" CoT" if md["include_reasoning"] else "")}


def release_dates():
    return {ModelConfig.from_cli(p).shortname.lower(): pd.to_datetime(ModelConfig.from_cli(p).release_date)
            for p in sorted(glob.glob(str(ROOT / "configs/models/*/*.yml")))}


def fig_reference_equivalence(rows, ref):
    fig, axes = plt.subplots(1, 3, figsize=(3.8, 1.35), gridspec_kw={"wspace": 0.75})
    for ax, dom in zip(axes[:2], ["coding", "math"]):
        r = ref[dom]
        ax.scatter(r.acc, r.theta, s=8, color="#2a78d6", lw=0)
        ax.set_title(f"{dom}  τ = {kendalltau(r.acc, r.theta).statistic:.2f}", loc="left", fontsize=7,
                     color="grey")
        ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", ".5", "1"])
        ax.set_ylim(-4, 4); ax.set_yticks([-4, 0, 4])
        ax.set_xlabel("Q2–Q4 accuracy")
        style(ax)
    axes[0].set_ylabel("IRT θ")
    ax = axes[2]
    ax.plot([0.3, 0.8], [0.3, 0.8], color="grey", lw=0.8, ls="--")
    for r in rows:
        ref_r = ref[r["domain"]].loc[r["bt"].index]
        x, y = (kendalltau(r["bt"], ref_r[c]).statistic for c in ("acc", "theta"))
        ax.plot(x, y, "o", ms=4, color=JUDGE_COLOR[r["judge"]], mfc="none" if r["cot"] else None, mew=1)
    ax.set_xlim(0.3, 0.8); ax.set_ylim(0.3, 0.8); ax.set_xticks([0.3, 0.55, 0.8]); ax.set_yticks([0.3, 0.55, 0.8])
    ax.set_xlabel("τ vs Q2–Q4 acc."); ax.set_ylabel("τ vs θ")
    ax.set_title("judge τ", loc="left", fontsize=7, color="grey")
    style(ax)
    save(fig, "01_reference_equivalence")


def fig_bt_vs_theta(rows, ref, cal):
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.6), sharey=True, gridspec_kw={"wspace": 0.25})
    for ax, dom in zip(axes, ["coding", "math"]):
        ax.plot([-4, 4], [-4, 4], color="grey", lw=0.8, ls="--")
        for r, th in zip(rows, cal):
            if r["domain"] != dom:
                continue
            true = ref[dom].theta.loc[th.index]
            ax.scatter(true, th, s=7, color=JUDGE_COLOR[r["judge"]], lw=0 if not r["cot"] else 0.6,
                       facecolor="none" if r["cot"] else None, edgecolor=JUDGE_COLOR[r["judge"]],
                       label=f"{r['label']} (r = {pearsonr(true, th)[0]:.2f})")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
        ax.set_xlabel("IRT θ")
        ax.set_title(dom, loc="left", fontsize=7, color="grey")
        style(ax)
    axes[0].set_ylabel("Calibrated BT ability")
    h, l = [], []
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels(); h += hh; l += ll
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, handletextpad=0.2)
    save(fig, "02_bt_vs_theta")


def fig_predicted_by_release(rows, cal, resps, dates):
    fig, axes = plt.subplots(2, 3, figsize=(4.4, 3.0), sharex=True, sharey=True)
    for ax, r, th in zip(axes.flat, rows, cal):
        sub = resps[r["domain"]][resps[r["domain"]].quartile.isin(HARD)]
        true = accuracy(sub).loc[th.index]
        pred = pd.Series({m: predicted_accuracy(th[m], sub) for m in th.index})
        x = pd.Series({m: dates[m] for m in th.index})
        ax.vlines(x, true, pred, color="grey", lw=0.6)
        ax.scatter(x, true, s=7, color="black", lw=0, label="actual", zorder=3)
        ax.scatter(x, pred, s=7, color=JUDGE_COLOR[r["judge"]], lw=0, zorder=3)
        ax.scatter([], [], s=7, color="#888", lw=0, label="predicted from BT")
        ax.set_title(f"{r['tag']}   MAE {np.abs(pred - true).mean():.2f}".replace("0.", "."),
                     loc="left", fontsize=6, color="grey")
        ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", ".5", "1"])
        style(ax)
    for ax in axes[1]:
        ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-01"))
        ax.set_xticks(pd.to_datetime(["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"]))
        ax.set_xticklabels(["2025", "Jul", "2026", "Jul"])
    fig.supylabel("Q2–Q4 accuracy", fontsize=8); fig.supxlabel("Release date", fontsize=8)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save(fig, "03_predicted_by_release")


if __name__ == "__main__":
    rows = list(runs())
    resps = {dom: load_domain(dom) for dom in {r["domain"] for r in rows}}
    ref = {dom: pd.DataFrame({"acc": accuracy(resp[resp.quartile.isin(HARD)]), "theta": theta_all(resp)})
           for dom, resp in resps.items()}
    cal = [calibrate(r["bt"], resps[r["domain"]]) for r in rows]
    dates = release_dates()
    fig_reference_equivalence(rows, ref)
    fig_bt_vs_theta(rows, ref, cal)
    fig_predicted_by_release(rows, cal, resps, dates)
    print(f"{len(rows)} runs -> {OUT}")
