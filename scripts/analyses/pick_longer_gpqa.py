"""P(pick longer transcript) on GPQA, next to length vs capability.

Left panel: per-judge probability of picking the longer of the two transcripts
(character count of the working, ANSWER line stripped), pooled over all
non-self pairs in gpqa-pilot-rd2 and data-expansion-gpqa; mean +/- t-based 95%
CI across judges; reasoning-effort variants (-low/-high) dropped as everywhere
else. Right panel: every answer model in the manuscript style GLM
(analysis/hibayes/gpqa_glm_per_model.csv, the per-model reasoning-filtered
frame) — mean transcript length from
analysis/hibayes/style_rates.csv against ECI, the GLM's capability regressor;
coloured by model family, Spearman rho on stdout.

Usage:
  uv run python scripts/analyses/pick_longer_gpqa.py
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, t as student_t

from judge.dataset import is_effort_variant, strip_answer
from mc_cost.gpqa import model_family

DATA = Path("interactive_dashboards/transcript_viewer/data")
DATASETS = ["gpqa-pilot-rd2", "data-expansion-gpqa"]
EFFORT_SUFFIXES = ("low", "high")
GLM_FRAME = Path("analysis/hibayes/gpqa_glm_per_model.csv")
STYLE_RATES = Path("analysis/hibayes/style_rates.csv")
ECI_CONFIG = Path("scripts/analyses/judge-elo/configs/gpqa.yaml")
OUT = Path("judge_results/manuscript_figures/pick_longer_gpqa.png")


def family(model):
    fam = model_family(model)
    return "llama" if fam == "meta" else fam  # Meta-Llama-* is the llama family


def transcript_chars(d):
    chars = {}
    for f in glob.glob(str(d / "transcripts/*.json")):
        model = Path(f).stem
        if model.endswith("__ep1"):
            continue
        for item, text in json.load(open(f)).items():
            chars[(model, item)] = len(strip_answer(text))
    return chars


def add_pick_longer(d, chars, tally):
    """Add this dataset's verdicts to the judge -> [picked longer, comparable]
    tally. Ties and items missing a transcript on either side are skipped."""
    for pf in glob.glob(str(d / "pairs/*__full.json")):
        major, minor, _ = Path(pf).stem.split("__")
        if major == minor:
            continue
        if any(is_effort_variant(m, EFFORT_SUFFIXES) for m in (major, minor)):
            continue
        for item, judges in json.load(open(pf)).items():
            la, lb = chars.get((major, item)), chars.get((minor, item))
            if la is None or lb is None or la == lb:
                continue
            longer_is_major = la > lb
            for judge, cell in judges.items():
                for o in cell["orders"]:
                    if o and o["pick"] in ("1", "2"):
                        tally[judge][1] += 1
                        tally[judge][0] += bool(o["picked_major"]) == longer_is_major


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        # text stays one editable <text> object per string in the svg
        "svg.fonttype": "none",
    })

    tally = defaultdict(lambda: [0, 0])
    for ds in DATASETS:
        d = DATA / ds
        add_pick_longer(d, transcript_chars(d), tally)

    fig, (ax_p, ax_len) = plt.subplots(
        1, 2, figsize=(4.8, 3.2), width_ratios=[1, 1.6],
        gridspec_kw={"bottom": 0.42, "top": 0.96, "wspace": 0.5})
    for ax in (ax_p, ax_len):
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 3))

    # left: one pooled strip, a point per judge
    rates = {j: k / n for j, (k, n) in tally.items() if n}
    order = sorted(rates, key=rates.get)
    per_judge = np.array([rates[j] for j in order])
    mean = per_judge.mean()
    ci = student_t.ppf(0.975, len(per_judge) - 1) * per_judge.std(ddof=1) / np.sqrt(len(per_judge))
    print(f"pooled: {len(per_judge)} judges, P(pick longer) mean {mean:.3f} +/- {ci:.3f}")
    for j in order:
        print(f"  {j}: {rates[j]:.3f} (n={tally[j][1]})")
    judge_colours = dict(zip(order, plt.cm.tab20.colors))
    jitter = np.linspace(-0.3, 0.3, len(order))
    for x, j in zip(jitter, order):
        ax_p.plot(x, rates[j], "o", ms=4, mew=0, color=judge_colours[j], label=j)
    ax_p.errorbar(0, mean, yerr=ci, fmt="_", color="black", capsize=0,
                  elinewidth=1, ms=10)
    ax_p.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax_p.set_xlim(-0.7, 0.7)
    ax_p.set_xticks([])
    ax_p.spines["bottom"].set_visible(False)
    ax_p.set_ylim(0.4, 1)
    ax_p.set_yticks([0.5, 0.75, 1])
    ax_p.set_ylabel("P(pick longer)")
    ax_p.legend(loc="upper left", bbox_to_anchor=(-0.35, -0.08), ncol=2,
                frameon=False, handletextpad=0.3, labelspacing=0.3,
                columnspacing=0.8)

    # right: the manuscript style-GLM pool, mean length vs ECI
    glm = pd.read_csv(GLM_FRAME, usecols=["left_model", "right_model"])
    pool = set(glm["left_model"]) | set(glm["right_model"])
    style = pd.read_csv(STYLE_RATES, usecols=["model", "n_chars"])
    eci = yaml.safe_load(ECI_CONFIG.read_text())["eci"]
    per_model = style[style["model"].isin(pool)].groupby("model")["n_chars"].mean()
    models = sorted(per_model.index)
    mean_len = per_model.loc[models].to_numpy() / 1000
    acc = np.array([eci[m] for m in models])
    rho, p = spearmanr(mean_len, acc)
    print(f"manuscript GLM pool: length vs ECI over {len(models)} models, "
          f"spearman rho={rho:.2f} (p={p:.3f})")
    # deepseek-v4-flash is not in the manuscript pool; added from the expansion
    # viewer export (full final-text lengths, ECI from the model meta)
    extra = "deepseek-v4-flash-0731"
    d = DATA / "data-expansion-gpqa"
    extra_chars = [len(t) for t in
                   json.load(open(d / f"transcripts/{extra}.json")).values()]
    models.append(extra)
    mean_len = np.append(mean_len, np.mean(extra_chars) / 1000)
    acc = np.append(acc, json.load(open(d / "index.json"))["models"][extra]["eci"])
    print(f"added {extra}: {mean_len[-1]:.2f} kchars at ECI {acc[-1]}")
    families = sorted({family(m) for m in models})
    family_colours = dict(zip(families, [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
        "#008300", "#8250c4", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22",
        "#d62728"]))
    seen = set()
    for m, y, x in zip(models, mean_len, acc):
        fam = family(m)
        ax_len.plot(x, y, "o", ms=4, mew=0, color=family_colours[fam],
                    alpha=0.85, label=None if fam in seen else fam)
        seen.add(fam)
    ax_len.set_ylim(0, 4)
    ax_len.set_yticks([0, 2, 4])
    ax_len.set_ylabel("mean transcript kchars")
    ax_len.set_xlabel("capability (ECI)")
    ax_len.legend(loc="upper center", bbox_to_anchor=(0.65, -0.18), ncol=2,
                  frameon=False, handletextpad=0.3, labelspacing=0.3,
                  columnspacing=1.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT, OUT.with_suffix(".svg")):
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
