"""Cumulative curve, calibrated: shape from sentence effects, scale from whole-span
measurements. Each span's sentence effects are scaled by one global factor so the
span's total equals its measured whole-span contribution."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge-whitebox")

S = pathlib.Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/8727e515-2405-4685-a43d-ead3d07c1e1c/scratchpad")
OUT = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/whitebox_dissection/figures/llama3.2-3B")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE = "#2a78d6"

scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
bg = pd.concat([pd.read_parquet(p) for p in sorted((S / "blankgroups_llama").rglob("part_*.parquet"))],
               ignore_index=True)
sp = pd.concat([pd.read_parquet(p) for p in sorted((S / "slotpref_full_llama").rglob("part_*.parquet"))])
sp["f_blank"] = sp["gap"] - sp["delta"]
blank = sp.set_index(["pair_uid", "order"])["f_blank"]

measured = bg.groupby("group").delta.mean()
summed = scores.groupby("group").score.sum() / scores.groupby(["pair_uid", "order"]).ngroups
factor = (measured / summed).to_dict()
print("calibration factors:", {k: round(v, 3) for k, v in factor.items()})
scores = scores.assign(cal=scores.score * scores.group.map(factor))

f_mean = bg.groupby(["pair_uid", "order"]).gap.first().mean()
grid = np.linspace(0, 1, 201)
curves, group_starts = [], {g: [] for g in scores.group.unique()}
for (uid, order), g in scores.groupby(["pair_uid", "order"]):
    g = g.sort_values("unit")
    ends = g.n_tokens.cumsum() / g.n_tokens.sum()
    start = blank.get((uid, order), 0.012)
    cum = start + g.cal.cumsum()
    curves.append(np.interp(grid, np.concatenate([[0], ends]), np.concatenate([[start], cum])))
    rel = (g.n_tokens.cumsum() - g.n_tokens) / g.n_tokens.sum()
    for grp in group_starts:
        sel = g.group.values == grp
        if sel.any():
            group_starts[grp].append(rel.values[sel][0])
curves = np.asarray(curves)
print("median curve end:", round(float(np.median(curves[:, -1])), 3),
      "| actual mean verdict:", round(float(f_mean), 3))

fig, (ax, axh) = plt.subplots(1, 2, figsize=(4.6, 2.6), sharey=True,
                              gridspec_kw={"width_ratios": [3.2, 0.9], "wspace": 0.06})
starts = sorted(((np.median(v), k) for k, v in group_starts.items() if v))
bounds = [s for s, _ in starts] + [1.0]
labels = {"instr_format": "instructions", "slot_labels": "", "question": "",
          "transcript_1": "transcript 1", "transcript_2": "transcript 2", "answer_readout": ""}
lim = 6.0
SHADE = {"transcript_1": ("0.82", 0.55), "transcript_2": ("0.62", 0.45)}
for i, (s, name) in enumerate(starts):
    if name in SHADE:
        colour, alpha = SHADE[name]
        ax.axvspan(bounds[i], bounds[i + 1], color=colour, alpha=alpha, lw=0, zorder=0)
    if labels[name] and bounds[i + 1] - bounds[i] > 0.12:
        ax.text((bounds[i] + bounds[i + 1]) / 2, lim * 1.02, labels[name],
                ha="center", va="bottom", fontsize=6, color="0.3")
sub = curves[np.random.default_rng(0).choice(len(curves), 300, replace=False)]
for c in sub:
    ax.plot(grid, c, color=BLUE, lw=0.3, alpha=0.06)
ax.plot(grid, curves.mean(axis=0), color="#1b4f9c", lw=1.5,
        label="verdict forming while reading (mean)")

# tracked examples: two per verdict, strong and axis-contained, endpoint sign agreeing
gap_by_key = scores.groupby(["pair_uid", "order"]).gap.first()
rng_t = np.random.default_rng(7)
LIGHT, DARK = "#8ab8e8", "#1b4f9c"
picked = {1: [], 2: []}
for i in rng_t.permutation(len(gap_by_key)):
    g = float(gap_by_key.iloc[i])
    side = 1 if g > 0 else 2
    if (abs(g) > 2 and abs(g) < lim and np.sign(curves[i, -1]) == np.sign(g)
            and np.abs(curves[i]).max() < lim and len(picked[side]) < 2):
        picked[side].append(i)
    if len(picked[1]) == 2 and len(picked[2]) == 2:
        break
examples = picked[1] + picked[2]
for j, i in enumerate(examples):
    ax.plot(grid, curves[i], color="#8ab8e8", lw=0.8, alpha=0.95,
            label="example prompts" if j == 0 else None)
    ax.plot(1.0, curves[i, -1], "o", color="#8ab8e8", ms=3, clip_on=False)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axhline(f_mean, color="k", lw=1, ls=":", label="actual mean verdict")
ax.set_xlabel("relative position in prompt")
ax.set_ylabel("log-odds towards transcript 1")
ax.set_xlim(0, 1)
ax.set_ylim(-lim, lim)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=6)

verdicts = bg.groupby(["pair_uid", "order"]).gap.first()
axh.hist(verdicts.clip(-lim, lim), bins=40, orientation="horizontal",
         color=BLUE, alpha=0.85)
axh.axhline(0, color="0.6", lw=0.8, ls="--")
axh.axhline(f_mean, color="k", lw=1, ls=":")
axh.set_xlabel("final verdicts\n(prompts)")
axh.set_xticks([])
axh.spines["left"].set_visible(False)
axh.tick_params(left=False)
share2 = float((verdicts < 0).mean())
print("share of verdicts favouring transcript 2: %.3f" % share2)
fig.tight_layout()
fig.savefig(OUT / "fig6_cumulative.png", dpi=300, bbox_inches="tight")
print("saved")
