"""Figures for judge_results/whitebox_dissection: Llama token-attribution results."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge-whitebox")
import mask_summary

S = pathlib.Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/8727e515-2405-4685-a43d-ead3d07c1e1c/scratchpad")
OUT = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/whitebox_dissection")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a78d6", "#eb6834"

scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
bg = pd.concat([pd.read_parquet(p) for p in sorted((S / "blankgroups_llama").rglob("part_*.parquet"))],
               ignore_index=True)
sp = pd.concat([pd.read_parquet(p) for p in sorted((S / "slotpref_full_llama").rglob("part_*.parquet"))])
sp["f_blank"] = sp["gap"] - sp["delta"]

SPANS = [("transcript_1", "transcript 1"), ("transcript_2", "transcript 2"),
         ("instr_format", "instructions"), ("slot_labels", "slot labels"),
         ("question", "question"), ("answer_readout", "answer readout")]

# ---------------------------------------------------------------- span effects
sum_ci, meas_ci = {}, {}
for grp, _ in SPANS:
    g = scores[scores.group == grp].groupby(["item", "pair_uid", "order"]).score.sum().reset_index()
    sum_ci[grp] = mask_summary.item_bootstrap(g, lambda f: f["score"].mean(), 500)
    m = bg[bg.group == grp].groupby(["item", "pair_uid", "order"]).delta.first().reset_index()
    meas_ci[grp] = mask_summary.item_bootstrap(m, lambda f: f["delta"].mean(), 500)

fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.1), sharey=True)
y = np.arange(len(SPANS))[::-1]
for ax, ci, xlabel in ((axes[0], sum_ci, "summed sentence scores"),
                       (axes[1], meas_ci, "measured ΔF, whole span blanked")):
    est = [ci[g]["estimate"] for g, _ in SPANS]
    lo = [ci[g]["estimate"] - ci[g]["lo"] for g, _ in SPANS]
    hi = [ci[g]["hi"] - ci[g]["estimate"] for g, _ in SPANS]
    ax.errorbar(est, y, xerr=[lo, hi], fmt="o", ms=3.5, color=BLUE, lw=1, capsize=0)
    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
axes[0].set_yticks(y, [n for _, n in SPANS])
fig.tight_layout()
fig.savefig(OUT / "fig_span_effects.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- length bias
slots = scores[scores.group.isin(["transcript_1", "transcript_2"])]
per = slots.groupby(["item", "pair_uid", "order", "group"]).agg(
    total=("score", "sum"), toks=("n_tokens", "sum")).reset_index()
t1 = per[per.group == "transcript_1"].set_index(["pair_uid", "order"])
t2 = per[per.group == "transcript_2"].set_index(["pair_uid", "order"])
w = pd.DataFrame({"t1": t1.total, "t2": t2.total, "n1": t1.toks, "n2": t2.toks}).dropna()
w["dlen"], w["net"] = w.n1 - w.n2, w.t1 + w.t2

fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
ax = axes[0]
ax.scatter(w.n1, w.t1, s=2, alpha=0.12, color=BLUE, label="transcript 1", lw=0)
ax.scatter(w.n2, -w.t2, s=2, alpha=0.12, color=ORANGE, label="transcript 2", lw=0)
for n, t, c in ((w.n1, w.t1, BLUE), (w.n2, -w.t2, ORANGE)):
    xs = np.linspace(n.quantile(0.02), n.quantile(0.98), 50)
    ax.plot(xs, np.polyval(np.polyfit(n, t, 1), xs), color=c, lw=1)
ax.set_xlabel("own length (tokens)")
ax.set_ylabel("pull towards own slot")
ax.set_xlim(0, float(pd.concat([w.n1, w.n2]).quantile(0.98)))
ax.set_ylim(float(pd.concat([w.t1, -w.t2]).quantile(0.005)),
            float(pd.concat([w.t1, -w.t2]).quantile(0.995)))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          handletextpad=0.1, markerscale=3)
ax = axes[1]
ax.scatter(w.dlen, w.net, s=2, alpha=0.12, color=BLUE, lw=0)
xs = np.linspace(w.dlen.quantile(0.02), w.dlen.quantile(0.98), 50)
ax.plot(xs, np.polyval(np.polyfit(w.dlen, w.net, 1), xs), color="k", lw=1)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("length advantage, t1 − t2 (tokens)")
ax.set_ylabel("net pull towards answer 1")
ax.set_xlim(float(w.dlen.quantile(0.01)), float(w.dlen.quantile(0.99)))
ax.set_ylim(float(w.net.quantile(0.005)), float(w.net.quantile(0.995)))
fig.tight_layout()
fig.savefig(OUT / "fig_length_bias.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------- gate and slot prior
fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.9))
r2 = scores.groupby(["pair_uid", "order"]).heldout_r2.first()
axes[0].hist(r2, bins=40, color=BLUE, alpha=0.85)
axes[0].axvline(0, color="0.6", lw=0.8, ls="--")
axes[0].axvline(r2.median(), color="k", lw=1)
axes[0].set_xlabel("held-out R² per prompt")
axes[0].set_ylabel("prompts")
axes[1].hist(sp.f_blank, bins=40, color=BLUE, alpha=0.85)
axes[1].axvline(0, color="0.6", lw=0.8, ls="--")
axes[1].set_xlabel("F on a fully blanked prompt")
axes[1].set_ylabel("prompts")
fig.tight_layout()
fig.savefig(OUT / "fig_gate_and_prior.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ position
bgw = bg[bg.group.isin(["transcript_1", "transcript_2"])].pivot_table(
    index=["item", "pair_uid", "order"], columns="group", values="delta").reset_index()
bgw["net"] = bgw["transcript_1"] + bgw["transcript_2"]
mass = w.reset_index().merge(per[["pair_uid", "order"]].drop_duplicates(), on=["pair_uid", "order"])
mass_item = slots.groupby(["item", "pair_uid", "order"]).first().reset_index()[["item", "pair_uid", "order"]]
massnet = mass.merge(mass_item, on=["pair_uid", "order"])
massnet_p = massnet.groupby(["item", "pair_uid"])["net"].mean().reset_index()
bg_p = bgw.groupby(["item", "pair_uid"])["net"].mean().reset_index()
ci_mass = mask_summary.item_bootstrap(massnet_p, lambda f: f["net"].mean(), 1000)
ci_bg = mask_summary.item_bootstrap(bg_p, lambda f: f["net"].mean(), 1000)

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
for ax, ci, ylab in ((axes[0], ci_mass, "score mass, k = 8\n(towards first slot)"),
                     (axes[1], ci_bg, "measured ΔF, whole span\n(towards first slot)")):
    ax.errorbar([0], [ci["estimate"]], yerr=[[ci["estimate"] - ci["lo"]], [ci["hi"] - ci["estimate"]]],
                fmt="o", ms=4, color=BLUE, lw=1, capsize=0)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_ylabel(ylab)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
fig.tight_layout()
fig.savefig(OUT / "fig_position_scales.png", dpi=300)
plt.close(fig)

print("figures written:", sorted(p.name for p in OUT.glob("fig_*.png")))
print("check values: mass tilt", round(ci_mass["estimate"], 3), "bg tilt", round(ci_bg["estimate"], 3))
