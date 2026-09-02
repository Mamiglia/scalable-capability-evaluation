"""Remade whitebox-dissection figures. Every axis is in verdict log-odds:
F = log P(judge answers "1") / P(judge answers "2"). Positive = towards transcript 1."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge-whitebox")
import mask_regression as mr
import mask_summary

S = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/tmp/whitebox")
OUT = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/whitebox_dissection/figures/llama3.2-3B")

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a78d6", "#eb6834"

scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
bg = pd.concat([pd.read_parquet(p) for p in sorted((S / "blankgroups_llama").rglob("part_*.parquet"))],
               ignore_index=True)

# ---------------------------------------------------------- 1. verdict bias
f = bg.groupby(["item", "pair_uid", "order"])["gap"].first().reset_index()
w = f.pivot_table(index=["item", "pair_uid"], columns="order", values="gap").dropna()
pair_sum = w[0] + w[1]

fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
axes[0].hist(f.gap, bins=45, color=BLUE, alpha=0.85)
axes[0].axvline(0, color="0.5", lw=0.8, ls="--")
axes[0].axvline(f.gap.mean(), color="k", lw=1)
axes[0].set_xlabel("verdict log-odds, one prompt\n(> 0 = judge picks transcript 1)")
axes[0].set_ylabel("prompts")
axes[1].hist(pair_sum, bins=45, color=BLUE, alpha=0.85)
axes[1].axvline(0, color="0.5", lw=0.8, ls="--")
axes[1].axvline(pair_sum.mean(), color="k", lw=1)
axes[1].set_xlabel("same pair, both orders summed\n(0 = no slot preference)")
axes[1].set_ylabel("pairs")
fig.tight_layout()
fig.savefig(OUT / "fig1_verdict_bias.png", dpi=300); fig.savefig(OUT / "fig1_verdict_bias.svg")
plt.close(fig)

# ------------------------------------------------- 2. per-token effect hists
slots = scores[scores.group.isin(["transcript_1", "transcript_2"])].copy()
slots["towards_own"] = np.where(slots.group == "transcript_1",
                                slots.beta_per_token, -slots.beta_per_token)
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.8), sharex=True)
for ax, grp, name, color in ((axes[0], "transcript_1", "transcript 1 sentences", BLUE),
                             (axes[1], "transcript_2", "transcript 2 sentences", ORANGE)):
    vals = slots.loc[slots.group == grp, "towards_own"]
    ax.hist(vals.clip(-2.5, 2.5), bins=100, color=color, alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.axvline(vals.mean(), color="k", lw=1)
    ax.set_ylabel(name.split()[0] + " " + name.split()[1] + "\nsentences", fontsize=7)
    print(name, "mean per-token pull towards own slot: %.4f, share>0 %.3f"
          % (vals.mean(), (vals > 0).mean()))
axes[1].set_xlabel("per-token pull towards its own transcript\n(log-odds; log count scale; tails clipped at ±2.5)")
fig.tight_layout()
fig.savefig(OUT / "fig2_token_effects.png", dpi=300); fig.savefig(OUT / "fig2_token_effects.svg")
plt.close(fig)

# ---------------------------------------------------------- 3. length bias
per = slots.groupby(["item", "pair_uid", "order", "group"]).agg(
    total=("score", "sum"), toks=("n_tokens", "sum")).reset_index()
per["towards_own"] = np.where(per.group == "transcript_1", per.total, -per.total)
t1 = per[per.group == "transcript_1"].set_index(["item", "pair_uid", "order"])
t2 = per[per.group == "transcript_2"].set_index(["item", "pair_uid", "order"])
pw = pd.DataFrame({"t1": t1.total, "t2": t2.total, "n1": t1.toks, "n2": t2.toks}).dropna().reset_index()
pw["dlen"], pw["net"] = pw.n1 - pw.n2, pw.t1 + pw.t2


def binned_ci(frame, xcol, ycol, bins=8):
    frame = frame.copy()
    frame["bin"] = pd.qcut(frame[xcol], bins, duplicates="drop")
    rows = []
    for b, g in frame.groupby("bin", observed=True):
        ci = mask_summary.item_bootstrap(g[["item", ycol]].rename(columns={ycol: "v"}),
                                         lambda x: x["v"].mean(), 300)
        rows.append({"x": g[xcol].mean(), **ci})
    return pd.DataFrame(rows)


fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
for grp, name, color in (("transcript_1", "transcript 1", BLUE),
                         ("transcript_2", "transcript 2", ORANGE)):
    g = per[per.group == grp]
    b = binned_ci(g, "toks", "towards_own")
    axes[0].plot(b.x, b.estimate, color=color, lw=1, label=name)
    axes[0].fill_between(b.x, b.lo, b.hi, color=color, alpha=0.18, lw=0)
axes[0].axhline(0, color="0.5", lw=0.8, ls="--")
axes[0].set_xlabel("transcript length (tokens)")
axes[0].set_ylabel("total pull towards\nits own slot (log-odds)")
axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
b = binned_ci(pw, "dlen", "net")
axes[1].plot(b.x, b.estimate, color="k", lw=1)
axes[1].fill_between(b.x, b.lo, b.hi, color="0.4", alpha=0.18, lw=0)
axes[1].axhline(0, color="0.5", lw=0.8, ls="--")
axes[1].axvline(0, color="0.5", lw=0.8, ls="--")
axes[1].set_xlabel("length advantage of transcript 1\nover transcript 2 (tokens)")
axes[1].set_ylabel("net pull towards\ntranscript 1 (log-odds)")
fig.tight_layout()
fig.savefig(OUT / "fig3_length_bias.png", dpi=300); fig.savefig(OUT / "fig3_length_bias.svg")
plt.close(fig)

# ------------------------------------------------------- 4. whole-span deletion
SPANS = [("transcript_1", "delete transcript 1"), ("transcript_2", "delete transcript 2"),
         ("instr_format", "delete instructions"), ("slot_labels", "delete slot labels"),
         ("question", "delete question"), ("answer_readout", "delete answer cue")]
cis = {}
for grp, _ in SPANS:
    g = bg[bg.group == grp].groupby(["item", "pair_uid", "order"]).delta.first().reset_index()
    # delta = F(full) - F(span blanked): flip sign to show the shift deletion CAUSES
    g["shift"] = -g.delta
    cis[grp] = mask_summary.item_bootstrap(g[["item", "shift"]], lambda x: x["shift"].mean(), 500)
fig, ax = plt.subplots(figsize=(3.6, 2.1))
y = np.arange(len(SPANS))[::-1]
est = [cis[g]["estimate"] for g, _ in SPANS]
lo = [cis[g]["estimate"] - cis[g]["lo"] for g, _ in SPANS]
hi = [cis[g]["hi"] - cis[g]["estimate"] for g, _ in SPANS]
ax.errorbar(est, y, xerr=[lo, hi], fmt="o", ms=3.5, color=BLUE, lw=1, capsize=0)
ax.axvline(0, color="0.5", lw=0.8, ls="--")
ax.set_yticks(y, [n for _, n in SPANS])
ax.set_xlabel("how the verdict shifts (log-odds)\n(+ = towards transcript 1)")
fig.tight_layout()
fig.savefig(OUT / "fig4_whole_span_deletion.png", dpi=300); fig.savefig(OUT / "fig4_whole_span_deletion.svg")
plt.close(fig)

# --------------------------------------------------------- 5. validation
# valid1106 only: the sentence fits being validated used the 512-draw campaign,
# and maskcamp_llama now also holds the 3584-draw extension plus pooled hardlinks
masks = mr.read_mask_parts(S / "maskcamp_llama" / "valid1106")
rng = np.random.default_rng(3)
sample_keys = list(masks.groupby(["pair_uid", "order"]).groups)
pick = [sample_keys[i] for i in rng.choice(len(sample_keys), 150, replace=False)]
meta_units = scores.set_index(["pair_uid", "order"])
pred_all, obs_all = [], []
for uid, order in pick:
    g = masks[(masks.pair_uid == uid) & (masks.order == order)]
    units = meta_units.loc[(uid, order)].sort_values("unit")
    unit_ids = np.repeat(units.unit.to_numpy(), units.n_tokens.to_numpy())
    X = mr.design_matrix(g.masked_positions.tolist(), unit_ids)
    y = g.delta.to_numpy(float)
    order_ix = np.random.default_rng(0).permutation(len(y))
    test = order_ix[0::4]
    train = np.setdiff1d(order_ix, test)
    beta = mr._ridge_solve(X[train], y[train], float(units.ridge.iloc[0]))
    pred_all.append(X[test] @ beta)
    obs_all.append(y[test])
pred, obs = np.concatenate(pred_all), np.concatenate(obs_all)
fig, ax = plt.subplots(figsize=(2.6, 2.4))
ax.scatter(pred, obs, s=2, alpha=0.1, color=BLUE, lw=0)
lims = np.percentile(np.concatenate([pred, obs]), [0.5, 99.5])
ax.plot(lims, lims, color="k", lw=1)
ax.set_xlim(*lims)
ax.set_ylim(*lims)
ax.set_xlabel("predicted verdict shift\n(8 random tokens blanked)")
ax.set_ylabel("measured verdict shift")
r = np.corrcoef(pred, obs)[0, 1]
print("validation scatter r = %.3f over %d held-out draws" % (r, len(pred)))
fig.tight_layout()
fig.savefig(OUT / "fig5_validation.png", dpi=300); fig.savefig(OUT / "fig5_validation.svg")
plt.close(fig)

# ------------------------------------------------- 6. cumulative curve, remade
sp = pd.concat([pd.read_parquet(p) for p in sorted((S / "slotpref_full_llama").rglob("part_*.parquet"))])
sp["f_blank"] = sp["gap"] - sp["delta"]
blank = sp.set_index(["pair_uid", "order"])["f_blank"]
grid = np.linspace(0, 1, 201)
curves, group_starts = [], {g: [] for g in scores.group.unique()}
for (uid, order), g in scores.groupby(["pair_uid", "order"]):
    g = g.sort_values("unit")
    ends = g.n_tokens.cumsum() / g.n_tokens.sum()
    start = blank.get((uid, order), 0.0126)
    cum = start + g.score.cumsum()
    curves.append(np.interp(grid, np.concatenate([[0], ends]), np.concatenate([[start], cum])))
    rel = (g.n_tokens.cumsum() - g.n_tokens) / g.n_tokens.sum()
    for grp in group_starts:
        sel = g.group.values == grp
        if sel.any():
            group_starts[grp].append(rel.values[sel][0])
curves = np.asarray(curves)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
starts = sorted(((np.median(v), k) for k, v in group_starts.items() if v))
bounds = [s for s, _ in starts] + [1.0]
labels = {"instr_format": "instructions", "slot_labels": "", "question": "",
          "transcript_1": "transcript 1", "transcript_2": "transcript 2", "answer_readout": ""}
lim = np.percentile(np.abs(curves), 98)
for i, (s, name) in enumerate(starts):
    if i % 2 == 1:
        ax.axvspan(bounds[i], bounds[i + 1], color="0.55", alpha=0.12, lw=0)
    if labels[name] and bounds[i + 1] - bounds[i] > 0.12:
        ax.text((bounds[i] + bounds[i + 1]) / 2, lim * 1.02, labels[name],
                ha="center", va="bottom", fontsize=6, color="0.45")
sub = curves[np.random.default_rng(0).choice(len(curves), 300, replace=False)]
for c in sub:
    ax.plot(grid, c, color=BLUE, lw=0.3, alpha=0.05)
ax.plot(grid, np.median(curves, axis=0), color=BLUE, lw=1, label="running sum of sentence effects")
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axhline(f.gap.mean(), color="k", lw=1, ls=":", label="actual mean verdict")
ax.set_xlabel("relative position in prompt")
ax.set_ylabel("log-odds towards transcript 1")
ax.set_xlim(0, 1)
ax.set_ylim(-lim, lim)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=1)
fig.tight_layout()
fig.savefig(OUT / "fig6_cumulative.png", dpi=300); fig.savefig(OUT / "fig6_cumulative.svg")
plt.close(fig)

print("written:", sorted(p.name for p in OUT.glob("fig*.png")))
