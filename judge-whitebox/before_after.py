"""Before/after component matching: Hungarian-aligns SVD components across two decomps of the
same key by |u.u'|*|v.v'|, flags badly-aligned (low-sim) pairs, and diffs their judging
attribution. Sign of a (u,v) pair is ambiguous in SVD; sim uses abs() so it's invariant to that.
"""
import argparse
import pathlib

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import importance_report
import svd_components
import wbconfig
import wbfig


def _key_sim(entry_a, entry_b):
    u_a, _, v_a = entry_a
    u_b, _, v_b = entry_b
    return ((u_a.T @ u_b).abs() * (v_a.T @ v_b).abs()).numpy()


def match(decompA, decompB):
    """Per shared key, Hungarian-match components maximising sim = |u.u'|*|v.v'|. Returns
    (pairs, summary): pairs has one row per (comp_a, comp_b); summary has one row per key with
    the matched energy sitting below before_after.min_sim (config) as a fraction of the total."""
    assert decompA.entries.keys() == decompB.entries.keys(), "decomp key sets differ"
    min_sim = wbconfig.cfg()["before_after"]["min_sim"]

    pair_rows, summary_rows = [], []
    for key, entry_a in decompA.entries.items():
        entry_b = decompB.entries[key]
        sim = _key_sim(entry_a, entry_b)
        row_idx, col_idx = linear_sum_assignment(-sim)

        s_a, s_b = entry_a[1], entry_b[1]
        unmatched_mass, total_energy = 0.0, 0.0
        for i, j in zip(row_idx, col_idx):
            sv_a, sv_b = s_a[i].item(), s_b[j].item()
            energy = (sv_a ** 2 + sv_b ** 2) / 2
            low = bool(sim[i, j] < min_sim)
            total_energy += energy
            unmatched_mass += energy if low else 0.0
            pair_rows.append({"key": key, "comp_a": int(i), "comp_b": int(j),
                               "sim": float(sim[i, j]), "sv_a": sv_a, "sv_b": sv_b, "low_sim": low})

        meta = decompA.meta[key]
        summary_rows.append({"key": key, "layer": meta["layer"], "kind": meta["kind"],
                              "unmatched_mass": unmatched_mass, "total_energy": total_energy,
                              "frac_unmatched": unmatched_mass / total_energy if total_energy else 0.0})

    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def importance_diff(matches, screenA, screenB):
    """matches: pairs table from match(). screenA/screenB: ScreenResult save dicts (attr_abs,
    n_prompts). Per-prompt-normalised attr_abs on each side, delta = b - a, sorted by |delta|.
    Low-sim pairs are kept (flagged), not dropped. screenA/screenB's model/adapter identity is
    recorded on every row for provenance, not asserted equal — A and B are meant to be different
    models/checkpoints here (that's the point of a before/after diff), unlike merge_table's
    judging vs control which must be the same run."""
    model_a, adapter_a = screenA.get("model"), screenA.get("adapter")
    model_b, adapter_b = screenB.get("model"), screenB.get("adapter")
    rows = []
    for r in matches.itertuples():
        a = screenA["attr_abs"][r.key][r.comp_a].item() / screenA["n_prompts"]
        b = screenB["attr_abs"][r.key][r.comp_b].item() / screenB["n_prompts"]
        rows.append({"key": r.key, "comp_a": r.comp_a, "comp_b": r.comp_b, "sim": r.sim,
                     "low_sim": r.low_sim, "attr_abs_a": a, "attr_abs_b": b, "delta": b - a,
                     "model_a": model_a, "adapter_a": adapter_a,
                     "model_b": model_b, "adapter_b": adapter_b})
    df = pd.DataFrame(rows)
    return df.reindex(df["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def fig_scatter(diff, tag, eps):
    """Matched-importance scatter, attr_abs A vs B, log-log, y = x reference, low-sim pairs a
    distinct colour, top-10 |delta| pairs labelled (diff arrives pre-sorted by |delta|)."""
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    x, y = diff["attr_abs_a"].clip(lower=eps), diff["attr_abs_b"].clip(lower=eps)
    low = diff["low_sim"]
    ax.scatter(x[~low], y[~low], s=4, alpha=0.35, color=wbfig.MUTED, linewidths=0, label="matched")
    ax.scatter(x[low], y[low], s=8, color=wbfig.PALETTE[1], linewidths=0, label="low-sim match")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, color=wbfig.GRID, linewidth=1, zorder=0, label="y = x")

    for r in diff.head(10).itertuples():
        layer, kind, head = importance_report._parse_key(r.key)
        ax.annotate(importance_report._label(layer, kind, head, r.comp_a),
                    (max(r.attr_abs_a, eps), max(r.attr_abs_b, eps)), fontsize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("attr_abs / prompt (A)")
    ax.set_ylabel("attr_abs / prompt (B)")
    ax.legend(frameon=False)
    wbfig.save(fig, f"before_after_{tag}_scatter")


def fig_unmatched(summary, tag):
    """Fraction of matched energy below min_sim, by layer, grouped bars per matrix kind."""
    pivot = summary.pivot_table(index="layer", columns="kind", values="frac_unmatched", aggfunc="mean")
    kinds = [k for k in importance_report.KINDS if k in pivot.columns]
    fig = wbfig.new_fig(wbfig.WIDE, 3.0)
    ax = fig.add_subplot(111)
    width = 0.8 / len(kinds)
    x = pivot.index.to_numpy(dtype=float)
    for i, kind in enumerate(kinds):
        ax.bar(x + (i - (len(kinds) - 1) / 2) * width, pivot[kind], width=width,
               color=wbfig.PALETTE[i % len(wbfig.PALETTE)], label=kind)
    ax.set_xlabel("layer")
    ax.set_ylabel("fraction unmatched mass")
    ax.legend(frameon=False, ncol=len(kinds))
    wbfig.save(fig, f"before_after_{tag}_unmatched")


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp-a", required=True)
    parser.add_argument("--decomp-b", required=True)
    parser.add_argument("--screen-a", required=True)
    parser.add_argument("--screen-b", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    decompA, decompB = svd_components.load(args.decomp_a), svd_components.load(args.decomp_b)
    screenA, screenB = torch.load(args.screen_a), torch.load(args.screen_b)

    pairs, summary = match(decompA, decompB)
    diff = importance_diff(pairs, screenA, screenB)
    merged = diff.merge(pairs[["key", "comp_a", "comp_b", "sv_a", "sv_b"]],
                         on=["key", "comp_a", "comp_b"], how="left")

    out_dir = pathlib.Path(c["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"before_after_{args.tag}.csv", index=False)
    summary.to_csv(out_dir / f"before_after_{args.tag}_summary.csv", index=False)

    fig_scatter(diff, args.tag, c["screen"]["act_eps"])
    fig_unmatched(summary, args.tag)

    print(f"n_pairs={len(pairs)} n_low_sim={int(pairs['low_sim'].sum())} -> {out_dir}")


if __name__ == "__main__":
    main()
