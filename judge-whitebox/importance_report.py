"""Importance report: merge judging/control screens into a per-component table + figures.

act_judge/act_ctrl are per-token means (activity sum / n_tokens of that set). attr_signed/attr_abs
are per-prompt means of the judging set's attribution (attribution is w.r.t. the judging loss only —
the control run's attribution reflects a different loss and isn't comparable). Candidates are
components with act_ratio = act_judge / (act_ctrl + eps) >= screen.act_ratio_min, ranked by attr_abs.
"""
import argparse
import pathlib
import re

import pandas as pd
import torch

import wbconfig
import wbfig

KINDS = ("q", "k", "v", "o", "gate", "up", "down")
KEY_RE = re.compile(r"^L(\d+)\.(" + "|".join(KINDS) + r")(?:\.h(\d+))?$")


def _parse_key(key):
    m = KEY_RE.match(key)
    assert m, f"unrecognised component key: {key}"
    layer, kind, head = m.groups()
    return int(layer), kind, (int(head) if head is not None else None)


def merge_table(judging, control, eps, judging_path=None, control_path=None):
    """judging/control: ScreenResult dicts as saved by screen.py (same keys, same decomp).
    judging_path/control_path: only used to name a legacy-format input in the warning below.

    model/adapter identity is asserted only when BOTH sides carry those fields — screens saved
    before this check existed don't have them, and must keep working (as a loud warning, not a
    crash) rather than get treated as a mismatch.
    """
    assert judging["activity"].keys() == control["activity"].keys(), "judging/control decomp mismatch"
    j_model, c_model = judging.get("model"), control.get("model")
    if j_model is None or c_model is None:
        legacy = [p for p, d in ((judging_path, judging), (control_path, control)) if d.get("model") is None]
        print(f"WARNING: legacy-format screen result (no model/adapter fields), skipping "
              f"judging/control identity check: {[p or '<unknown>' for p in legacy]}")
    else:
        assert j_model == control["model"] and judging.get("adapter") == control.get("adapter"), (
            f"judging/control model+adapter mismatch: judging model={j_model!r} "
            f"adapter={judging.get('adapter')!r} ({judging_path or '<unknown>'}) vs control "
            f"model={control['model']!r} adapter={control.get('adapter')!r} ({control_path or '<unknown>'})")
    n_tok_j, n_tok_c, n_prompt_j = judging["n_tokens"], control["n_tokens"], judging["n_prompts"]

    rows = []
    for key, act_j in judging["activity"].items():
        act_c = control["activity"][key]
        assert act_j.shape == act_c.shape, f"component count mismatch at {key}"
        asig_j, aabs_j = judging["attr_signed"][key], judging["attr_abs"][key]
        layer, kind, head = _parse_key(key)
        for i in range(act_j.shape[0]):
            aj, ac = act_j[i].item() / n_tok_j, act_c[i].item() / n_tok_c
            rows.append({
                "key": key, "comp": i, "layer": layer, "kind": kind, "head": head,
                "act_judge": aj, "act_ctrl": ac, "act_ratio": aj / (ac + eps),
                "attr_signed": asig_j[i].item() / n_prompt_j,
                "attr_abs": aabs_j[i].item() / n_prompt_j,
            })
    return pd.DataFrame(rows)


def top_components(csv_path, n):
    """Top n candidates by attr_abs, for downstream ablation."""
    df = pd.read_csv(csv_path)
    top = df[df["is_candidate"]].sort_values("attr_abs", ascending=False).head(n)
    return [(row.key, int(row.comp)) for row in top.itertuples()]


def _label(layer, kind, head, comp):
    if pd.isna(head):
        return f"L{layer}.{kind}#{comp}"
    return f"L{layer}.{kind}.h{int(head)}#{comp}"


def fig_depth(df, tag):
    """Attribution-vs-depth: one line per matrix kind, summed |attribution| per layer."""
    depth = df.groupby(["layer", "kind"])["attr_abs"].sum().unstack("kind")
    fig = wbfig.new_fig(wbfig.WIDE, 3.0)
    ax = fig.add_subplot(111)
    for i, kind in enumerate(KINDS):
        if kind in depth.columns:
            ax.plot(depth.index, depth[kind], label=kind, color=wbfig.PALETTE[i % len(wbfig.PALETTE)])
    ax.set_xlabel("layer")
    ax.set_ylabel("sum |attribution| / prompt")
    ax.legend(frameon=False, ncol=len(KINDS))
    wbfig.save(fig, f"importance_{tag}_depth")


def fig_scatter(df, tag, eps):
    """Judging vs control activity per component, candidates highlighted, y=x reference."""
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    x, y = df["act_ctrl"].clip(lower=eps), df["act_judge"].clip(lower=eps)
    cand = df["is_candidate"]
    ax.scatter(x[~cand], y[~cand], s=4, alpha=0.35, color=wbfig.MUTED, linewidths=0, label="component")
    ax.scatter(x[cand], y[cand], s=8, color=wbfig.PALETTE[1], linewidths=0, label="candidate")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, color=wbfig.GRID, linewidth=1, zorder=0, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("control activity / token")
    ax.set_ylabel("judging activity / token")
    ax.legend(frameon=False)
    wbfig.save(fig, f"importance_{tag}_scatter")


def fig_top30(df, tag, top_n):
    """Bar chart of the top-n candidates by attr_abs, labelled, signed attribution, zero line."""
    top = df[df["is_candidate"]].sort_values("attr_abs", ascending=False).head(top_n)
    labels = [_label(r.layer, r.kind, r.head, r.comp) for r in top.itertuples()]
    fig = wbfig.new_fig(wbfig.COL, max(2.0, 0.18 * len(top) + 0.4))
    ax = fig.add_subplot(111)
    y_pos = list(range(len(top)))
    ax.barh(y_pos, top["attr_signed"], color=wbfig.PALETTE[0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color=wbfig.INK, linewidth=0.8)
    ax.set_xlabel("signed attribution / prompt")
    wbfig.save(fig, f"importance_{tag}_top30")


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--judging", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    judging = torch.load(args.judging)
    control = torch.load(args.control)
    eps, ratio_min = c["screen"]["act_eps"], c["screen"]["act_ratio_min"]

    df = merge_table(judging, control, eps, judging_path=args.judging, control_path=args.control)
    df["is_candidate"] = df["act_ratio"] >= ratio_min
    df = df.sort_values("attr_abs", ascending=False).reset_index(drop=True)

    out_path = pathlib.Path(c["paths"]["out"]) / f"importance_{args.tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    fig_depth(df, args.tag)
    fig_scatter(df, args.tag, eps)
    fig_top30(df, args.tag, args.top_n)

    print(f"n_components={len(df)} n_candidates={int(df['is_candidate'].sum())} -> {out_path}")


if __name__ == "__main__":
    main()
