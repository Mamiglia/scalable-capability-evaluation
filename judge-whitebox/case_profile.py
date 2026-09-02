"""Case profiling: NMF factors of per-prompt component activity, related back to prompt context
(slot order, item, author-model family) and to component kind (which matrix a factor draws from).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF

import importance_report
import wbconfig
import wbfig
import wbpairs


def family(model_name):
    """Author-model family: '-'-joined tokens up to (not including) the first token that starts
    with a digit, lowercased. "open-mixtral-8x22b" -> "open-mixtral", "DeepSeek-R1-0528" ->
    "deepseek-r1", "claude-3-5-haiku-20241022" -> "claude". Falls back to the lowercased whole
    name if the first token itself starts with a digit."""
    kept = []
    for token in model_name.split("-"):
        if token[:1].isdigit():
            break
        kept.append(token)
    return "-".join(kept).lower() if kept else model_name.lower()


def _grouped_mean_long(W, group_values, variable):
    """Mean of each W column within each group, tidy: variable, group, factor, value, n."""
    n_factors = W.shape[1]
    dfw = pd.DataFrame(W, columns=range(n_factors))
    dfw["_group"] = np.asarray(group_values)
    long = dfw.groupby("_group").mean().reset_index().melt(
        id_vars="_group", var_name="factor", value_name="value")
    long["n"] = long["_group"].map(dfw.groupby("_group").size())
    long = long.rename(columns={"_group": "group"})
    long.insert(0, "variable", variable)
    return long[["variable", "group", "factor", "value", "n"]]


def _context_report(W, prompt_meta):
    """Mean factor weight grouped by order / item / model_1 family / model_2 family."""
    fam1 = prompt_meta["model_1"].map(family)
    fam2 = prompt_meta["model_2"].map(family)
    return pd.concat([
        _grouped_mean_long(W, prompt_meta["order"].to_numpy(), "order"),
        _grouped_mean_long(W, prompt_meta["item"].to_numpy(), "item"),
        _grouped_mean_long(W, fam1.to_numpy(), "model_1_family"),
        _grouped_mean_long(W, fam2.to_numpy(), "model_2_family"),
    ], ignore_index=True)


def _kind_composition(H, top_k_components):
    """Per factor, fraction of H's mass held by each component kind (columns sum to 1 per factor)."""
    kinds = [importance_report._parse_key(key)[1] for key, _ in top_k_components]
    dfh = pd.DataFrame(H.T)
    dfh["_group"] = kinds
    mass = dfh.groupby("_group").sum()
    frac = mass.div(mass.sum(axis=0), axis=1)
    long = frac.reset_index().melt(id_vars="_group", var_name="factor", value_name="value")
    long["n"] = long["_group"].map(dfh.groupby("_group").size())
    long = long.rename(columns={"_group": "group"})
    long.insert(0, "variable", "kind")
    return long[["variable", "group", "factor", "value", "n"]]


def profile(per_prompt_activity, top_k_components, prompt_meta, n_factors, seed, max_iter):
    """NMF on [n_prompts, k] activity (nonneg by construction, clipped as a guard).
    Returns W [n_prompts, f], H [f, k], report (context group means + kind composition, tidy)."""
    X = np.clip(np.asarray(per_prompt_activity, dtype=float), 0, None)
    nmf = NMF(n_components=n_factors, init="nndsvda", random_state=seed, max_iter=max_iter)
    W = nmf.fit_transform(X)
    H = nmf.components_
    report = pd.concat([_context_report(W, prompt_meta), _kind_composition(H, top_k_components)],
                        ignore_index=True)
    return W, H, report


def _load_prompt_meta(screen_dict, c):
    """Reconstruct the exact pairs-df rows the judging screen ran over, aligned with
    per_prompt_activity: pairs_df filtered by the screen's own split/limit, then positionally
    selected by kept_row_idx (see wbpairs.select_kept_rows). set_name is checked here too,
    before touching wbpairs.load_pairs (which needs a real config), so a control-set screen
    fails fast regardless of what `c` is."""
    assert screen_dict.get("set_name") == "judging", (
        f"case_profile requires a judging-set screen result (rows align 1:1 with pairs_df); "
        f"got set_name={screen_dict.get('set_name')!r}. Re-run screen.py with --set judging.")
    split, limit = screen_dict.get("split"), screen_dict.get("limit")
    if split is None:
        print("WARNING: screen result has no recorded split — reconstructing pairs with no split "
              "filter, which is wrong unless the screen itself was run without one.")
    pairs_df = wbpairs.load_pairs(c, split=split, limit=limit)
    return wbpairs.select_kept_rows(pairs_df, screen_dict)


def _h_table(H, top_k_components):
    cols = [f"{key}#{comp}" for key, comp in top_k_components]
    return pd.DataFrame(H, columns=cols)


def _top_bins(report, variable, max_bins):
    """Groups of `variable`, most-frequent first (by planted `n`), capped at max_bins."""
    sub = report[report["variable"] == variable][["group", "n"]].drop_duplicates()
    return sub.sort_values("n", ascending=False).head(max_bins)["group"].tolist()


def fig_context(report, tag, max_context_bins):
    """Heatmap: factors (y) x context bins (x) = order (both), top item bins, top family bins."""
    bins = [("order", g) for g in sorted(report[report["variable"] == "order"]["group"].unique())]
    for variable in ("item", "model_1_family", "model_2_family"):
        bins += [(variable, g) for g in _top_bins(report, variable, max_context_bins)]
    factors = sorted(report["factor"].unique())

    by_key = {(r.variable, r.group, r.factor): r.value for r in report.itertuples()}
    mat = np.array([[by_key.get((v, g, f), np.nan) for (v, g) in bins] for f in factors])

    fig = wbfig.new_fig(max(wbfig.WIDE, 0.35 * len(bins)), max(2.0, 0.3 * len(factors) + 0.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, cmap=wbfig.HEAT, aspect="auto")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{v}:{g}" for v, g in bins], rotation=90, fontsize=6)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors)
    ax.set_xlabel(f"context bin (item/family capped to top {max_context_bins} most frequent)")
    ax.set_ylabel("factor")
    fig.colorbar(im, ax=ax, label="mean factor weight")
    wbfig.save(fig, f"case_{tag}_context")


def fig_kinds(report, tag):
    """Stacked bars: fraction of each factor's component-kind mass, per matrix kind."""
    kinds_df = report[report["variable"] == "kind"]
    pivot = kinds_df.pivot_table(index="factor", columns="group", values="value").fillna(0.0)
    kinds = [k for k in importance_report.KINDS if k in pivot.columns]

    fig = wbfig.new_fig(wbfig.COL, 3.0)
    ax = fig.add_subplot(111)
    bottom = np.zeros(len(pivot))
    for i, kind in enumerate(kinds):
        ax.bar(pivot.index, pivot[kind], bottom=bottom, color=wbfig.PALETTE[i % len(wbfig.PALETTE)],
               label=kind)
        bottom += pivot[kind].to_numpy()
    ax.set_xticks(list(pivot.index))
    ax.set_xlabel("factor")
    ax.set_ylabel("fraction of component-kind mass")
    ax.legend(frameon=False, ncol=len(kinds))
    wbfig.save(fig, f"case_{tag}_kinds")


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--factors", type=int, default=c["case_profile"]["n_factors"])
    args = parser.parse_args()

    screen_dict = torch.load(args.screen)
    prompt_meta = _load_prompt_meta(screen_dict, c)

    W, H, report = profile(screen_dict["per_prompt_activity"].numpy(), screen_dict["top_k_components"],
                            prompt_meta, args.factors, c["case_profile"]["nmf_seed"],
                            c["case_profile"]["nmf_max_iter"])

    out_dir = pathlib.Path(c["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / f"case_profile_{args.tag}.csv", index=False)
    _h_table(H, screen_dict["top_k_components"]).to_csv(
        out_dir / f"case_profile_{args.tag}_H.csv", index_label="factor")

    fig_context(report, args.tag, c["case_profile"]["max_context_bins"])
    fig_kinds(report, args.tag)

    print(f"n_factors={args.factors} n_prompts={W.shape[0]} -> {out_dir}")


if __name__ == "__main__":
    main()
