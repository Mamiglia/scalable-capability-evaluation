"""Generate the IRT validation figures from the stored tables."""

from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules of this analysis
from data import pick_epoch  # noqa: E402
from plots import (  # noqa: E402COLOR, LABEL, cat_scatter, cat_scatter_joint, curve_plot, density_plot,
                       heatmap, joint_pair, scatter, scree_plot, strip, sweep_gap_plot, wright_map)

# categorical palette (dataviz default, light surface); "Other" stays muted grey
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER = "#a8a79d"


def tau(x, y):
    """Kendall's tau-b (tie-corrected)."""
    return pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).corr(method="kendall").iloc[0, 1]


def org_family(s):
    """Epoch's Organization string to one family label."""
    if not isinstance(s, str):
        return "Other"
    s = s.split(",")[0].strip()
    return "Google" if s.startswith("Google") else s


def org_colors(df):
    """Top families by model count get a palette slot, the rest fold into Other."""
    counts = df["family"].value_counts()
    top = list(counts.index[:len(PALETTE)])
    order = top + (["Other"] if len(counts) > len(top) else [])
    colors = dict(zip(top, PALETTE))
    colors["Other"] = OTHER
    df["family"] = df["family"].where(df["family"].isin(top), "Other")
    return order, colors


def theta_figures(root: Path, cap: pd.DataFrame, dia: pd.DataFrame):
    """theta vs published accuracy and vs release date, coloured by model family."""
    fig = root / "figures"
    dia = dia.assign(family=dia["Organization"].map(org_family))
    dt = pd.to_datetime(dia["Release date"], errors="coerce")
    dia = dia.assign(year=dt.dt.year + (dt.dt.dayofyear - 1) / 365.25)
    c = cap.merge(dia[["Model version", "mean_score", "year", "family"]],
                  left_on="model", right_on="Model version")
    order, colors = org_colors(c)

    cat_scatter(c, "mean_score", "theta_1pl", "family", order, colors,
                "published accuracy (Epoch)", "capability θ (1PL)",
                str(fig / "theta_vs_accuracy"),
                note=f"Kendall τ {tau(c.mean_score, c.theta_1pl):+.3f}  ·  n={len(c)}/{len(cap)}")
    cy = c.dropna(subset=["year"])
    cat_scatter(cy, "year", "theta_1pl", "family", order, colors,
                "release date", "capability θ (1PL)",
                str(fig / "theta_vs_year"),
                note=f"Kendall τ {tau(cy.year, cy.theta_1pl):+.3f}  ·  n={len(cy)}/{len(cap)}")


DOMAIN_ORDER = ["Physics", "Chemistry", "Biology"]
DOMAIN_COLOR = {"Physics": "#2a78d6", "Chemistry": "#eb6834", "Biology": "#1baf7a"}


def item_figure(root: Path, dif: pd.DataFrame, data: Path):
    """Item difficulty b vs empirical item %-correct, coloured by subject domain."""
    resp = pick_epoch(pd.read_parquet(data / "responses.parquet"))
    rate = resp.groupby(["bench", "item"])["correct"].mean().rename("passrate").reset_index()
    d = dif.merge(rate, on=["bench", "item"])
    note = None
    out = str(root / "figures" / "b_vs_passrate")
    dom_path = data / "item_domains.csv"
    if dom_path.exists():
        d = d.merge(pd.read_csv(dom_path)[["item", "domain"]], on="item")
        note = f"Kendall τ {tau(d.passrate, d.b_1pl):+.3f}  ·  n={len(d)}"
        cat_scatter_joint(d, "passrate", "b_1pl", "domain", DOMAIN_ORDER, DOMAIN_COLOR,
                          "item %-correct", "difficulty b (1PL)", out, note=note,
                          note_xy=(0.03, 0.06), bw_scale=0.6)
    else:
        scatter(d, "passrate", "b_1pl", "item %-correct", "difficulty b (1PL)", out,
                color=COLOR["item_base"], note_xy=(0.03, 0.06),
                note=f"Kendall τ {tau(d.passrate, d.b_1pl):+.3f}  ·  n={len(d)}")


def agreement_figure(root: Path, cap: pd.DataFrame, dia: pd.DataFrame):
    """1PL vs 2PL capability agreement, coloured by model family."""
    fam = dia.assign(family=dia["Organization"].map(org_family))[["Model version", "family"]]
    c = cap.merge(fam, left_on="model", right_on="Model version", how="left")
    c["family"] = c["family"].fillna("Other")
    order, colors = org_colors(c)
    cat_scatter(c, "theta_1pl", "theta_2pl", "family", order, colors,
                "θ (1PL)", "θ (2PL)", str(root / "figures" / "theta_1pl_vs_2pl"), identity=True,
                note=f"Kendall τ {tau(c.theta_1pl, c.theta_2pl):+.3f}  ·  n={len(c)}")


def wright_figure(root: Path, cap: pd.DataFrame, dif: pd.DataFrame):
    """Model ability and item difficulty on one latent scale."""
    wright_map(cap["theta_1pl"].to_numpy(), dif["b_1pl"].to_numpy(),
               str(root / "figures" / "wright_map"))


def item_pca(data: Path):
    """PCA on the item response space (items as variables). Returns loadings + var."""
    pe = pick_epoch(pd.read_parquet(data / "responses.parquet"))
    R = pe.pivot_table(index="model", columns="item", values="correct")
    items = list(R.columns)
    X = np.array(R.to_numpy(float), copy=True)
    cm = np.nanmean(X, 0)
    ix = np.where(np.isnan(X))
    X[ix] = np.take(cm, ix[1])
    _, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    dom = pd.read_csv(data / "item_domains.csv").set_index("item")["domain"]
    load = pd.DataFrame({"item": items, "domain": [dom[i] for i in items], "pc1": Vt[0], "pc2": Vt[1]})
    if load["pc1"].mean() < 0:                                 # orient PC1 positive (g-loading)
        load["pc1"] *= -1
    if load.loc[load.domain == "Physics", "pc2"].mean() < 0:   # orient PC2 so Physics is +
        load["pc2"] *= -1
    return load, var


def lda2(X, y):
    """Fisher LDA: project X (n x 2) onto the 2 axes that best separate classes y."""
    cls = np.unique(y)
    mu = X.mean(0)
    Sw = sum((X[y == c] - X[y == c].mean(0)).T @ (X[y == c] - X[y == c].mean(0)) for c in cls)
    Sb = sum(len(X[y == c]) * np.outer(X[y == c].mean(0) - mu, X[y == c].mean(0) - mu) for c in cls)
    ev, W = np.linalg.eig(np.linalg.solve(Sw, Sb))
    return X @ W[:, np.argsort(ev.real)[::-1]].real


def mirt_rot(root: Path, data: Path):
    """MIRT-K2 item loadings projected onto the 2 LDA axes that separate subjects."""
    dom = pd.read_csv(data / "item_domains.csv")[["item", "domain"]]
    mi = pd.read_csv(root / "gpqa_mirt2_items.csv").merge(dom, on="item")
    Z = lda2(mi[["a1", "a2"]].to_numpy(), mi["domain"].to_numpy())
    mi["r1"], mi["r2"] = Z[:, 0], Z[:, 1]
    if mi.loc[mi.domain == "Biology", "r1"].mean() < mi["r1"].mean():      # biology high, based on PC1
        mi["r1"] *= -1
    if mi.loc[mi.domain == "Chemistry", "r2"].mean() < mi["r2"].mean():    # chemistry high, based on PC2
        mi["r2"] *= -1
    return mi


def pca_figures(root: Path, data: Path):
    """PCA scree + item loadings PC1 vs PC2 by subject."""
    load, var = item_pca(data)
    scree_plot(range(1, 11), var[:10], str(root / "figures" / "pca_scree"), highlight=2)
    cat_scatter(load, "pc1", "pc2", "domain", DOMAIN_ORDER, DOMAIN_COLOR,
                f"PC1 ({var[0]:.0%})", f"PC2 ({var[1]:.0%})",
                str(root / "figures" / "pca_loadings"))


def mirt_rotation_figure(root: Path, data: Path):
    """MIRT-K2 item loadings on the two subject LDA axes."""
    cat_scatter(mirt_rot(root, data), "r1", "r2", "domain", DOMAIN_ORDER, DOMAIN_COLOR,
                "MIRT LD1", "MIRT LD2",
                str(root / "figures" / "mirt2_loadings"))


def loadings_pair_figure(root: Path, data: Path):
    """PCA and MIRT-K2 item loadings side by side, each with subject KDE marginals."""
    load, var = item_pca(data)
    mi = mirt_rot(root, data)
    common = dict(group="domain", order=DOMAIN_ORDER, colors=DOMAIN_COLOR)
    joint_pair([
        {"df": load, "xcol": "pc1", "ycol": "pc2", "title": "PCA",
         "xlabel": f"PC1 ({var[0]:.0%})", "ylabel": f"PC2 ({var[1]:.0%})", **common},
        {"df": mi, "xcol": "r1", "ycol": "r2", "title": "MIRT-K2 (LDA)",
         "xlabel": "LD1", "ylabel": "LD2", **common},
    ], str(root / "figures" / "loadings_pair"))


def disc_figure(root: Path, data: Path):
    """2PL item discrimination a: pooled and by subject."""
    dom = pd.read_csv(data / "item_domains.csv")[["item", "domain"]]
    d = pd.read_csv(root / "gpqa_discrimination.csv").merge(dom, on="item")
    density_plot(d, "a_2pl", "domain", DOMAIN_ORDER, DOMAIN_COLOR,
                 "item discrimination a (2PL)", str(root / "figures" / "discrimination"))


def ladder_figure(root: Path):
    """Held-out MAE at full data: baselines + every IRT spec (single 10-fold CV)."""
    raw = pd.read_csv(root / "model_comparison_raw.csv")
    base = raw[raw["spec"] == "1pl"]
    specs = list(dict.fromkeys(raw["spec"]))
    bkeys = ["overall_base", "model_base", "item_base", "both_add", "both_logit"]
    labels = [LABEL[k] for k in bkeys] + [
        s.upper() if s[0].isdigit() else s.replace("mirt-K", "MIRT-K") for s in specs]
    values = [base[k].to_numpy() for k in bkeys] + [
        raw.loc[raw["spec"] == s, "irt"].to_numpy() for s in specs]
    colors = [COLOR[k] for k in bkeys] + [COLOR["irt"]] * len(specs)
    strip(labels, values, colors, "held-out MAE",
          str(root / "figures" / "heldout_ladder"), width=3.7, rot=40)


def comparison_figure(root: Path):
    """Held-out accuracy per IRT spec (1PL / 2PL / MIRT-K); zoomed to the differences."""
    raw = pd.read_csv(root / "model_comparison_raw.csv")
    specs = list(dict.fromkeys(raw["spec"]))
    labels = [s.upper() if s[0].isdigit() else s.replace("mirt-K", "MIRT-K") for s in specs]
    values = [raw.loc[raw["spec"] == s, "irt"].to_numpy() for s in specs]
    strip(labels, values, COLOR["irt"], "held-out MAE",
          str(root / "figures" / "model_comparison"), width=4.2, rot=30, connect=True)


SWEEP_SERIES = ("model_base", "item_base", "both_add", "irt_1pl", "irt_2pl")


def sweep_figure(root: Path):
    """Model-count sweep (2PL) led by 2PL's gain over the both-marginals baseline."""
    raw = pd.read_csv(root / "required_model_count_raw.csv")
    sweep_gap_plot(raw, "n_models", "number of models",
                   str(root / "figures" / "required_model_count"),
                   series=SWEEP_SERIES, a="irt_2pl", b="both_add")


def item_sweep_figure(root: Path):
    """Item-count sweep (2PL, all models kept)."""
    raw = pd.read_csv(root / "required_item_count_raw.csv")
    curve_plot(raw, "n_items", "number of items",
               str(root / "figures" / "required_item_count"), series=SWEEP_SERIES)


def grid_figure(root: Path):
    """Heatmap of 2PL held-out MAE over (n_models, n_items)."""
    raw = pd.read_csv(root / "sweep_grid_raw.csv")
    mat = raw.groupby(["n_models", "n_items"])["irt"].mean().unstack("n_items")
    heatmap(mat, str(root / "figures" / "sweep_grid"),
            "number of items", "number of models", cbar_label="held-out MAE (2PL)")


def main(root="irt_results/epoch_gpqa", data="external_data/epoch-gpqa"):
    root, data = Path(root), Path(data)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    cap = pd.read_csv(root / "gpqa_capabilities.csv")
    dif = pd.read_csv(root / "gpqa_difficulties.csv")
    dia = pd.read_csv(data / "gpqa_diamond.csv")
    theta_figures(root, cap, dia)
    item_figure(root, dif, data)
    agreement_figure(root, cap, dia)
    wright_figure(root, cap, dif)
    pca_figures(root, data)
    disc_figure(root, data)
    mirt_rotation_figure(root, data)
    loadings_pair_figure(root, data)
    ladder_figure(root)
    comparison_figure(root)
    sweep_figure(root)
    item_sweep_figure(root)
    grid_figure(root)


if __name__ == "__main__":
    main()
