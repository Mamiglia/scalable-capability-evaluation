"""Generate the IRT validation figures from the stored tables."""

from pathlib import Path

import pandas as pd

from irt.data import pick_epoch
from irt.plots import (COLOR, LABEL, cat_scatter, curve_plot, heatmap, scatter,
                       strip, sweep_gap_plot, wright_map)

# categorical palette (dataviz default, light surface); "Other" stays muted grey
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER = "#a8a79d"


def spearman(x, y):
    """Rank correlation, no scipy."""
    r = pd.DataFrame({"x": x, "y": y}).rank()
    return r["x"].corr(r["y"])


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
                note=f"Spearman {spearman(c.mean_score, c.theta_1pl):+.3f}  ·  n={len(c)}/{len(cap)}")
    cy = c.dropna(subset=["year"])
    cat_scatter(cy, "year", "theta_1pl", "family", order, colors,
                "release date", "capability θ (1PL)",
                str(fig / "theta_vs_year"),
                note=f"Spearman {spearman(cy.year, cy.theta_1pl):+.3f}  ·  n={len(cy)}/{len(cap)}")


DOMAIN_ORDER = ["Physics", "Chemistry", "Biology"]
DOMAIN_COLOR = {"Physics": "#2a78d6", "Chemistry": "#eb6834", "Biology": "#1baf7a"}


def item_figure(root: Path, dif: pd.DataFrame, data: Path):
    """Item difficulty b vs empirical item %-correct, coloured by subject domain."""
    resp = pick_epoch(pd.read_parquet(data / "responses.parquet"))
    rate = resp.groupby(["benchmark", "item"])["correct"].mean().rename("passrate").reset_index()
    d = dif.merge(rate, on=["benchmark", "item"])
    note = None
    out = str(root / "figures" / "b_vs_passrate")
    dom_path = data / "item_domains.csv"
    if dom_path.exists():
        d = d.merge(pd.read_csv(dom_path)[["item", "domain"]], on="item")
        note = f"Spearman {spearman(d.passrate, d.b_1pl):+.3f}  ·  n={len(d)}"
        cat_scatter(d, "passrate", "b_1pl", "domain", DOMAIN_ORDER, DOMAIN_COLOR,
                    "item %-correct", "difficulty b (1PL)", out, note=note, note_xy=(0.03, 0.06))
    else:
        scatter(d, "passrate", "b_1pl", "item %-correct", "difficulty b (1PL)", out,
                color=COLOR["item_base"], note_xy=(0.03, 0.06),
                note=f"Spearman {spearman(d.passrate, d.b_1pl):+.3f}  ·  n={len(d)}")


def agreement_figure(root: Path, cap: pd.DataFrame):
    """1PL vs 2PL capability agreement."""
    scatter(cap, "theta_1pl", "theta_2pl", "θ (1PL)", "θ (2PL)",
            str(root / "figures" / "theta_1pl_vs_2pl"), color=COLOR["irt"], identity=True,
            note=f"Spearman {spearman(cap.theta_1pl, cap.theta_2pl):+.3f}  ·  n={len(cap)}")


def wright_figure(root: Path, cap: pd.DataFrame, dif: pd.DataFrame):
    """Model ability and item difficulty on one latent scale."""
    wright_map(cap["theta_1pl"].to_numpy(), dif["b_1pl"].to_numpy(),
               str(root / "figures" / "wright_map"))


def ladder_figure(root: Path):
    """Held-out accuracy of the four predictors at the full model count."""
    raw = pd.read_csv(root / "required_model_count_raw.csv")
    raw = raw[raw["n_models"] == raw["n_models"].max()]
    keys = ["overall_base", "model_base", "item_base", "irt"]
    strip([LABEL[k] for k in keys], [raw[k].to_numpy() for k in keys],
          [COLOR[k] for k in keys], "held-out accuracy",
          str(root / "figures" / "heldout_ladder"))


def comparison_figure(root: Path):
    """Held-out accuracy per IRT spec (1PL / 2PL / MIRT-K); zoomed to the differences."""
    raw = pd.read_csv(root / "model_comparison_raw.csv")
    specs = list(dict.fromkeys(raw["spec"]))
    labels = [s.upper() if s[0].isdigit() else s.replace("mirt-K", "MIRT-K") for s in specs]
    values = [raw.loc[raw["spec"] == s, "irt"].to_numpy() for s in specs]
    strip(labels, values, COLOR["irt"], "held-out accuracy",
          str(root / "figures" / "model_comparison"))


def sweep_figure(root: Path):
    """Model-count sweep with the IRT-over-model gap panel."""
    raw = pd.read_csv(root / "required_model_count_raw.csv")
    sweep_gap_plot(raw, "n_models", "number of models",
                   str(root / "figures" / "required_model_count"))


def item_sweep_figure(root: Path):
    """Item-count sweep (all models kept)."""
    raw = pd.read_csv(root / "required_item_count_raw.csv")
    curve_plot(raw, "n_items", "number of items",
               str(root / "figures" / "required_item_count"))


def grid_figure(root: Path):
    """Heatmap of IRT held-out accuracy over (n_models, n_items)."""
    raw = pd.read_csv(root / "sweep_grid_raw.csv")
    mat = raw.groupby(["n_models", "n_items"])["irt"].mean().unstack("n_items")
    heatmap(mat, str(root / "figures" / "sweep_grid"),
            "number of items", "number of models")


def main(root="irt_results/epoch_gpqa", data="external_data/epoch-gpqa"):
    root, data = Path(root), Path(data)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    cap = pd.read_csv(root / "gpqa_capabilities.csv")
    dif = pd.read_csv(root / "gpqa_difficulties.csv")
    dia = pd.read_csv(data / "gpqa_diamond.csv")
    theta_figures(root, cap, dia)
    item_figure(root, dif, data)
    agreement_figure(root, cap)
    wright_figure(root, cap, dif)
    ladder_figure(root)
    comparison_figure(root)
    sweep_figure(root)
    item_sweep_figure(root)
    grid_figure(root)


if __name__ == "__main__":
    main()
