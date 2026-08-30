"""Build the final judge-verdict result tables and manuscript figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = {
    "raw-l1": "raw-l1",
    "raw-ridge": "raw-ridge-grid",
    "pc-l1": "pc-l1-global30",
    "pc-ridge": "pc-ridge-global30-grid",
}
LABELS = {
    "raw-l1": "Raw L1", "raw-ridge": "Raw L2",
    "pc-l1": "PC L1", "pc-ridge": "PC L2",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1,
    "svg.fonttype": "none",
})


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_position(("outward", 5))
    axis.spines["bottom"].set_position(("outward", 5))
    axis.tick_params(length=2.5, width=0.8)


def _save(figure, root: Path, name: str) -> None:
    output = root / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=300)
    figure.savefig(output / f"{name}.svg")
    plt.close(figure)
    if "<text" not in (output / f"{name}.svg").read_text():
        raise RuntimeError(f"{name}.svg does not contain editable text")


def _validate(directory: Path, manifest_name: str) -> None:
    manifest = json.loads((directory / manifest_name).read_text())
    if manifest.get("state") != "completed":
        raise RuntimeError(f"incomplete result: {directory}")
    for name, expected in manifest["artifact_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {directory / name}")


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _feature_labels(root: Path) -> dict[str, str]:
    dictionary = pd.read_parquet(root / "metadata/feature-dictionary.parquet")
    output = {}
    for row in dictionary.itertuples():
        members = json.loads(row.leaf_members)
        label = " + ".join(_humanize(member) for member in members)
        output[row.feature] = label
    return output


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance, effects, selections = [], [], []
    for variant, relative in VARIANTS.items():
        directory = root / relative
        _validate(directory / "fit", "manifest.json")
        _validate(directory / "lesions", "summary-manifest.json")
        uncertainty = pd.read_csv(directory / "fit/uncertainty.csv")
        for metric in ("accuracy", "log_loss"):
            item = uncertainty.query("metric == @metric and method == 'item_clustered'").iloc[0]
            fold = uncertainty.query("metric == @metric and method == 'conservative_fold'").iloc[0]
            performance.append({
                "variant": variant, "metric": metric, "estimate": item.estimate,
                "item_lower": item.lower, "item_upper": item.upper,
                "fold_lower": fold.lower, "fold_upper": fold.upper,
            })
        effects.append(pd.read_csv(directory / "lesions/lesions.csv").assign(variant=variant))
        if variant.endswith("l1"):
            selected = pd.read_csv(directory / "fit/selection.csv")
            frequency = selected.feature.value_counts().rename_axis("feature").reset_index(name="folds")
            frequency["fold_share"] = frequency.folds / 16
            selections.append(frequency.assign(variant=variant))
    return pd.DataFrame(performance), pd.concat(effects), pd.concat(selections)


def performance_figure(root: Path, performance: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(3.2, 1.6))
    positions = np.arange(len(VARIANTS))[::-1]
    table = performance.set_index(["metric", "variant"])
    for axis, metric, label in zip(axes, ("accuracy", "log_loss"), ("Accuracy", "Log loss")):
        for position, variant in zip(positions, VARIANTS):
            row = table.loc[(metric, variant)]
            axis.errorbar(
                row.estimate, position,
                xerr=[[row.estimate - row.item_lower], [row.item_upper - row.estimate]],
                fmt="o", color="#333333", ms=4, capsize=0, lw=1,
            )
        values = performance[performance.metric == metric]
        low = np.floor(values.item_lower.min() * 50) / 50
        high = np.ceil(values.item_upper.max() * 50) / 50
        axis.set_xlim(low, high); axis.set_xticks((low, (low + high) / 2, high))
        axis.set_yticks(
            positions,
            [LABELS[variant] for variant in VARIANTS] if metric == "accuracy" else [],
        )
        axis.set_xlabel(label); _style(axis)
    figure.subplots_adjust(left=0.28, right=0.98, bottom=0.28, top=0.96, wspace=0.24)
    _save(figure, root, "performance")


def _effect_figure(
    root: Path,
    effects: pd.DataFrame,
    variants: tuple[str, str],
    name: str,
    labels: dict[str, str],
    *,
    grouped: bool = False,
) -> None:
    effects = effects[effects.variant.isin(variants)]
    if grouped:
        kinds = {"lesion-block", "singular-block", "lesion-family", "singular-family"}
        candidates = effects[effects.kind.isin(kinds)]
        variables = candidates.groupby("variable").estimate.apply(lambda x: x.abs().max()).nlargest(12).index.tolist()
    else:
        candidates = effects[effects.kind.isin({"lesion-feature", "singular-feature"})]
        variables = candidates[candidates.kind == "lesion-feature"].groupby("variable").estimate.apply(
            lambda x: x.abs().max()).nlargest(10).index.tolist()
    figure, axes = plt.subplots(1, 2, figsize=(3.5, min(3.1, 1.05 + 0.23 * len(variables))), sharey=True)
    positions = np.arange(len(variables))[::-1]
    plotted = []
    for axis, variant in zip(axes, variants):
        table = candidates[candidates.variant == variant].set_index(["kind", "variable"])
        for offset, prefix, label, color, marker in (
            (-0.08, "lesion", "Removed", "#222222", "o"),
            (0.08, "singular", "Alone", "#888888", "s"),
        ):
            for position, variable in zip(positions + offset, variables):
                matches = table.loc[
                    [index for index in table.index if index[0].startswith(prefix) and index[1] == variable]
                ] if any(index[0].startswith(prefix) and index[1] == variable for index in table.index) else pd.DataFrame()
                if matches.empty:
                    continue
                row = matches.iloc[0]
                plotted.extend([row.lower, row.upper])
                axis.errorbar(
                    row.estimate, position,
                    xerr=[[row.estimate - row.lower], [row.upper - row.estimate]],
                    fmt=marker, color=color, ms=3.5, capsize=0, lw=1,
                )
        axis.axvline(0, color="#999999", lw=0.8, ls="--")
        axis.text(0, 1.03, LABELS[variant].split()[-1], transform=axis.transAxes, color="#666666", fontsize=7)
        axis.set_xlabel("Δ log loss"); _style(axis)
    bound = max(0.01, np.ceil(max(abs(np.asarray(plotted))) * 100) / 100)
    for axis in axes:
        axis.set_xlim(-bound, bound); axis.set_xticks((-bound, 0, bound))
    axes[0].set_xticklabels((f"−{bound:g}", "0", ""))
    axes[1].set_xticklabels(("", "0", f"{bound:g}"))
    axes[0].set_yticks(positions, [
        textwrap.fill(labels.get(variable, _humanize(variable)), 28)
        for variable in variables
    ])
    handles = [
        plt.Line2D([], [], marker="o", color="#222222", ls="", label="Removed", ms=4),
        plt.Line2D([], [], marker="s", color="#888888", ls="", label="Alone", ms=4),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.44, right=0.92, bottom=0.19, top=0.95, wspace=0.42)
    _save(figure, root, name)


def loadings_figure(root: Path, labels: dict[str, str]) -> pd.DataFrame:
    loadings = pd.read_parquet(root / "global-pc-basis/loadings.parquet")
    selected_rows = []
    for component in ("pc_001", "pc_002"):
        figure, axis = plt.subplots(figsize=(2.6, 2.7))
        values = loadings[loadings.component == component].set_index("feature").loading
        selected = values.loc[values.abs().nlargest(6).index].sort_values()
        selected_rows.append(pd.DataFrame({
            "component": component, "feature": selected.index, "loading": selected.values,
        }))
        positions = np.arange(len(selected))
        axis.scatter(selected, positions, color="#333333", s=10, zorder=3)
        axis.hlines(positions, 0, selected, color="#777777", lw=1)
        bound = np.ceil(max(abs(selected)) * 20) / 20
        axis.set_xlim(-bound, bound); axis.set_xticks((-bound, 0, bound))
        axis.set_yticks(positions, [
            textwrap.fill(labels.get(feature, _humanize(feature)), 20)
            for feature in selected.index
        ])
        axis.set_xlabel("Loading")
        axis.text(0, 1.03, f"PC {int(component.split('_')[1])}", transform=axis.transAxes, color="#666666", fontsize=7)
        _style(axis)
        figure.subplots_adjust(left=0.62, right=0.96, bottom=0.16, top=0.94)
        _save(figure, root, f"{component.replace('_00', '')}-loadings")
    return pd.concat(selected_rows, ignore_index=True)


def build(root: Path) -> None:
    missing = [relative for relative in VARIANTS.values() if not (root / relative / "fit/manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed variants: {missing}")
    tables = root / "tables"; tables.mkdir(exist_ok=True)
    performance, effects, selections = load_results(root)
    performance.to_csv(tables / "performance.csv", index=False)
    effects.to_csv(tables / "effects.csv", index=False)
    selections.to_csv(tables / "l1-selection-frequency.csv", index=False)
    labels = _feature_labels(root)
    performance_figure(root, performance)
    _effect_figure(root, effects, ("raw-l1", "raw-ridge"), "raw-feature-effects", labels)
    _effect_figure(root, effects, ("raw-l1", "raw-ridge"), "raw-group-effects", labels, grouped=True)
    _effect_figure(root, effects, ("pc-l1", "pc-ridge"), "pc-effects", {})
    loadings_figure(root, labels).to_csv(tables / "pc1-pc2-loadings.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build(parser.parse_args().root)


if __name__ == "__main__":
    main()
