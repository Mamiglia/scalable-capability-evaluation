"""Nature-style figures for readable judge-verdict feature EDA."""
import argparse
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure

from .associations import block_correlations, capability_correlations, verdict_correlations
from .basic_eda import correlation_pairs
from .config import CONFIG
from .data import load_features
from .feature_definitions import definitions_for
from .pca import align_pc_signs, pca_tables
from .preprocessing import correlation_groups


BENCHMARKS: dict[str, str] = {"gpqa": "GPQA", "aime": "AIME", "intercode": "InterCode",
                              "swe_bench_mini": "SWE-bench Mini"}
JUDGES: dict[str, str] = {
    "gpt-oss-120b": "gpt-oss", "deepseek-r1-0528": "R1", "o4-mini": "o4-mini",
    "gemma-4-31b-it": "Gemma", "deepseek-v4-flash-0731": "V4 Flash",
    "Llama-3.1-8B-Instruct": "Llama-8B", "ling-2.6-flash": "Ling",
}
BLOCK_LABELS: dict[str, str] = {
    "clause-descriptors": "Clause desc.", "clause-motifs": "Motifs",
    "clause-plan": "Plan", "intermediate-errors": "Errors",
    "derivation": "Deriv.", "exposition": "Expos.", "structure": "Struct.",
}
COLORS: dict[str, str] = {"gpqa": "#2a78d6", "aime": "#eb6834", "intercode": "#1baf7a",
                          "swe_bench_mini": "#4a3aa7"}
DIVERGING = LinearSegmentedColormap.from_list("muted_diverging", ["#2a78d6", "#f7f7f4", "#eb6834"])
SEQUENTIAL = LinearSegmentedColormap.from_list("muted_sequential", ["#f7f7f4", "#4a3aa7"])


def _style() -> None:
    """Apply the project's compact Nature-manuscript plotting defaults."""
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
        "axes.linewidth": 0.7,
    })


def _ordinary(ax: Axes) -> None:
    """Use offset left/bottom spines for non-heatmap axes."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _pretty(value: object, width: int = 24) -> str:
    """Turn a snake-case feature name into a compact wrapped label."""
    return textwrap.fill(str(value).replace("_", " "), width=width)


def _pair_label(feature_a: str, feature_b: str, bench: str | None) -> str:
    """Format one correlation-pair label without multi-line wrapping."""
    def short(value: str) -> str:
        return textwrap.shorten(str(value).replace("_", " "), width=18, placeholder="…")
    benchmark = {"swe_bench_mini": "SWE", "intercode": "IC"}.get(
        bench, BENCHMARKS.get(bench, bench)) if bench else ""
    return f"{short(feature_a)} ↔ {short(feature_b)}" + (f" · {benchmark}" if benchmark else "")


def _verdict_matrices(table: pd.DataFrame, top: int) -> list[tuple[str, pd.DataFrame]]:
    """Build benchmark panels with one globally ranked, shared feature row set."""
    rank = (table.groupby("feature").correlation.apply(lambda values: values.abs().mean())
            .nlargest(top).index)
    return [(bench, frame.pivot(index="feature", columns="judge", values="correlation").reindex(rank))
            for bench, frame in table.groupby("bench", sort=False)]


def _pca_loading_matrices(
    table: pd.DataFrame,
    pcs: int,
    top: int,
) -> list[tuple[str, pd.DataFrame]]:
    """Build benchmark loading panels with globally shared feature rows."""
    frame = table[table.pc <= pcs]
    rank = frame.groupby("feature").abs_loading.mean().nlargest(top).index
    return [(bench, group.pivot(index="feature", columns="pc", values="loading")
             .reindex(rank).rename(columns=lambda pc: f"PC{pc}"))
            for bench, group in frame.groupby("bench", sort=False)]


def _save(fig: Figure, output_dir: str | Path, name: str) -> None:
    """Save a fixed-size 300-DPI PNG and an editable-text SVG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    fig.savefig(output_dir / f"{name}.svg")
    plt.close(fig)


def _heatmap(
    matrix: pd.DataFrame,
    output_dir: str | Path,
    name: str,
    *,
    cmap: Colormap = DIVERGING,
    symmetric: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    panel: str | None = None,
    size: tuple[float, float] = (2.6, 2.4),
    colorbar_label: str | None = None,
) -> None:
    """Render one labelled matrix with a compact external colour bar."""
    _style()
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    values = matrix.to_numpy(float)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    limit = max(limit, 0.01)
    vmin, vmax = (-limit, limit) if symmetric else (0, limit)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    xlabels = [JUDGES.get(x, BENCHMARKS.get(x, _pretty(x, 14))) for x in matrix.columns]
    ax.set_xticks(range(matrix.shape[1]), xlabels, rotation=90, ha="center")
    ax.set_yticks(range(matrix.shape[0]), [_pretty(x) for x in matrix.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if panel:
        ax.set_title(panel, loc="left", color="#666666", fontsize=8, pad=3)
    ticks = [vmin, 0, vmax] if symmetric else [vmin, vmax]
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, ticks=ticks)
    colorbar.ax.tick_params(labelsize=7, length=2)
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=7)
    _save(fig, output_dir, name)


def plot_verdict_heatmaps(
    table: pd.DataFrame,
    output_dir: str | Path,
    top: int = 12,
) -> None:
    """Plot per-judge verdict correlations with identical feature rows in every panel."""
    _style()
    groups, limit = _verdict_matrices(table, top), 0.0
    # A shared colour range makes magnitude comparable across benchmarks.
    for _, matrix in groups:
        limit = max(limit, float(np.nanmax(np.abs(matrix.to_numpy()))))
    rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(4.4, min(4.0, 1.9 * rows)),
                             layout="constrained", squeeze=False)
    image = None
    for ax, (bench, matrix) in zip(axes.flat, groups):
        image = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]), [JUDGES.get(x, x) for x in matrix.columns],
                      rotation=90, ha="center", fontsize=5.5)
        ax.set_yticks(range(matrix.shape[0]), [_pretty(x, 19) for x in matrix.index], fontsize=5.5)
        ax.set_title(BENCHMARKS.get(bench, bench), loc="left", color="#666666", fontsize=8, pad=2)
        ax.set_xlabel("Judge")
        ax.set_ylabel("Feature")
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes.flat[:len(groups)].tolist(), fraction=0.025, pad=0.02,
                           ticks=[-limit, 0, limit])
    colorbar.set_label("Verdict correlation", fontsize=7)
    _save(fig, output_dir, "verdict-correlations")


def plot_capability_heatmap(
    table: pd.DataFrame,
    output_dir: str | Path,
    top: int = 20,
) -> None:
    """Plot within-item rank correlations between features and model accuracy."""
    rank = table.groupby("feature").within_item_rank_rho.apply(lambda x: x.abs().max()).nlargest(top).index
    matrix = table[table.feature.isin(rank)].pivot(index="feature", columns="bench", values="within_item_rank_rho")
    matrix = matrix.reindex(rank)
    height = min(3.2, max(1.6, 0.12 * len(matrix) + 0.7))
    _heatmap(matrix, output_dir, "capability-correlations", xlabel="Benchmark", ylabel="Feature",
             size=(2.7, height))


def plot_block_correlations(table: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot within-/between-block redundancy averaged across benchmarks."""
    summary = table.groupby(["block_a", "block_b"], as_index=False).mean_abs_correlation.mean()
    blocks = sorted(set(summary.block_a) | set(summary.block_b))
    matrix = pd.DataFrame(np.nan, index=blocks, columns=blocks)
    for row in summary.itertuples():
        matrix.loc[row.block_a, row.block_b] = row.mean_abs_correlation
        matrix.loc[row.block_b, row.block_a] = row.mean_abs_correlation
    matrix = matrix.rename(index=BLOCK_LABELS, columns=BLOCK_LABELS)
    size = min(3.2, max(2.0, 0.23 * len(blocks) + 0.8))
    _heatmap(matrix, output_dir, "block-correlations", cmap=SEQUENTIAL, symmetric=False,
             xlabel="Feature block", ylabel="Feature block", size=(size, size),
             colorbar_label="Distinct-feature mean |ρ|")


def plot_top_correlation_pairs(
    table: pd.DataFrame,
    output_dir: str | Path,
    top: int = 3,
) -> None:
    """Plot the strongest signed feature-pair correlations per benchmark."""
    _style()
    groups = list(table.groupby("bench", sort=False))
    fig, axes = plt.subplots(len(groups), 1, figsize=(4.0, min(3.2, 0.8 * len(groups))),
                             layout="constrained", squeeze=False)
    for index, (ax, (bench, frame)) in enumerate(zip(axes.flat, groups)):
        shown = frame.nlargest(top, "abs_correlation").sort_values("abs_correlation")
        labels = [_pair_label(a, b, None)
                  for a, b in shown[["feature_a", "feature_b"]].itertuples(index=False)]
        y = np.arange(len(shown))
        ax.hlines(y, 0, shown.correlation, color="#999999", lw=1)
        # Correlations can equal ±1; disabling clipping keeps endpoint markers circular.
        ax.scatter(shown.correlation, y, color="#222222", s=8, zorder=2, clip_on=False)
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_yticks(y, labels, fontsize=5.5)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_xlabel("Feature correlation" if index == len(groups) - 1 else "")
        if index < len(groups) - 1:
            ax.set_xticklabels([])
        ax.set_title(BENCHMARKS.get(bench, bench), loc="left", color="#666666", fontsize=8, pad=2)
        _ordinary(ax)
    _save(fig, output_dir, "top-feature-correlations")


def plot_scree(table: pd.DataFrame, output_dir: str | Path) -> None:
    """Plot per-component and cumulative variance curves for each benchmark."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7), layout="constrained")
    for bench, frame in table.groupby("bench", sort=False):
        color = COLORS.get(bench, "#555555")
        label = BENCHMARKS.get(bench, bench)
        axes[0].plot(frame.pc, frame.explained_variance_ratio, color=color, lw=1,
                     marker="o", ms=2.5, label=label, clip_on=False)
        axes[1].plot(frame.pc, frame.cumulative, color=color, lw=1,
                     marker="o", ms=2.5, label=label, clip_on=False)
    maximum = int(table.pc.max())
    ticks = sorted(set([1, maximum] + ([5] if maximum >= 5 else []) + ([10] if maximum >= 10 else [])))
    for ax in axes:
        ax.set_xlim(1, maximum)
        ax.set_xticks(ticks)
        ax.set_xlabel("Principal component")
        _ordinary(ax)
    axes[0].set_ylabel("Explained variance")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([0, 0.5, 1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    _save(fig, output_dir, "pca-scree")


def plot_pca_loadings(
    table: pd.DataFrame,
    output_dir: str | Path,
    pcs: int = 4,
    top_per_pc: int = 3,
    alignment: str | None = None,
) -> None:
    """Plot aligned leading-PC loadings with identical feature rows in every panel."""
    _style()
    groups = _pca_loading_matrices(table, pcs, pcs * top_per_pc)
    limit = 0.0
    for _, matrix in groups:
        limit = max(limit, float(np.nanmax(np.abs(matrix.to_numpy()))))
    rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(4.4, min(4.0, 1.9 * rows)),
                             layout="constrained", squeeze=False)
    image = None
    for ax, (bench, matrix) in zip(axes.flat, groups):
        image = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]), matrix.columns)
        ax.set_yticks(range(matrix.shape[0]), [_pretty(x, 19) for x in matrix.index], fontsize=5.5)
        ax.set_title(BENCHMARKS.get(bench, bench), loc="left", color="#666666", fontsize=8, pad=2)
        ax.set_xlabel("Component")
        ax.set_ylabel("Feature")
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes.flat[:len(groups)].tolist(), fraction=0.025, pad=0.02,
                           ticks=[-limit, 0, limit])
    colorbar.set_label("PC loading" + (f" → {alignment}" if alignment else ""), fontsize=7)
    _save(fig, output_dir, "pca-loadings")


def plot_correlation_groups(
    correlations: pd.DataFrame,
    groups: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Plot signed within-group Pearson correlations for shared groups."""
    _style()
    group_dir = Path(output_dir) / "groupings"
    group_dir.mkdir(parents=True, exist_ok=True)
    for old in group_dir.glob("group-*.*"):
        old.unlink()

    benches = correlations.bench.drop_duplicates().tolist()
    cmap = DIVERGING.with_extremes(bad="#dddddd")
    for group_id, members in groups[groups.group_size > 1].groupby(
            "correlation_group", sort=False):
        features = members.feature.tolist()
        ncols, nrows = min(2, len(benches)), int(np.ceil(len(benches) / 2))
        side = min(1.6, max(1.15, 0.16 * len(features) + 0.6))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(min(3.5, ncols * side + 0.7), min(3.2, nrows * side + 0.3)),
            layout="constrained", squeeze=False,
        )
        image = None
        for index, (ax, bench) in enumerate(zip(axes.flat, benches)):
            values = np.where(np.eye(len(features), dtype=bool), 1.0, np.nan)
            matrix = pd.DataFrame(values, index=features, columns=features)
            subset = correlations[
                (correlations.bench == bench)
                & correlations.feature_a.isin(features)
                & correlations.feature_b.isin(features)
            ]
            for pair in subset.itertuples():
                matrix.loc[pair.feature_a, pair.feature_b] = pair.correlation
                matrix.loc[pair.feature_b, pair.feature_a] = pair.correlation
            image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
            labels = [feature.replace("_", " ") for feature in features]
            ax.set_xticks(range(len(features)), labels, rotation=90, ha="center", fontsize=5.5)
            ax.set_yticks(range(len(features)), labels, fontsize=5.5)
            ax.tick_params(labelbottom=index // ncols == nrows - 1,
                           labelleft=index % ncols == 0)
            panel = textwrap.fill(BENCHMARKS.get(bench, bench), width=10)
            ax.set_title(panel, loc="left", color="#666666",
                         fontsize=8, pad=2)
        for ax in axes.flat[len(benches):]:
            ax.set_visible(False)
        colorbar = fig.colorbar(image, ax=axes.flat[:len(benches)].tolist(), fraction=0.025,
                               pad=0.02, ticks=[-1, 0, 1])
        colorbar.set_label("Pearson r", fontsize=7)
        _save(fig, group_dir, group_id)


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    """Read one saved model artifact and validate its plotting columns."""
    table = pd.read_csv(path)
    if missing := sorted(required - set(table)):
        raise ValueError(f"{path} missing plotting columns: {missing}")
    if table.empty:
        raise ValueError(f"{path} contains no plotting rows")
    return table


def _configured_order(values: pd.Series, configured: Sequence[str]) -> list[str]:
    """Keep configured labels first, then retain saved-table order."""
    present = values.drop_duplicates().astype(str).tolist()
    return [value for value in configured if value in present] + [
        value for value in present if value not in configured
    ]


def _symmetric_limit(*values: pd.Series) -> float:
    """Choose a readable common scale with explicit endpoint ticks."""
    finite = np.concatenate([value.to_numpy(float) for value in values])
    maximum = float(np.nanmax(np.abs(finite))) if np.isfinite(finite).any() else 1.0
    return max(0.1, np.ceil(maximum * 2) / 2)


def _interval_columns(table: pd.DataFrame, metric: str | None = None) -> tuple[str, str] | None:
    """Find saved interval columns without manufacturing uncertainty."""
    options = [("lower", "upper")]
    if metric:
        options.extend([(f"delta_{metric}_lower", f"delta_{metric}_upper"),
                        (f"{metric}_lower", f"{metric}_upper")])
    return next((pair for pair in options if set(pair) <= set(table)), None)


def plot_coefficients(
    coefficient_csv: str | Path,
    output_dir: str | Path,
    *,
    judges: Sequence[str] = CONFIG.judges,
    benchmarks: Sequence[str] = CONFIG.benchmarks,
    feature_order: Sequence[str] = CONFIG.regressors,
    coefficient_limit: float | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Plot saved effective coefficients; only HiBayes rows receive intervals."""
    _style()
    table = _read_csv(coefficient_csv, {"backend", "feature", "judge", "benchmark", "effective"})
    outputs: dict[str, tuple[Path, Path]] = {}
    for backend, frame in table.groupby("backend", sort=False):
        features = _configured_order(frame.feature, feature_order)
        panel_judges = _configured_order(frame.judge, judges)
        panel_benches = _configured_order(frame.benchmark, benchmarks)
        interval = _interval_columns(frame) if backend == "hibayes" else None
        scale_values = [frame.effective]
        if interval:
            scale_values.extend([frame[interval[0]], frame[interval[1]]])
        limit = coefficient_limit if coefficient_limit is not None else _symmetric_limit(*scale_values)
        rows, columns = len(panel_judges), len(panel_benches)
        height = max(1.25 * rows, 0.18 * len(features) * rows + 0.65)
        fig, axes = plt.subplots(
            rows, columns, figsize=(min(4.4, max(1.8, 1.55 * columns + 0.25)), height),
            layout="constrained", squeeze=False, sharex=True,
        )
        for row, judge in enumerate(panel_judges):
            for column, benchmark in enumerate(panel_benches):
                ax = axes[row, column]
                subset = frame[(frame.judge == judge) & (frame.benchmark == benchmark)].copy()
                subset["feature_index"] = subset.feature.map({feature: index for index, feature in enumerate(features)})
                subset = subset.dropna(subset=["effective", "feature_index"]).sort_values("feature_index")
                folds = _configured_order(
                    subset.fold if "fold" in subset else pd.Series("saved", index=subset.index), ()
                )
                fold_dodge = dict(zip(folds, np.linspace(-0.16, 0.16, len(folds))))
                y = subset.feature_index.to_numpy(float) + (
                    subset.fold.map(fold_dodge).to_numpy(float) if "fold" in subset else 0
                )
                if interval:
                    lower, upper = interval
                    xerr = np.vstack((subset.effective - subset[lower], subset[upper] - subset.effective))
                    ax.errorbar(subset.effective, y, xerr=xerr, fmt="none", color="#555555",
                                lw=1, capsize=0, clip_on=False, zorder=1)
                ax.scatter(subset.effective, y, s=18, color="#222222", clip_on=False, zorder=2)
                ax.axvline(0, color="#999999", lw=0.8, ls="--", zorder=0)
                ax.set_xlim(-limit, limit)
                ax.set_xticks([-limit, 0, limit])
                ax.set_ylim(len(features) - 0.5, -0.5)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels([_pretty(feature, 17) for feature in features] if column == 0 else [])
                ax.set_title(f"{JUDGES.get(judge, judge)} · {BENCHMARKS.get(benchmark, benchmark)}",
                             loc="left", color="#666666", fontsize=7, pad=2)
                if row == rows - 1:
                    ax.set_xlabel("Effective coefficient")
                _ordinary(ax)
        name = f"coefficients-{backend}"
        _save(fig, output_dir, name)
        base = Path(output_dir) / name
        outputs[str(backend)] = (base.with_suffix(".png"), base.with_suffix(".svg"))
    return outputs


def plot_lesions(
    lesion_csv: str | Path,
    output_dir: str | Path,
    *,
    metric: str = CONFIG.lesion_metric,
    feature_order: Sequence[str] = CONFIG.regressors,
) -> tuple[Path, Path]:
    """Plot saved paired lesion deltas and their saved bootstrap intervals."""
    value = f"delta_{metric}"
    table = _read_csv(lesion_csv, {"backend", "feature", value})
    features = _configured_order(table.feature, feature_order)
    backends = table.backend.drop_duplicates().astype(str).tolist()
    interval = _interval_columns(table, metric)
    scale_values = [table[value]]
    if interval:
        scale_values.extend([table[interval[0]], table[interval[1]]])
    limit = _symmetric_limit(*scale_values)
    _style()
    fig, ax = plt.subplots(
        figsize=(2.25, max(1.35, 0.24 * len(features) + 0.6)), layout="constrained"
    )
    colors = {"frequentist": "#2a78d6", "hibayes": "#4a3aa7"}
    dodges = np.linspace(-0.16, 0.16, len(backends)) if len(backends) > 1 else np.array([0.0])
    for backend, dodge in zip(backends, dodges):
        subset = table[table.backend == backend].set_index("feature").reindex(features).dropna(subset=[value])
        y = np.array([features.index(feature) for feature in subset.index], dtype=float) + dodge
        color = colors.get(backend, "#555555")
        if interval:
            lower, upper = interval
            xerr = np.vstack((subset[value] - subset[lower], subset[upper] - subset[value]))
            ax.errorbar(subset[value], y, xerr=xerr, fmt="none", color=color, lw=1,
                        capsize=0, clip_on=False, zorder=1)
        ax.scatter(subset[value], y, s=18, color=color, label={
            "frequentist": "Frequentist", "hibayes": "HiBayes",
        }.get(backend, backend), clip_on=False, zorder=2)
    ax.axvline(0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])
    ax.set_ylim(len(features) - 0.5, -0.5)
    ax.set_yticks(range(len(features)), [_pretty(feature, 17) for feature in features])
    ax.set_xlabel("Δ log loss" if metric == "log_loss" else "Δ accuracy")
    _ordinary(ax)
    if len(backends) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(backends), frameon=False)
    _save(fig, output_dir, "lesions")
    base = Path(output_dir) / "lesions"
    return base.with_suffix(".png"), base.with_suffix(".svg")


def plot_saved_model_results(config=CONFIG) -> dict[str, tuple[Path, Path]]:
    """Render configured saved model outputs without invoking any fitting code."""
    coefficient_files = sorted(config.fit_dir.rglob("coefficients.csv"))
    lesion_files = sorted(config.lesion_dir.rglob("lesions.csv"))
    if not coefficient_files:
        raise FileNotFoundError(f"no coefficients.csv files below {config.fit_dir}")
    if not lesion_files:
        raise FileNotFoundError(f"no lesions.csv files below {config.lesion_dir}")
    def saved_table(paths: list[Path], root: Path) -> pd.DataFrame:
        tables = []
        for path in paths:
            table = pd.read_csv(path)
            parts = path.relative_to(root).parts
            if "backend" not in table and parts:
                table["backend"] = parts[0]
            if "fold" not in table and len(parts) > 2:
                table["fold"] = parts[1]
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    coefficient_table = saved_table(coefficient_files, config.fit_dir)
    lesion_table = saved_table(lesion_files, config.lesion_dir)
    hibayes = coefficient_table[coefficient_table.backend == "hibayes"]
    scale_values = [coefficient_table.effective]
    if interval := _interval_columns(hibayes):
        scale_values.extend([hibayes[interval[0]], hibayes[interval[1]]])
    coefficient_limit = _symmetric_limit(*scale_values)
    config.model_figure_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".plot-", dir=config.model_figure_dir) as temporary:
        coefficient_path = Path(temporary) / "coefficients.csv"
        lesion_path = Path(temporary) / "lesions.csv"
        coefficient_table.to_csv(coefficient_path, index=False)
        lesion_table.to_csv(lesion_path, index=False)
        outputs = plot_coefficients(coefficient_path, config.model_figure_dir,
                                    judges=config.judges, benchmarks=config.benchmarks,
                                    feature_order=config.regressors,
                                    coefficient_limit=coefficient_limit)
        outputs["lesions"] = plot_lesions(lesion_path, config.model_figure_dir,
                                           metric=config.lesion_metric,
                                           feature_order=config.regressors)
        return outputs


def main(argv: Sequence[str] | None = None) -> None:
    """Compute tables and render the configured readable-feature EDA bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=CONFIG.results_dir)
    parser.add_argument("--blocks", nargs="+", default=list(CONFIG.feature_blocks))
    parser.add_argument("--splits", nargs="+", default=list(CONFIG.splits))
    parser.add_argument("--config", type=Path, default=CONFIG.design_matrix_config)
    args = parser.parse_args(argv)

    features = list(CONFIG.regressors) or None
    transcripts, transcript_dictionary = load_features(
        "transcript", args.blocks, features=features, splits=args.splits, config=args.config)
    pairs, pair_dictionary = load_features(
        "pair", args.blocks, features=features, splits=args.splits, config=args.config)
    verdict = verdict_correlations(pairs[pairs.judge.isin(CONFIG.judges)], pair_dictionary)
    capability = capability_correlations(transcripts, transcript_dictionary)
    identity = [column for column in ["bench", "item", "model_left", "model_right",
                "epoch_left", "epoch_right", "log_left", "log_right"] if column in pairs]
    distinct_pairs = pairs.drop_duplicates(identity) if identity else pairs
    correlations = correlation_pairs(distinct_pairs, pair_dictionary, threshold=0, method="spearman")
    grouping_correlations = correlation_pairs(
        distinct_pairs, pair_dictionary, threshold=0, method="pearson")
    groups = correlation_groups(
        grouping_correlations, pair_dictionary, CONFIG.correlation_threshold)
    blocks = block_correlations(correlations)
    variance, raw_loadings = pca_tables(
        transcripts, transcript_dictionary, CONFIG.pca_components,
        common_support=CONFIG.pca_common_support)
    stronger_loadings, stronger_directions = align_pc_signs(
        raw_loadings, capability, "within_item_rank_rho", alignment="stronger")
    judge_loadings, judge_directions = align_pc_signs(
        raw_loadings, verdict, "correlation", alignment="judge")
    loadings = {"stronger": stronger_loadings, "judge": judge_loadings}[CONFIG.pca_alignment]

    data_dir, figure_dir = args.output_dir / "data", args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "verdict-correlations.csv": verdict,
        "capability-correlations.csv": capability,
        "feature-correlations.csv": correlations,
        "grouping-correlations.csv": grouping_correlations,
        "correlation-groups.csv": groups,
        "block-correlations.csv": blocks,
        "pca-variance.csv": variance,
        "pca-loadings.csv": loadings,
        "pca-loadings-raw.csv": raw_loadings,
        "pca-loadings-stronger.csv": stronger_loadings,
        "pca-loadings-judge.csv": judge_loadings,
        "pca-directions-stronger.csv": stronger_directions,
        "pca-directions-judge.csv": judge_directions,
        "feature-definitions.csv": definitions_for(transcript_dictionary),
    }.items():
        frame.to_csv(data_dir / name, index=False)

    plot_verdict_heatmaps(verdict, figure_dir, CONFIG.top_verdict_features)
    plot_capability_heatmap(capability, figure_dir, CONFIG.top_capability_features)
    plot_block_correlations(blocks, figure_dir)
    plot_top_correlation_pairs(correlations, figure_dir)
    plot_correlation_groups(grouping_correlations, groups, figure_dir)
    plot_scree(variance, figure_dir)
    plot_pca_loadings(loadings, figure_dir, alignment=CONFIG.pca_alignment)
    print(f"rows: transcripts={len(transcripts)} pairs={len(pairs)}; "
          f"features={len(transcript_dictionary)} blocks={len(args.blocks)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
