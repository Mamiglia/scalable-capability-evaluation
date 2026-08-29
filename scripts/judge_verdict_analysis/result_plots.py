"""Compute and plot judge-verdict results from explicit saved artifacts only."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t


VARIANT_LABELS = {
    "readable": "Readable",
    "pca-20": "PCA-20",
    "readable-surprisal": "Readable +\nsurprisal",
    "readable-sae": "Readable + SAE",
    "readable-surprisal-sae": "Readable + both",
}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7")
BLOCK_COLORS = {
    "surface": "#2a78d6", "structure": "#eb6834", "rubric": "#1baf7a",
    "exposition": "#eda100", "derivation": "#e87ba4",
    "intermediate-errors": "#4a3aa7", "clause-descriptors": "#778899",
    "clause-motifs": "#a16f46", "clause-plan": "#5f8f67",
}
FEATURE_LABELS = {"bare_answer": "minimal non-answer prose"}


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if table.empty:
        raise ValueError(f"{path} contains no rows")
    return table


def _require(table: pd.DataFrame, required: set[str], path: str | Path) -> None:
    if missing := sorted(required - set(table)):
        raise ValueError(f"{path} missing columns: {missing}")


def _with_fold_dimensions(table: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    parsed = table.fold.astype(str).str.rsplit("__item-", n=1, expand=True)
    if parsed.shape[1] != 2 or parsed.isna().any().any() or (parsed == "").any().any():
        raise ValueError(f"{path} fold values must use MODEL_FOLD__item-ITEM_FOLD")
    output = table.copy()
    for column, values in (("model_fold", parsed[0]), ("item_fold", parsed[1])):
        if column in output and not output[column].astype(str).reset_index(drop=True).equals(
            values.astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"{path} has inconsistent fold and {column} values")
        output[column] = values.astype(str).to_numpy()
    return output


def load_prediction_artifacts(
    paths: Mapping[str, Sequence[str | Path] | str | Path],
) -> pd.DataFrame:
    """Load saved out-of-fold prediction shards named by model variant."""
    frames = []
    for variant, variant_paths in paths.items():
        files = [variant_paths] if isinstance(variant_paths, (str, Path)) else list(variant_paths)
        if not files:
            raise ValueError(f"{variant} has no prediction artifacts")
        for path_value in files:
            path = Path(path_value)
            table = _read_table(path)
            if "benchmark" in table and "bench" in table:
                if not table.benchmark.equals(table.bench):
                    raise ValueError(f"{path} has inconsistent bench and benchmark columns")
                table = table.drop(columns="bench")
            elif "bench" in table:
                table = table.rename(columns={"bench": "benchmark"})
            _require(table, {"judge", "benchmark", "observed", "probability"}, path)
            if "fold" not in table:
                table.insert(0, "fold", path.parent.name)
            table = _with_fold_dimensions(table, path)
            for column in ("item", "model_left", "model_right"):
                if column not in table:
                    table[column] = ""
            output = table[[
                "fold", "model_fold", "item_fold", "judge", "benchmark",
                "item", "model_left", "model_right", "observed", "probability",
            ]].copy()
            output.insert(0, "variant", str(variant))
            frames.append(output)
    if not frames:
        raise ValueError("no prediction artifacts supplied")
    output = pd.concat(frames, ignore_index=True)
    probability = output.probability.to_numpy(float)
    observed = output.observed.to_numpy()
    if (
        not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or not np.isin(observed, [0, 1]).all()
    ):
        raise ValueError("prediction outcomes and probabilities must be finite values in [0, 1]")
    output["observed"] = output.observed.astype(int)
    output["probability"] = probability
    return output


def load_rejudge_summary(path: str | Path) -> pd.DataFrame:
    """Load the saved exact-rejudge cell summary and its conservative floor."""
    table = _read_table(path)
    if "benchmark" not in table and "bench" in table:
        table = table.rename(columns={"bench": "benchmark"})
    required = {"judge", "benchmark", "wilson_95_low"}
    _require(table, required, path)
    output = table[["judge", "benchmark", "wilson_95_low"]].copy()
    values = output.wilson_95_low.to_numpy(float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("wilson_95_low must contain finite values in [0, 1]")
    if output.duplicated(["judge", "benchmark"]).any():
        raise ValueError("exact-rejudge summary has duplicate judge×benchmark cells")
    output["wilson_95_low"] = values
    return output


def _fold_interval(values: pd.Series) -> tuple[float, float]:
    count = len(values)
    if count < 3:
        return np.nan, np.nan
    margin = float(t.ppf(0.975, count - 1) * values.std(ddof=1) / np.sqrt(count))
    mean = float(values.mean())
    return max(0.0, mean - margin), min(1.0, mean + margin)


def _effect_interval(values: np.ndarray) -> tuple[float, float]:
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    mean = float(values.mean())
    return mean - margin, mean + margin


def accuracy_comparison(
    predictions: pd.DataFrame, rejudge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold-replicated OOF accuracy overall and by judge×benchmark."""
    _require(
        predictions,
        {"variant", "fold", "judge", "benchmark", "observed", "probability"},
        "predictions",
    )
    _require(rejudge, {"judge", "benchmark", "wilson_95_low"}, "rejudge")
    rows = _with_fold_dimensions(predictions, "predictions").merge(
        rejudge, on=["judge", "benchmark"], how="left", validate="many_to_one",
    )
    if rows.wilson_95_low.isna().any():
        missing = rows.loc[rows.wilson_95_low.isna(), ["judge", "benchmark"]].drop_duplicates()
        raise ValueError(f"rejudge summary is missing prediction cells: {missing.to_dict('records')}")
    rows["correct"] = ((rows.probability >= 0.5) == rows.observed).astype(float)
    scoped = pd.concat([
        rows.assign(scope="overall", group_judge="", group_benchmark=""),
        rows.assign(scope="judge×benchmark", group_judge=rows.judge,
                    group_benchmark=rows.benchmark),
    ], ignore_index=True)
    keys = ["variant", "scope", "group_judge", "group_benchmark"]
    summary = scoped.groupby(keys, sort=False, as_index=False).agg(
        estimate=("correct", "mean"), exact_floor=("wilson_95_low", "mean"),
        row_count=("correct", "size"), model_fold_count=("model_fold", "nunique"),
        item_fold_count=("item_fold", "nunique"),
    )
    points = []
    for dimension in ("model_fold", "item_fold"):
        frame = scoped.groupby([*keys, dimension], sort=False, as_index=False).agg(
            accuracy=("correct", "mean"), exact_floor=("wilson_95_low", "mean"),
            row_count=("correct", "size"),
        ).rename(columns={dimension: "replicate"})
        frame.insert(2, "replication_dimension", dimension)
        points.append(frame)
    replication_scores = pd.concat(points, ignore_index=True).rename(columns={
        "group_judge": "judge", "group_benchmark": "benchmark",
    })
    intervals = []
    for group, frame in replication_scores.groupby(
        ["variant", "scope", "judge", "benchmark"], sort=False,
    ):
        model_values = frame.loc[frame.replication_dimension == "model_fold", "accuracy"]
        item_values = frame.loc[frame.replication_dimension == "item_fold", "accuracy"]
        if len(model_values) >= 3 and len(item_values) >= 3:
            model_interval, item_interval = _fold_interval(model_values), _fold_interval(item_values)
            lower, upper = min(model_interval[0], item_interval[0]), max(model_interval[1], item_interval[1])
            method = "conservative union of model-fold/item-fold t95"
        else:
            lower, upper = np.nan, np.nan
            method = "model-fold/item-fold points"
        intervals.append((*group, lower, upper, method))
    interval_table = pd.DataFrame(intervals, columns=[
        "variant", "scope", "group_judge", "group_benchmark", "lower", "upper",
        "uncertainty_method",
    ])
    summary = summary.merge(interval_table, on=keys, validate="one_to_one").rename(columns={
        "group_judge": "judge", "group_benchmark": "benchmark",
    })
    return summary, replication_scores


def load_lesion_summary(
    lesion_path: str | Path, feature_dictionary_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load saved lesion effects and optionally attach feature blocks."""
    table = _read_table(lesion_path)
    _require(table, {"kind", "variable", "reference", "estimate", "lower", "upper"}, lesion_path)
    output = table.copy()
    values = output[["estimate", "lower", "upper"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:, 1] > values[:, 0]) or np.any(values[:, 0] > values[:, 2]):
        raise ValueError("lesion estimates and intervals must be finite and ordered")
    if feature_dictionary_path is not None:
        dictionary = _read_table(feature_dictionary_path)
        _require(dictionary, {"feature", "block"}, feature_dictionary_path)
        if dictionary.feature.duplicated().any():
            raise ValueError("feature dictionary contains duplicate features")
        block = dictionary[["feature", "block"]].rename(columns={"feature": "variable"})
        output = output.drop(columns="block", errors="ignore").merge(
            block, on="variable", how="left", validate="many_to_one",
        )
    return output


def load_benchmark_feature_effects(
    item_deltas_path: str | Path, feature_dictionary_path: str | Path,
    *, by_judge: bool = False,
) -> pd.DataFrame:
    """Summarize saved item deltas by benchmark with two-way fold uncertainty."""
    rows = _read_table(item_deltas_path)
    required = {
        "bench", "fold", "kind", "variable", "row_count", "delta_log_loss",
    }
    if by_judge:
        required.add("judge")
    _require(rows, required, item_deltas_path)
    rows = rows[rows.kind.isin(("singular-feature", "lesion-feature"))].copy()
    if rows.empty:
        raise ValueError("item deltas contain no feature effects")
    rows = _with_fold_dimensions(rows, item_deltas_path)
    rows["effect"] = rows.kind.map({
        "singular-feature": "singular", "lesion-feature": "unique",
    })
    dictionary = _read_table(feature_dictionary_path)
    _require(dictionary, {"feature", "block"}, feature_dictionary_path)
    blocks = dictionary[["feature", "block"]].rename(columns={"feature": "variable"})
    rows = rows.merge(blocks, on="variable", how="left", validate="many_to_one")
    if rows.block.isna().any():
        raise ValueError("feature dictionary is missing item-delta features")
    summaries = []
    group_columns = ["bench", "variable", "block", "effect"]
    if by_judge:
        group_columns.insert(0, "judge")
    for keys, frame in rows.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values_by_key = dict(zip(group_columns, keys))
        weights = frame.row_count.to_numpy(float)
        estimate = float(np.average(frame.delta_log_loss, weights=weights))
        dimensions = []
        for dimension in ("model_fold", "item_fold"):
            dimensions.append(np.array([
                np.average(group.delta_log_loss, weights=group.row_count)
                for _, group in frame.groupby(dimension, sort=False)
            ], dtype=float))
        if all(len(values) >= 3 for values in dimensions):
            intervals = [_effect_interval(values) for values in dimensions]
            lower, upper = min(value[0] for value in intervals), max(value[1] for value in intervals)
        else:
            lower, upper = np.nan, np.nan
        summary = {
            "benchmark": values_by_key["bench"], "feature": values_by_key["variable"],
            "block": values_by_key["block"], "effect": values_by_key["effect"],
            "estimate": estimate, "lower": lower, "upper": upper,
            "model_fold_count": len(dimensions[0]), "item_fold_count": len(dimensions[1]),
            "row_count": int(weights.sum()),
        }
        if by_judge:
            summary["judge"] = values_by_key["judge"]
        summaries.append(summary)
    return pd.DataFrame(summaries)


def load_lesion_coefficients(artifact_index_path: str | Path) -> pd.DataFrame:
    """Load the complete full-reference coefficient checkpoint for every lesion fold."""
    artifact_index_path = Path(artifact_index_path)
    index = _read_table(artifact_index_path)
    _require(index, {"fold", "specification", "coefficients"}, artifact_index_path)
    folds = set(index.fold.astype(str))
    index = index[index.specification == "full"]
    if (index.empty or index.fold.duplicated().any()
            or set(index.fold.astype(str)) != folds):
        raise ValueError("artifact index must contain one full coefficient checkpoint per fold")
    frames = []
    for row in index.itertuples(index=False):
        path = Path(row.coefficients)
        if not path.is_absolute():
            path = artifact_index_path.parent / path
        table = _read_table(path)
        _require(table, {"feature", "judge", "benchmark", "effective"}, path)
        effective = table.effective.to_numpy(float)
        if not np.isfinite(effective).all():
            raise ValueError("effective coefficients must be finite")
        frames.append(table.assign(fold=str(row.fold), effective=effective)[[
            "fold", "feature", "judge", "benchmark", "effective",
        ]])
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["fold", "feature", "judge", "benchmark"]).any():
        raise ValueError("full coefficient checkpoints contain duplicate cells")
    supports = [
        set(zip(frame.feature, frame.judge, frame.benchmark))
        for _, frame in output.groupby("fold", sort=False)
    ]
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError("full coefficient checkpoints lack identical feature×judge×benchmark cell support")
    return output


def summarize_effective_coefficients(
    coefficients: pd.DataFrame, *, by: Sequence[str] = (),
) -> pd.DataFrame:
    """Average requested cells within fold, then report mean and range over folds."""
    _require(coefficients, {"fold", "feature", "effective", *by}, "coefficients")
    keys = [*by, "feature"]
    per_fold = coefficients.groupby(
        [*keys, "fold"], sort=False, as_index=False,
    ).effective.mean()
    return (per_fold.groupby(keys, sort=False, as_index=False).effective
            .agg(estimate="mean", lower="min", upper="max", fold_count="size"))


def _paired_effects(
    table: pd.DataFrame, singular_kind: str, unique_kind: str, key: str,
) -> pd.DataFrame:
    columns = ["variable", "estimate", "lower", "upper"]
    singular = table[table.kind == singular_kind][columns].rename(columns={
        "variable": key, "estimate": "singular", "lower": "singular_lower",
        "upper": "singular_upper",
    })
    unique = table[table.kind == unique_kind][columns].rename(columns={
        "variable": key, "estimate": "unique", "lower": "unique_lower",
        "upper": "unique_upper",
    })
    if singular[key].duplicated().any() or unique[key].duplicated().any():
        raise ValueError(f"duplicate saved effects for {singular_kind} or {unique_kind}")
    paired = singular.merge(
        unique, on=key, how="outer", validate="one_to_one", indicator=True,
    )
    if not paired._merge.eq("both").all():
        missing = paired.loc[paired._merge != "both", [key, "_merge"]].to_dict("records")
        raise ValueError(f"incomplete paired effects for {singular_kind}/{unique_kind}: {missing}")
    return paired.drop(columns="_merge")


def feature_effects(lesions: pd.DataFrame) -> pd.DataFrame:
    """Pair singular improvement with unique removal loss for each feature."""
    output = _paired_effects(lesions, "singular-feature", "lesion-feature", "feature")
    if "block" in lesions:
        blocks = lesions[lesions.kind.str.endswith("-feature")][["variable", "block"]].dropna().drop_duplicates()
        if blocks.variable.duplicated().any():
            raise ValueError("features map to multiple blocks")
        output = output.merge(
            blocks.rename(columns={"variable": "feature"}), on="feature", how="left",
            validate="one_to_one",
        )
    if output.empty:
        raise ValueError("lesion summary contains no paired feature effects")
    return output.sort_values("singular", ascending=False, kind="stable").reset_index(drop=True)


def group_effects(lesions: pd.DataFrame) -> pd.DataFrame:
    """Pair singular and unique effects for every saved block and family."""
    frames = []
    for level in ("block", "family"):
        frame = _paired_effects(
            lesions, f"singular-{level}", f"lesion-{level}", "group",
        )
        if not frame.empty:
            frame.insert(0, "level", level)
            frames.append(frame)
    if not frames:
        raise ValueError("lesion summary contains no paired block or family effects")
    return pd.concat(frames, ignore_index=True)


def coefficient_selection_stability(
    selection_paths: Sequence[str | Path], coefficient_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize readable-fit selection and mean effective coefficients across folds."""
    if not selection_paths:
        raise ValueError("no readable selection artifacts supplied")
    selections: dict[str, set[str]] = {}
    for value in selection_paths:
        path = Path(value)
        table = _read_table(path)
        _require(table, {"feature"}, path)
        if table.feature.duplicated().any() or path.parent.name in selections:
            raise ValueError("selection artifacts must contain one unique file and feature set per fold")
        selections[path.parent.name] = set(table.feature.astype(str))
    coefficient_frames = []
    seen_folds = set()
    for value in coefficient_paths:
        path = Path(value)
        table = _read_table(path)
        _require(table, {"feature", "effective"}, path)
        fold = path.parent.name
        if fold in seen_folds:
            raise ValueError("coefficient artifacts must contain one file per fold")
        seen_folds.add(fold)
        values = table.effective.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("effective coefficients must be finite")
        table = table.assign(fold=fold, effective=values)
        coefficient_frames.append(table[["fold", "feature", "effective"]])
    if not coefficient_frames:
        raise ValueError("no readable coefficient artifacts supplied")
    if set(selections) != seen_folds:
        raise ValueError(
            f"selection and coefficient fold sets differ: {sorted(selections)} != {sorted(seen_folds)}"
        )
    fold_coefficients = pd.concat(coefficient_frames, ignore_index=True).groupby(
        ["fold", "feature"], sort=False, as_index=False,
    ).effective.mean()
    features = sorted(
        set().union(*selections.values()) | set(fold_coefficients.feature.astype(str))
    )
    full_index = pd.MultiIndex.from_product(
        [list(selections), features], names=["fold", "feature"],
    )
    fold_coefficients = fold_coefficients.set_index(["fold", "feature"]).reindex(
        full_index, fill_value=0.0,
    ).reset_index()
    rows = []
    for feature in features:
        values = fold_coefficients.loc[fold_coefficients.feature == feature, "effective"]
        lower, upper = (np.nan, np.nan)
        if len(values) >= 3:
            margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            lower, upper = float(values.mean() - margin), float(values.mean() + margin)
        rows.append({
            "feature": feature,
            "selected_folds": sum(feature in selected for selected in selections.values()),
            "total_folds": len(selections),
            "selection_rate": sum(feature in selected for selected in selections.values()) / len(selections),
            "coefficient_fold_count": len(values),
            "coefficient_mean": float(values.mean()),
            "coefficient_sd": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
            "coefficient_lower": lower, "coefficient_upper": upper,
            "sign_consistency": max(float((values > 0).mean()), float((values < 0).mean())),
        })
    summary = pd.DataFrame(rows).sort_values(
        ["selection_rate", "coefficient_mean"], ascending=[False, False], kind="stable",
    ).reset_index(drop=True)
    return summary, fold_coefficients


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
        "axes.linewidth": 0.7,
    })


def _ordinary(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _save(fig: Figure, output_dir: str | Path, name: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png, svg = output_dir / f"{name}.png", output_dir / f"{name}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    if "<text" not in svg.read_text():
        raise RuntimeError(f"editable SVG text was not preserved in {svg}")
    return png, svg


def _symmetric_limit(*frames: pd.DataFrame, columns: Sequence[str]) -> float:
    values = np.concatenate([
        frame[list(columns)].to_numpy(float).ravel() for frame in frames if not frame.empty
    ])
    maximum = float(np.nanmax(np.abs(values[np.isfinite(values)])))
    return max(0.05, np.ceil(maximum * 20 - 1e-12) / 20)


def _rounded_limits(
    frame: pd.DataFrame, columns: Sequence[str],
) -> tuple[tuple[float, float], list[float]]:
    values = frame[list(columns)].to_numpy(float).ravel()
    values = values[np.isfinite(values)]
    lower, upper = min(0.0, float(values.min())), max(0.0, float(values.max()))
    span = upper - lower
    if span == 0:
        step = 0.05
    else:
        raw = span / 4
        magnitude = 10 ** np.floor(np.log10(raw))
        step = next(factor * magnitude for factor in (1, 2, 2.5, 5, 10)
                    if factor * magnitude >= raw)
    lower = float(np.round(np.floor(lower / step) * step, 12))
    upper = float(np.round(np.ceil(upper / step) * step, 12))
    if lower == upper:
        upper = float(np.round(lower + step, 12))
    return (lower, upper), list(dict.fromkeys((lower, 0.0, upper)))


def _estimate_limits(values: pd.Series) -> tuple[tuple[float, float], list[float]]:
    values = values.to_numpy(float)
    values = values[np.isfinite(values)]
    lower, upper = min(0.0, float(values.min())), max(0.0, float(values.max()))
    span = upper - lower
    step = 0.05 if span == 0 else 10 ** np.floor(np.log10(span / 4))
    lower = float(np.round(np.floor(lower / step) * step, 12))
    upper = float(np.round(np.ceil(upper / step) * step, 12))
    if lower == upper:
        upper = float(np.round(lower + step, 12))
    return (lower, upper), [lower, upper]


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = f"{abs(value):g}"
    if magnitude.startswith("0."):
        magnitude = magnitude[1:]
    return ("−" if value < 0 else "") + magnitude


def _feature_label(value: object) -> str:
    return FEATURE_LABELS.get(str(value), str(value).replace("_", " "))


def _variant_order(values: pd.Series) -> list[str]:
    present = values.drop_duplicates().astype(str).tolist()
    configured = [variant for variant in VARIANT_LABELS if variant in present]
    return configured + [variant for variant in present if variant not in configured]


def plot_accuracy_comparison(
    summary: pd.DataFrame, fold_scores: pd.DataFrame, output_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    """Plot OOF accuracy overall and in shared-scale judge×benchmark cells."""
    _style()
    variants = _variant_order(summary.variant)
    overall = summary[summary.scope == "overall"].set_index("variant").reindex(variants)
    fig, ax = plt.subplots(figsize=(1.8, 2.8), layout="constrained")
    x = np.arange(len(variants))
    for index, variant in enumerate(variants):
        row = overall.loc[variant]
        if row.model_fold_count >= 3 and row.item_fold_count >= 3:
            ax.errorbar(index, row.estimate, yerr=[[row.estimate - row.lower], [row.upper - row.estimate]],
                        fmt="none", color="#2a78d6", lw=1, capsize=0, clip_on=False)
        else:
            points = fold_scores[
                (fold_scores.scope == "overall") & (fold_scores.variant == variant)
            ]
            for dimension, offset, marker in (
                ("model_fold", -0.06, "o"), ("item_fold", 0.06, "x"),
            ):
                values = points[points.replication_dimension == dimension].accuracy
                colors = ({"facecolors": "none", "edgecolors": "#2a78d6"}
                          if marker == "o" else {"color": "#2a78d6"})
                ax.scatter(np.full(len(values), index + offset), values, marker=marker,
                           s=7, lw=0.7, clip_on=False, zorder=2, **colors)
        ax.scatter(index, row.estimate, s=16, color="#2a78d6", clip_on=False, zorder=3)
    ax.scatter(x, overall.exact_floor, marker="D", s=10, facecolors="none",
               edgecolors="#777777", lw=0.8, clip_on=False, zorder=3)
    ax.set_xlim(-0.5, len(variants) - 0.5)
    _ordinary(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [VARIANT_LABELS.get(value, value).replace("\n", " ") for value in variants],
        rotation=90, ha="center", rotation_mode="anchor",
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("OOF accuracy")
    fig.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2a78d6",
               markeredgecolor="#2a78d6", label="Mean"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#2a78d6", label="Model fold"),
        Line2D([], [], marker="x", color="#2a78d6", lw=0, label="Item fold"),
        Line2D([], [], marker="D", color="none", markerfacecolor="none",
               markeredgecolor="#777777", label="Rejudge\nfloor"),
    ], loc="outside upper center", ncol=2, frameon=False)
    outputs = {"oof-accuracy-overall": _save(fig, output_dir, "oof-accuracy-overall")}

    cells = summary[summary.scope == "judge×benchmark"].copy()
    cell_order = cells[["judge", "benchmark"]].drop_duplicates().itertuples(index=False, name=None)
    cell_order = list(cell_order)
    labels = [f"{judge} · {benchmark}" for judge, benchmark in cell_order]
    height = min(3.2, max(2.2, 0.11 * len(cell_order) + 0.9))
    fig, ax = plt.subplots(figsize=(3.5, height), layout="constrained")
    dodges = dict(zip(variants, np.linspace(-0.25, 0.25, len(variants))))
    cell_index = {cell: index for index, cell in enumerate(cell_order)}
    for color, variant in zip(COLORS, variants):
        frame = cells[cells.variant == variant]
        for row in frame.itertuples():
            y = cell_index[(row.judge, row.benchmark)] + dodges[variant]
            if row.model_fold_count >= 3 and row.item_fold_count >= 3:
                ax.errorbar(row.estimate, y, xerr=[[row.estimate - row.lower], [row.upper - row.estimate]],
                            fmt="none", color=color, lw=1, capsize=0, clip_on=False)
            else:
                points = fold_scores[
                    (fold_scores.scope == "judge×benchmark")
                    & (fold_scores.variant == variant)
                    & (fold_scores.judge == row.judge)
                    & (fold_scores.benchmark == row.benchmark)
                ]
                for dimension, marker in (("model_fold", "o"), ("item_fold", "x")):
                    values = points[points.replication_dimension == dimension].accuracy
                    colors = ({"facecolors": "none", "edgecolors": color}
                              if marker == "o" else {"color": color})
                    ax.scatter(values, np.full(len(values), y), marker=marker, s=5,
                               lw=0.6, clip_on=False, **colors)
            ax.scatter(row.estimate, y, s=12, color=color, clip_on=False, zorder=3)
    floor_rows = cells.drop_duplicates(["judge", "benchmark"])
    ax.scatter(floor_rows.exact_floor, [cell_index[(row.judge, row.benchmark)] for row in floor_rows.itertuples()],
               marker="D", s=10, facecolors="none", edgecolors="#555555", lw=0.8,
               clip_on=False, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(len(cell_order) - 0.5, -0.5)
    ax.set_yticks(range(len(cell_order)), [textwrap.fill(label, 28) for label in labels])
    ax.set_xlabel("OOF accuracy")
    _ordinary(ax)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=color,
               markeredgecolor=color,
               label={
                   "readable": "Readable", "pca-20": "PCA-20",
                   "readable-surprisal": "+ surprisal", "readable-sae": "+ SAE",
                   "readable-surprisal-sae": "+ both",
               }.get(variant, variant))
        for color, variant in zip(COLORS, variants)
    ]
    handles.append(Line2D([], [], marker="D", color="none", markerfacecolor="none",
                          markeredgecolor="#555555", label="Rejudge floor"))
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="#777777", label="Model fold"))
    handles.append(Line2D([], [], marker="x", color="#777777", lw=0, label="Item fold"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    outputs["oof-accuracy-by-cell"] = _save(fig, output_dir, "oof-accuracy-by-cell")
    return outputs


def _block_color(block: object) -> str:
    return BLOCK_COLORS.get(str(block), "#777777")


def _block_legend(fig: Figure, blocks: Sequence[object]) -> None:
    unique = list(dict.fromkeys(map(str, blocks)))
    fig.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor=_block_color(block),
               markeredgecolor=_block_color(block),
               label=textwrap.fill(block.replace("-", " "), width=13))
        for block in unique
    ], loc="outside lower center", ncol=min(2, len(unique)), frameon=False,
       fontsize=6, columnspacing=0.8, handletextpad=0.4)


def plot_feature_scatter(
    features: pd.DataFrame, output_dir: str | Path, *, top_labels: int = 8,
) -> tuple[Path, Path]:
    """Plot paired singular-vs-unique feature effects, colored by feature block."""
    _require(features, {
        "feature", "block", "singular", "singular_lower", "singular_upper",
        "unique", "unique_lower", "unique_upper",
    }, "features")
    _style()
    xlim, xticks = _estimate_limits(features.singular)
    ylim, yticks = _estimate_limits(features.unique)
    fig, ax = plt.subplots(figsize=(3.2, 3.2), layout="constrained")
    for block, frame in features.groupby("block", sort=False):
        ax.scatter(frame.singular, frame.unique, s=15, color=_block_color(block),
                   clip_on=False, label=str(block))
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks, [_format_tick(value) for value in xticks])
    ax.set_yticks(yticks, [_format_tick(value) for value in yticks])
    ax.set_xlabel("Singular improvement")
    ax.set_ylabel("Unique removal loss")
    _ordinary(ax)
    _block_legend(fig, features.block)
    priority = (
        "hedging_count", "think_aloud_per_100_words", "bare_answer", "tool_errors",
        "rigour", "depth", "self_corrections", "efficiency",
    )
    preferred = features[features.feature.isin(priority)].assign(
        order=lambda frame: frame.feature.map({value: index for index, value in enumerate(priority)}),
    ).sort_values("order")
    fallback = features[~features.feature.isin(priority)].assign(
        order=lambda frame: frame.unique.abs(),
    ).sort_values(["order", "singular"], ascending=False, kind="stable")
    labelled = pd.concat([preferred, fallback], ignore_index=True).head(top_labels)
    origin = np.array([(-xlim[0]) / (xlim[1] - xlim[0]),
                       (-ylim[0]) / (ylim[1] - ylim[0])])
    points = np.column_stack((
        (labelled.singular - xlim[0]) / (xlim[1] - xlim[0]),
        (labelled.unique - ylim[0]) / (ylim[1] - ylim[0]),
    ))
    radial = points - origin
    norms = np.linalg.norm(radial, axis=1)
    radial = radial / np.where(norms[:, None] == 0, 1, norms[:, None])
    angles = np.arctan2(radial[:, 1], radial[:, 0])
    stagger = np.empty(len(labelled))
    stagger[np.argsort(angles, kind="stable")] = (
        np.arange(len(labelled)) - (len(labelled) - 1) / 2
    ) * 0.045
    annotations = []
    geometry = []
    for index, row in enumerate(labelled.itertuples()):
        direction = radial[index] if norms[index] else np.array([1.0, 0.0])
        tangent = np.array([-direction[1], direction[0]])
        label_position = points[index] + direction * 0.06 + tangent * stagger[index]
        leader = label_position - points[index]
        annotation = ax.annotate(
            textwrap.fill(_feature_label(row.feature), width=16),
            (row.singular, row.unique), xytext=label_position,
            textcoords="axes fraction", ha="left" if leader[0] >= 0 else "right",
            va="bottom" if leader[1] >= 0 else "top", fontsize=6,
            color="#444444", annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.5},
        )
        annotations.append(annotation)
        geometry.append((points[index], direction, tangent, stagger[index]))
    fig.canvas.draw()
    fig.set_layout_engine(None)
    accepted = []
    placements = list(zip(annotations, geometry))
    placements.sort(
        key=lambda value: (
            value[1][0] + value[1][1] * 0.06 + value[1][2] * value[1][3]
        )[1],
        reverse=True,
    )
    for annotation, (point, direction, tangent, initial) in placements:
        placed = False
        for distance in (0.06, 0.10, 0.14, 0.18):
            for delta in (
                0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12, 0.16, -0.16,
                0.20, -0.20, 0.24, -0.24, 0.28, -0.28, 0.32, -0.32,
            ):
                position = point + direction * distance + tangent * (initial + delta)
                leader = position - point
                annotation.set_position(position)
                annotation.set_ha("left" if leader[0] >= 0 else "right")
                annotation.set_va("bottom" if leader[1] >= 0 else "top")
                fig.canvas.draw()
                box = annotation.get_window_extent(fig.canvas.get_renderer())
                if (not any(box.overlaps(other.get_window_extent(fig.canvas.get_renderer()))
                            for other in accepted)
                        and fig.bbox.contains(box.x0, box.y0)
                        and fig.bbox.contains(box.x1, box.y1)):
                    placed = True
                    break
            if placed:
                break
        if not placed:
            plt.close(fig)
            raise RuntimeError(
                f"could not place scatter labels without overlap or clipping: {annotation.get_text()}"
            )
        accepted.append(annotation)
    return _save(fig, output_dir, "feature-scatter")


def _plot_feature_axis(
    ax, values: pd.DataFrame, effect: str, *, labels: bool = True,
    wrap_width: int | None = 17, label_size: float = 5.5,
) -> None:
    point = values[effect].to_numpy(float)
    lower, upper = values[f"{effect}_lower"].to_numpy(float), values[f"{effect}_upper"].to_numpy(float)
    y = np.arange(len(values))
    color = {"singular": "#2a78d6", "unique": "#eb6834", "coefficient": "#4a3aa7"}[effect]
    ax.errorbar(point, y, xerr=np.vstack((point - lower, upper - point)),
                fmt="none", color=color, lw=1, capsize=0)
    ax.scatter(point, y, s=12, color=color, clip_on=False, zorder=3)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    if effect == "coefficient":
        limit = _symmetric_limit(values, columns=(effect,))
        limits, ticks = (-limit, limit), [-limit, 0, limit]
    else:
        limits, ticks = _estimate_limits(values[effect])
    ax.set_xlim(limits)
    ax.set_xticks(ticks, [_format_tick(value) for value in ticks])
    if labels:
        tick_labels = [_feature_label(value) for value in values.feature]
        if wrap_width is not None:
            tick_labels = [textwrap.fill(value, wrap_width) for value in tick_labels]
        ax.set_yticks(y, tick_labels)
    else:
        ax.set_yticks(y)
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.5, len(values) - 0.5)
    ax.tick_params(axis="y", labelsize=label_size, length=0, pad=1)
    ax.set_title(effect.capitalize(), loc="left", color="#666666", fontsize=8, pad=2)
    ax.set_xlabel("β (1 SD)" if effect == "coefficient" else "Δ log loss")
    _ordinary(ax)


def _attach_coefficients(features: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    _require(coefficients, {"feature", "estimate", "lower", "upper"}, "coefficients")
    renamed = coefficients[["feature", "estimate", "lower", "upper"]].rename(columns={
        "estimate": "coefficient", "lower": "coefficient_lower",
        "upper": "coefficient_upper",
    })
    output = features.merge(renamed, on="feature", how="left", validate="one_to_one")
    if output.coefficient.isna().any():
        raise ValueError("coefficient summary is missing feature rows")
    return output


def plot_feature_overview(
    features: pd.DataFrame, coefficients: pd.DataFrame, output_dir: str | Path,
    *, top: int = 12,
) -> tuple[Path, Path]:
    """Plot shared feature rows by singular, unique, and effective coefficient."""
    required = {
        "feature", "block", "singular", "singular_lower", "singular_upper",
        "unique", "unique_lower", "unique_upper",
    }
    _require(features, required, "features")
    features = _attach_coefficients(features, coefficients)
    rank = features[["singular", "unique", "coefficient"]].abs().rank(pct=True).max(axis=1)
    values = features.assign(rank=rank).nlargest(min(top, len(features)), "rank").sort_values("rank")
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(4.0, 2.65), sharey=True, layout="constrained")
    for index, effect in enumerate(("singular", "unique", "coefficient")):
        _plot_feature_axis(axes[index], values, effect, labels=index == 0)
    return _save(fig, output_dir, "feature-effects-overview")


def plot_feature_blocks(
    features: pd.DataFrame, coefficients: pd.DataFrame, output_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    """Plot one shared-row three-column figure for every source block."""
    required = {
        "feature", "block", "singular", "singular_lower", "singular_upper",
        "unique", "unique_lower", "unique_upper",
    }
    _require(features, required, "features")
    features = _attach_coefficients(features, coefficients)
    outputs = {}
    for block, frame in features.groupby("block", sort=False):
        frame = frame.assign(
            rank=frame[["singular", "unique", "coefficient"]].abs().rank(pct=True).max(axis=1),
        ).sort_values("rank")
        height = min(3.2, max(1.4, 0.105 * len(frame) + 0.75))
        _style()
        fig, axes = plt.subplots(1, 3, figsize=(4.0, height), sharey=True, layout="constrained")
        for index, effect in enumerate(("singular", "unique", "coefficient")):
            _plot_feature_axis(
                axes[index], frame, effect, labels=index == 0,
                wrap_width=None, label_size=5,
            )
        name = f"feature-effects-{block}"
        outputs[name] = _save(fig, output_dir, name)
    return outputs


def plot_group_effects(
    groups: pd.DataFrame, output_dir: str | Path,
) -> tuple[Path, Path]:
    """Plot paired singular/unique block and family effects."""
    required = {
        "level", "group", "singular", "singular_lower", "singular_upper",
        "unique", "unique_lower", "unique_upper",
    }
    _require(groups, required, "groups")
    _style()
    limit = _symmetric_limit(
        groups, columns=("singular_lower", "singular_upper", "unique_lower", "unique_upper"),
    )
    frame = groups.assign(rank=groups[["singular", "unique"]].abs().max(axis=1)).sort_values("rank")
    fig, ax = plt.subplots(figsize=(2.8, 3.0), layout="constrained")
    y = np.arange(len(frame))
    labels = []
    for row in frame.itertuples():
        label = str(row.group).replace("-", " ")
        if row.level == "family" and row.group == "planning":
            label = "planning (clause-plan)"
        labels.append(f"{row.level.title()} · {label}")
    for effect, dodge, color in (
        ("singular", -0.12, "#2a78d6"), ("unique", 0.12, "#eb6834"),
    ):
        ax.errorbar(frame[effect], y + dodge,
                    xerr=np.vstack((frame[effect] - frame[f"{effect}_lower"],
                                    frame[f"{effect}_upper"] - frame[effect])),
                    fmt="none", color=color, lw=1, capsize=0)
        ax.scatter(frame[effect], y + dodge, s=12, color=color, clip_on=False, zorder=3)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])
    ax.set_yticks(y, labels, fontsize=5.5)
    ax.set_ylim(-0.5, len(frame) - 0.5)
    ax.set_xlabel("Δ log loss")
    _ordinary(ax)
    fig.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2a78d6",
               markeredgecolor="#2a78d6", label="Singular"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#eb6834",
               markeredgecolor="#eb6834", label="Unique removal"),
    ], loc="outside upper center", ncol=2, frameon=False)
    return _save(fig, output_dir, "group-effects")


def _selected_feature_rows(effects: pd.DataFrame, top: int) -> list[str]:
    support = []
    for feature, frame in effects[effects.effect == "unique"].groupby("feature", sort=False):
        positive = frame[frame.lower > 0]
        support.append({
            "feature": feature, "positive_support": len(positive),
            "mean_positive": positive.estimate.mean(), "peak": frame.estimate.max(),
        })
    selected = (pd.DataFrame(support).fillna({"mean_positive": -np.inf}).sort_values(
        ["positive_support", "mean_positive", "peak", "feature"],
        ascending=[False, False, False, True], kind="stable",
    ).head(top).feature.tolist())
    return list(reversed(selected))


def _plot_benchmark_grid(
    effects: pd.DataFrame, coefficients: pd.DataFrame, output_dir: str | Path,
    *, name: str, top: int = 10, variables: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Plot four benchmarks by singular, unique, and coefficient columns."""
    _require(effects, {
        "benchmark", "feature", "block", "effect", "estimate", "lower", "upper",
    }, "benchmark effects")
    _require(coefficients, {
        "benchmark", "feature", "estimate", "lower", "upper", "fold_count",
    }, "benchmark coefficients")
    benchmarks = [value for value in ("aime", "gpqa", "intercode", "swe_bench_mini")
                  if value in set(effects.benchmark)]
    if not benchmarks:
        raise ValueError("benchmark effects contain no plottable rows")
    variables = list(variables) if variables is not None else _selected_feature_rows(effects, top)
    if not variables:
        raise ValueError("benchmark effects contain no plottable rows")
    _style()
    fig, axes = plt.subplots(
        len(benchmarks), 3, figsize=(4.0, 1.375 * len(benchmarks)),
        layout="constrained", squeeze=False,
    )
    column_points = [[], [], []]
    used_blocks = []
    benchmark_labels = {
        "aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
        "swe_bench_mini": "SWE",
    }
    for row_index, benchmark in enumerate(benchmarks):
        benchmark_rows = effects[effects.benchmark == benchmark]
        block_map = benchmark_rows.drop_duplicates("feature").set_index("feature").block.to_dict()
        y = np.arange(len(variables))
        for column, effect in enumerate(("singular", "unique")):
            ax = axes[row_index, column]
            values = (benchmark_rows[benchmark_rows.effect == effect]
                      .set_index("feature").reindex(variables))
            if values.estimate.isna().any():
                raise ValueError(f"{benchmark} is missing selected {effect} effect rows")
            for point_y, value in enumerate(values.itertuples()):
                if pd.isna(value.estimate):
                    continue
                block = block_map[value.Index]
                if block not in used_blocks:
                    used_blocks.append(block)
                color = _block_color(block)
                if np.isfinite(value.lower) and np.isfinite(value.upper):
                    ax.errorbar(value.estimate, point_y,
                                xerr=[[value.estimate - value.lower], [value.upper - value.estimate]],
                                fmt="none", color=color, lw=0.8, capsize=0)
                ax.plot(value.estimate, point_y, "o", ms=2.2, color=color,
                        clip_on=False, zorder=3)
            column_points[column].append(values.estimate.to_numpy(float))
        values = (coefficients[coefficients.benchmark == benchmark]
                  .set_index("feature").reindex(variables))
        if values[["estimate", "lower", "upper"]].isna().any().any():
            raise ValueError(f"{benchmark} is missing coefficient rows for selected features")
        ax = axes[row_index, 2]
        for point_y, value in enumerate(values.itertuples()):
            if pd.isna(value.estimate):
                continue
            color = _block_color(block_map[value.Index])
            ax.errorbar(value.estimate, point_y,
                        xerr=[[value.estimate - value.lower], [value.upper - value.estimate]],
                        fmt="none", color=color, lw=0.8, capsize=0)
            ax.plot(value.estimate, point_y, "o", ms=2.2, color=color,
                    clip_on=False, zorder=3)
        column_points[2].append(values.estimate.to_numpy(float))
        tick_labels = [_feature_label(value) for value in variables]
        for column, ax in enumerate(axes[row_index]):
            ax.axvline(0, color="#999999", lw=0.6, ls="--")
            ax.set_yticks(y, tick_labels if column == 0 else [], fontsize=3.8)
            ax.set_ylim(0, len(variables) - 1)
            ax.tick_params(axis="y", length=0, pad=1)
            ax.tick_params(axis="x", labelsize=5.5, length=2, width=0.6, pad=1)
            if row_index < len(benchmarks) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            _ordinary(ax)
        axes[row_index, 0].text(
            -0.03, 1.02, benchmark_labels[benchmark],
            transform=axes[row_index, 0].transAxes, ha="right", va="bottom",
            fontsize=5.5, color="#666666",
        )
    for column, title in enumerate(("Singular", "Unique", "Coefficient")):
        axes[0, column].set_title(title, loc="left", color="#666666", fontsize=7, pad=10)
        points = pd.Series(np.concatenate(column_points[column]))
        if column == 2:
            limit = _symmetric_limit(pd.DataFrame({"value": points}), columns=("value",))
            limits, ticks = (-limit, limit), [-limit, 0, limit]
        else:
            limits, ticks = _estimate_limits(points)
        for ax in axes[:, column]:
            ax.set_xlim(limits)
            ax.set_xticks(ticks, [_format_tick(value) for value in ticks])
    axes[-1, 0].set_xlabel("Δ log loss")
    axes[-1, 1].set_xlabel("Δ log loss")
    axes[-1, 2].set_xlabel("β (1 SD)")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=3, color=_block_color(block),
               label=str(block).replace("-", " "))
        for block in used_blocks
    ], loc="outside lower center", ncol=3, frameon=False, fontsize=5.5)
    return _save(fig, output_dir, name)


def plot_benchmark_feature_grid(
    effects: pd.DataFrame, coefficients: pd.DataFrame, output_dir: str | Path,
    *, top: int = 10,
) -> tuple[Path, Path]:
    """Plot pooled-judge benchmark effects in the capability contribution grammar."""
    required = {"aime", "gpqa", "intercode", "swe_bench_mini"}
    if missing := sorted(required - set(effects.benchmark)):
        raise ValueError(f"pooled benchmark grid requires all four benchmarks; missing {missing}")
    return _plot_benchmark_grid(
        effects, coefficients, output_dir, name="feature-effects-by-benchmark", top=top,
    )


def plot_judge_benchmark_grids(
    effects: pd.DataFrame, coefficients: pd.DataFrame, output_dir: str | Path,
    *, top: int = 10,
) -> dict[str, tuple[Path, Path]]:
    """Plot each judge's available benchmarks with exact cell coefficients."""
    _require(effects, {"judge"}, "judge benchmark effects")
    _require(coefficients, {"judge"}, "judge benchmark coefficients")
    outputs = {}
    variables = _selected_feature_rows(effects, top)
    for judge, frame in effects.groupby("judge", sort=False):
        name = f"feature-effects-{str(judge).replace('/', '-')}-by-benchmark"
        outputs[name] = _plot_benchmark_grid(
            frame, coefficients[coefficients.judge == judge], output_dir,
            name=name, top=top, variables=variables,
        )
    return outputs


def plot_annotation_comparison(groups: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Compare annotations-only improvement with annotation-removal loss."""
    frame = groups[(groups.level == "family") & (groups.group == "annotations")]
    if len(frame) != 1:
        raise ValueError("expected one paired annotations-family effect")
    row = frame.iloc[0]
    limit = _symmetric_limit(
        frame, columns=("singular_lower", "singular_upper", "unique_lower", "unique_upper"),
    )
    _style()
    fig, ax = plt.subplots(figsize=(1.7, 1.1), layout="constrained")
    for effect, y, color in (("singular", -0.08, "#2a78d6"), ("unique", 0.08, "#eb6834")):
        ax.errorbar(row[effect], y,
                    xerr=[[row[effect] - row[f"{effect}_lower"]],
                          [row[f"{effect}_upper"] - row[effect]]],
                    fmt="none", color=color, lw=1, capsize=0)
        ax.scatter(row[effect], y, s=14, color=color, clip_on=False, zorder=3)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-limit, 0, limit])
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Δ log loss")
    _ordinary(ax)
    fig.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2a78d6",
               markeredgecolor="#2a78d6", label="Singular"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#eb6834",
               markeredgecolor="#eb6834", label="Unique"),
    ], loc="outside lower center", ncol=2, frameon=False)
    return _save(fig, output_dir, "annotation-comparison")


def plot_stability(
    stability: pd.DataFrame, fold_coefficients: pd.DataFrame, output_dir: str | Path,
    *, top: int = 15,
) -> tuple[Path, Path]:
    """Write the stability table and plot selection/coefficient behavior across folds."""
    required = {
        "feature", "selection_rate", "coefficient_mean", "coefficient_fold_count",
        "coefficient_lower", "coefficient_upper",
    }
    _require(stability, required, "stability")
    _require(fold_coefficients, {"fold", "feature", "effective"}, "fold coefficients")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stability.to_csv(output_dir / "coefficient-selection-stability.csv", index=False)
    frame = stability.head(top).iloc[::-1].reset_index(drop=True)
    limit = _symmetric_limit(frame, columns=("coefficient_mean",))
    _style()
    height = min(3.2, max(1.6, 0.18 * len(frame) + 0.9))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, height), layout="constrained", sharey=True)
    y = np.arange(len(frame))
    axes[0].scatter(frame.selection_rate, y, s=15, color="#2a78d6", clip_on=False)
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].set_xlabel("Selection rate")
    axes[0].set_yticks(y, [textwrap.fill(_feature_label(value), 20) for value in frame.feature])
    axes[1].axvline(0, color="#999999", lw=0.8, ls="--")
    for row_y, row in zip(y, frame.itertuples()):
        if row.coefficient_fold_count >= 3:
            axes[1].errorbar(row.coefficient_mean, row_y,
                             xerr=[[row.coefficient_mean - row.coefficient_lower],
                                   [row.coefficient_upper - row.coefficient_mean]],
                             fmt="none", color="#eb6834", lw=1, capsize=0, clip_on=False)
        else:
            values = fold_coefficients[fold_coefficients.feature == row.feature].effective
            axes[1].scatter(values, np.full(len(values), row_y), s=6, facecolors="none",
                            edgecolors="#eb6834", lw=0.6, clip_on=False)
    axes[1].scatter(frame.coefficient_mean, y, s=15, color="#eb6834", clip_on=False, zorder=3)
    axes[1].set_xlim(-limit, limit)
    axes[1].set_xticks([-limit, 0, limit])
    axes[1].set_xlabel("Mean coefficient")
    for ax in axes:
        _ordinary(ax)
    fig.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2a78d6",
               markeredgecolor="#2a78d6", label="Selected folds"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#eb6834",
               markeredgecolor="#eb6834", label="Fold mean"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#eb6834", label="Per fold (N < 3)"),
    ], loc="outside lower center", ncol=3, frameon=False)
    return _save(fig, output_dir, "coefficient-selection-stability")


def write_result_plots(
    prediction_paths: Mapping[str, Sequence[str | Path] | str | Path],
    rejudge_path: str | Path,
    lesion_path: str | Path,
    feature_dictionary_path: str | Path,
    selection_paths: Sequence[str | Path],
    coefficient_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    item_deltas_path: str | Path | None = None,
    judge_item_deltas_path: str | Path | None = None,
) -> dict[str, dict[str, Path | tuple[Path, Path]]]:
    """Write all requested tables and figures without fitting or discovering inputs."""
    if set(prediction_paths) != set(VARIANT_LABELS):
        raise ValueError(
            "top-level workflow requires exactly the five requested variants: "
            f"{list(VARIANT_LABELS)}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dirs = {
        name: output_dir / name
        for name in ("overview", "by-feature-group", "by-benchmark", "by-judge", "secondary")
    }
    for path in figure_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    predictions = load_prediction_artifacts(prediction_paths)
    summary, fold_scores = accuracy_comparison(predictions, load_rejudge_summary(rejudge_path))
    lesions = load_lesion_summary(lesion_path, feature_dictionary_path)
    features, groups = feature_effects(lesions), group_effects(lesions)
    stability, fold_coefficients = coefficient_selection_stability(selection_paths, coefficient_paths)
    lesion_coefficients = load_lesion_coefficients(Path(lesion_path).with_name("artifact-index.csv"))
    overall_coefficients = summarize_effective_coefficients(lesion_coefficients)
    benchmark_coefficients = summarize_effective_coefficients(
        lesion_coefficients, by=("benchmark",),
    )
    cell_coefficients = summarize_effective_coefficients(
        lesion_coefficients, by=("judge", "benchmark"),
    )
    tables = {
        "accuracy-comparison": summary, "fold-accuracy": fold_scores,
        "feature-effects": features, "group-effects": groups,
        "coefficient-selection-stability": stability,
        "coefficient-effects": overall_coefficients,
        "benchmark-coefficients": benchmark_coefficients,
        "judge-benchmark-coefficients": cell_coefficients,
    }
    benchmark_effects = None
    if item_deltas_path is not None:
        benchmark_effects = load_benchmark_feature_effects(
            item_deltas_path, feature_dictionary_path,
        )
        tables["benchmark-feature-effects"] = benchmark_effects
    judge_effects = None
    if judge_item_deltas_path is not None:
        judge_effects = load_benchmark_feature_effects(
            judge_item_deltas_path, feature_dictionary_path, by_judge=True,
        )
        tables["judge-benchmark-feature-effects"] = judge_effects
    table_paths = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        table_paths[name] = path
    figures = {
        **plot_accuracy_comparison(summary, fold_scores, figure_dirs["overview"]),
        "stability": plot_stability(stability, fold_coefficients, figure_dirs["overview"]),
        "feature-effects-overview": plot_feature_overview(
            features, overall_coefficients, figure_dirs["overview"],
        ),
        "group-effects": plot_group_effects(groups, figure_dirs["overview"]),
        **plot_feature_blocks(
            features, overall_coefficients, figure_dirs["by-feature-group"],
        ),
    }
    if benchmark_effects is not None:
        figures["feature-effects-by-benchmark"] = plot_benchmark_feature_grid(
            benchmark_effects, benchmark_coefficients, figure_dirs["by-benchmark"],
        )
    if judge_effects is not None:
        figures.update(plot_judge_benchmark_grids(
            judge_effects, cell_coefficients, figure_dirs["by-judge"],
        ))
    figures["annotation-comparison"] = plot_annotation_comparison(
        groups, figure_dirs["overview"],
    )
    figures["feature-scatter-secondary"] = plot_feature_scatter(
        features, figure_dirs["secondary"],
    )
    return {"tables": table_paths, "figures": figures}
