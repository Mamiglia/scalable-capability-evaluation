"""Build final pairwise-capability interpretation tables and figures."""

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t


VARIANTS = {
    "raw-l1": ("raw-l1", "raw-l1", "raw", "L1"),
    "raw-l2": ("raw-ridge", "raw-l2", "raw", "L2"),
    "pc-l1": ("pc-l1", "pc-l1", "pc", "L1"),
    "pc-l2": ("pc-ridge", "pc-l2", "pc", "L2"),
}
BENCHMARKS = ("aime", "gpqa", "intercode", "swe_bench_mini")
BENCHMARK_LABELS = {
    "aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode",
    "swe_bench_mini": "SWE",
}
JUDGE_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v4-flash-0731": "DeepSeek V4",
    "gemma-4-31b-it": "Gemma 4",
    "gpt-oss-120b": "GPT-OSS",
    "o4-mini": "o4-mini",
}
JUDGE_COLORS = dict(zip(JUDGE_LABELS, (
    "#4a3aa7", "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4",
)))
BLOCK_COLORS = {
    "surface": "#2a78d6", "structure": "#eb6834", "rubric": "#1baf7a",
    "exposition": "#eda100", "derivation": "#e87ba4",
    "intermediate-errors": "#4a3aa7", "clause-descriptors": "#6595bd",
    "clause-motifs": "#c67a55", "clause-plan": "#4f9982", "pc": "#777777",
    "mixed": "#777777", "unknown": "#777777",
}
ECI_MODELS = {
    "deepseek-r1-0528": "DeepSeek-R1 (May 2025)",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash 0731",
    "gemma-4-31b-it": "Gemma 4 31B IT",
    "gpt-oss-120b": "gpt-oss-120b",
    "o4-mini": "o4-mini",
}
DETERMINISTIC_EXCLUDED_BLOCKS = frozenset({"rubric", "intermediate-errors", "annotations"})


def backproject_pc_coefficients(
    coefficients: pd.DataFrame, loadings: pd.DataFrame, score_scales: pd.Series,
) -> pd.DataFrame:
    """Map unit-variance PC coefficients back into composite coordinates."""
    required = {"feature", "effective"}
    if missing := required - set(coefficients):
        raise ValueError(f"PC coefficients missing columns: {sorted(missing)}")
    if missing := {"component", "feature", "loading"} - set(loadings):
        raise ValueError(f"PC loadings missing columns: {sorted(missing)}")
    components = coefficients.feature.astype(str).drop_duplicates().tolist()
    matrix = loadings.pivot(index="component", columns="feature", values="loading").reindex(
        index=components,
    )
    dimensions = [column for column in coefficients if column not in required]
    values = coefficients.pivot(index=dimensions, columns="feature", values="effective")
    values = values.reindex(columns=matrix.index)
    scales = score_scales.reindex(components).to_numpy(float)
    if (values.isna().any().any() or matrix.isna().any().any()
            or not np.isfinite(matrix.to_numpy(float)).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("PC coefficients, loadings, and score scales must be finite and complete")
    projected = pd.DataFrame(
        (values.to_numpy(float) / scales) @ matrix.to_numpy(float),
        index=values.index,
        columns=matrix.columns,
    )
    return (projected.rename_axis(columns="feature").stack().rename("effective")
            .reset_index())


def expand_leaf_coefficients(
    coefficients: pd.DataFrame,
    composites: list[dict[str, object]],
    source_blocks: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Expand composite coefficients through benchmark-specific leaf weights."""
    required = {"benchmark", "feature", "effective"}
    if missing := required - set(coefficients):
        raise ValueError(f"composite coefficients missing columns: {sorted(missing)}")
    records = {str(record["output_name"]): record for record in composites}
    dimensions = [column for column in coefficients if column not in {"feature", "effective"}]
    values: defaultdict[tuple[object, ...], float] = defaultdict(float)
    for row in coefficients.itertuples(index=False):
        feature = str(row.feature)
        if feature not in records:
            raise ValueError(f"composite provenance missing feature: {feature}")
        record = records[feature]
        weights = record["leaf_weights_by_benchmark"].get(str(row.benchmark), {})
        base = tuple(getattr(row, column) for column in dimensions)
        for leaf, weight in weights.items():
            values[(*base, str(leaf))] += float(row.effective) * float(weight)
    return pd.DataFrame([
        {
            **dict(zip([*dimensions, "feature"], key)),
            "effective": value,
            "block": (source_blocks or {}).get(str(key[-1]), "unknown"),
        }
        for key, value in values.items()
    ])


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
        "axes.linewidth": 0.7, "lines.linewidth": 1,
        "scatter.edgecolors": "none", "lines.markeredgewidth": 0,
    })


def _ordinary(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _save(fig, directory: Path, name: str, data: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data.to_csv(directory / f"{name}.csv", index=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    if any("axes sizes collapsed to zero" in str(item.message) for item in caught):
        plt.close(fig)
        raise RuntimeError(f"constrained layout collapsed: {directory / name}")
    fig.set_layout_engine(None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(directory / f"{name}.png", dpi=300)
        fig.savefig(directory / f"{name}.svg")
    plt.close(fig)
    if any("axes sizes collapsed to zero" in str(item.message) for item in caught):
        raise RuntimeError(f"constrained layout collapsed: {directory / name}")
    if "<text" not in (directory / f"{name}.svg").read_text():
        raise RuntimeError(f"editable SVG text missing: {directory / f'{name}.svg'}")


def _fold_dimensions(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    parts = output.fold.astype(str).str.rsplit("__item-", n=1, expand=True)
    if parts.shape[1] != 2 or parts.isna().any().any():
        raise ValueError("folds must use MODEL_FOLD__item-N")
    output["model_fold"], output["item_fold"] = parts[0], parts[1]
    return output


def _t_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan, np.nan
    margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return float(values.mean() - margin), float(values.mean() + margin)


def _fold_summary(
    rows: pd.DataFrame, keys: list[str], value: str, weight: str | None = None,
) -> pd.DataFrame:
    per_fold_keys = [*keys, "fold"]
    if weight:
        grouped = rows.assign(_weighted=rows[value] * rows[weight]).groupby(
            per_fold_keys, as_index=False, sort=False,
        ).agg(_weighted=("_weighted", "sum"), _weight=(weight, "sum"))
        grouped[value] = grouped._weighted / grouped._weight
    else:
        grouped = rows.groupby(per_fold_keys, as_index=False, sort=False)[value].mean()
    return (grouped.groupby(keys, as_index=False, sort=False)[value]
            .agg(estimate="mean", lower="min", upper="max", fold_count="size"))


def _benchmark_lesions(item_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = _fold_dimensions(item_deltas)
    rows = rows[rows.kind.str.endswith(("-feature", "-block", "-family"))].copy()
    summaries = []
    keys = ["bench", "kind", "variable", "reference"]
    for values, frame in rows.groupby(keys, sort=False):
        estimate = float(np.average(frame.delta_accuracy, weights=frame.row_count))
        intervals = []
        for dimension in ("model_fold", "item_fold"):
            by_fold = np.array([
                np.average(group.delta_accuracy, weights=group.row_count)
                for _, group in frame.groupby(dimension, sort=False)
            ])
            intervals.append(_t_interval(by_fold))
        summaries.append({
            **dict(zip(keys, values)), "estimate": estimate,
            "lower": min(interval[0] for interval in intervals),
            "upper": max(interval[1] for interval in intervals),
            "model_fold_count": frame.model_fold.nunique(),
            "item_fold_count": frame.item_fold.nunique(),
            "row_count": int(frame.row_count.sum()),
        })
    output = pd.DataFrame(summaries).rename(columns={"bench": "benchmark"})
    output[["estimate", "lower", "upper"]] *= 100
    return output


def _paired_effects(
    lesions: pd.DataFrame, level: str = "feature", extra: tuple[str, ...] = (),
) -> pd.DataFrame:
    singular_kind, unique_kind = f"singular-{level}", f"lesion-{level}"
    columns = [*extra, "variable", "estimate", "lower", "upper"]
    singular = lesions[lesions.kind == singular_kind][columns].rename(columns={
        "variable": level, "estimate": "singular", "lower": "singular_lower",
        "upper": "singular_upper",
    })
    unique = lesions[lesions.kind == unique_kind][columns].rename(columns={
        "variable": level, "estimate": "unique", "lower": "unique_lower",
        "upper": "unique_upper",
    })
    return singular.merge(
        unique, on=[*extra, level], how="inner", validate="one_to_one",
    )


def _feature_labels(composites: list[dict[str, object]]) -> dict[str, str]:
    labels = {}
    for record in composites:
        feature = str(record["output_name"])
        leaves = ["minimal non-answer prose" if leaf == "bare_answer"
                  else str(leaf).replace("_", " ") for leaf in record["leaf_members"]]
        labels[feature] = leaves[0] if len(leaves) == 1 else " / ".join(leaves[:2]) + (
            f" (+{len(leaves) - 2})" if len(leaves) > 2 else ""
        )
    return labels


def _load_capability_coefficients(root: Path, fit_name: str, variant: str) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "fits" / fit_name).glob("model-fold-*__item-*/coefficients.csv")):
        frames.append(pd.read_csv(path).assign(fold=path.parent.name, variant=variant))
    if len(frames) != 16:
        raise ValueError(f"expected 16 capability coefficient folds for {variant}, found {len(frames)}")
    return pd.concat(frames, ignore_index=True)


def _load_capability_selections(root: Path, fit_name: str, variant: str) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "fits" / fit_name).glob("model-fold-*__item-*/selection.csv")):
        frames.append(pd.read_csv(path).assign(fold=path.parent.name, variant=variant))
    if len(frames) != 16:
        raise ValueError(f"expected 16 capability selection folds for {variant}, found {len(frames)}")
    return pd.concat(frames, ignore_index=True)


def _coefficient_stats(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left, right = np.asarray(left, float), np.asarray(right, float)
    left_positive, right_positive = left >= 0, right >= 0
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
        "same_sign_share": float(np.mean(left_positive == right_positive)),
        "positive_positive": int(np.sum(left_positive & right_positive)),
        "positive_negative": int(np.sum(left_positive & ~right_positive)),
        "negative_positive": int(np.sum(~left_positive & right_positive)),
        "negative_negative": int(np.sum(~left_positive & ~right_positive)),
        "count": len(left),
    }


def _quadrant_text(row: object) -> str:
    counts = np.array([
        row.positive_positive, row.positive_negative,
        row.negative_positive, row.negative_negative,
    ], dtype=float)
    shares = counts / counts.sum()
    return (f"r={row.pearson:.2f} · same={row.same_sign_share:.0%}\n"
            f"++ {shares[0]:.0%}  +− {shares[1]:.0%}\n"
            f"−+ {shares[2]:.0%}  −− {shares[3]:.0%}")


def summarize_capability_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize saved held-out capability probabilities overall and by benchmark."""
    required = {"variant", "bench", "observed", "probability"}
    if missing := required - set(predictions):
        raise ValueError(f"capability predictions missing columns: {sorted(missing)}")
    rows = predictions.copy()
    probability = rows.probability.to_numpy(float)
    observed = rows.observed.to_numpy()
    if (not np.isfinite(probability).all() or not np.isin(observed, (0, 1)).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError(
            "capability predictions must contain finite probabilities and binary outcomes"
        )
    observed = observed.astype(int)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    rows["accuracy"] = ((probability >= 0.5) == observed).astype(float)
    rows["log_loss"] = -(observed * np.log(clipped) + (1 - observed) * np.log1p(-clipped))

    def aggregate(by: list[str]) -> pd.DataFrame:
        return rows.groupby(by, as_index=False).agg(
            rows=("observed", "size"), accuracy=("accuracy", "mean"),
            log_loss=("log_loss", "mean"),
        )

    return aggregate(["variant"]), aggregate(["variant", "bench"])


def pc_score_scales(explained_variance: pd.DataFrame) -> pd.Series:
    """Return the saved standard deviation of each frozen global PC score."""
    required = {"component", "explained_variance"}
    if missing := required - set(explained_variance):
        raise ValueError(f"PC variance table missing columns: {sorted(missing)}")
    if explained_variance.component.duplicated().any():
        raise ValueError("PC variance table contains duplicate components")
    scales = explained_variance.set_index("component").explained_variance.astype(float).pow(0.5)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("PC score scales must be finite and positive")
    return scales


def build_core_tables(
    root: Path, basis_root: Path, provenance_path: Path, ablation_root: Path,
) -> dict[str, pd.DataFrame]:
    """Load pinned artifacts and derive capability importance tables."""
    provenance = json.loads(provenance_path.read_text())
    composites = provenance["final_composites"]
    source_blocks = {str(key): str(value) for key, value in provenance["source_blocks"].items()}
    labels = _feature_labels(composites)
    loadings = pd.read_parquet(basis_root / "loadings.parquet")
    variance = pd.read_csv(basis_root / "explained-variance.csv").head(30)
    variance_scale = pc_score_scales(variance)
    raw_dictionary = pd.read_parquet(
        root / "fits/raw-l1/model-fold-0__item-0/feature-dictionary.parquet"
    )
    blocks = raw_dictionary.set_index("feature").block.astype(str).to_dict()

    capability_coefficients, judge_coefficients, selections = [], [], []
    prediction_tables, fit_diagnostics = [], []
    feature_tables, group_tables, benchmark_tables = [], [], []
    lesion_tables = []
    for variant, (fit_name, judge_name, space, penalty) in VARIANTS.items():
        capability = _load_capability_coefficients(root, fit_name, variant)
        capability["task"] = "capability"
        capability["space"], capability["penalty"] = space, penalty
        capability["standardized_effect"] = capability.effective
        capability_coefficients.append(capability)

        judge = pd.read_csv(root / "canonical-judge-fits" / judge_name / "coefficients.csv")
        judge["variant"], judge["task"] = variant, "judge"
        judge["space"], judge["penalty"] = space, penalty
        judge["standardized_effect"] = judge.effective
        judge_coefficients.append(judge)

        selection = _load_capability_selections(root, fit_name, variant)
        selection["space"], selection["penalty"] = space, penalty
        selections.append(selection)
        prediction_tables.append(_load_capability_predictions(root, fit_name, variant))
        diagnostic_paths = sorted(
            (root / "fits" / fit_name).glob("model-fold-*__item-*/diagnostics.json")
        )
        if len(diagnostic_paths) != 16:
            raise ValueError(f"expected 16 capability diagnostics for {variant}")
        fit_diagnostics.extend({
            "variant": variant, "fold": path.parent.name,
            **json.loads(path.read_text()),
        } for path in diagnostic_paths)

        lesion_dir = ablation_root / fit_name
        lesions = pd.read_csv(lesion_dir / "lesions.csv")
        if lesions.fold_count.nunique() != 1 or lesions.fold_count.iloc[0] != 16:
            raise ValueError(f"{variant} capability lesions do not contain exactly 16 folds")
        lesions[["estimate", "lower", "upper", "delta_accuracy"]] *= 100
        lesions["variant"], lesions["space"], lesions["penalty"] = variant, space, penalty
        lesion_tables.append(lesions)
        features = _paired_effects(lesions, "feature")
        features["variant"], features["space"], features["penalty"] = variant, space, penalty
        features["block"] = features.feature.map(blocks) if space == "raw" else "pc"
        features["label"] = features.feature.map(labels).fillna(
            features.feature.str.replace("_", " ")
        )
        feature_tables.append(features)
        if space == "raw":
            groups = pd.concat([
                _paired_effects(lesions, level).rename(columns={level: "group"}).assign(level=level)
                for level in ("block", "family")
            ], ignore_index=True)
            groups["variant"], groups["penalty"] = variant, penalty
            group_tables.append(groups)
        benchmark = _benchmark_lesions(pd.read_parquet(
            lesion_dir / "item-deltas.parquet"
        ))
        benchmark = _paired_effects(benchmark, "feature", ("benchmark",))
        benchmark["variant"], benchmark["space"], benchmark["penalty"] = (
            variant, space, penalty,
        )
        benchmark["block"] = benchmark.feature.map(blocks) if space == "raw" else "pc"
        benchmark["label"] = benchmark.feature.map(labels).fillna(
            benchmark.feature.str.replace("_", " ")
        )
        benchmark_tables.append(benchmark)

    capability_coefficients = pd.concat(capability_coefficients, ignore_index=True)
    judge_coefficients = pd.concat(judge_coefficients, ignore_index=True)
    selections = pd.concat(selections, ignore_index=True)
    variant_metrics, benchmark_metrics = summarize_capability_predictions(
        pd.concat(prediction_tables, ignore_index=True),
    )
    coefficient_overall, coefficient_benchmark, stability = [], [], []
    for variant, frame in capability_coefficients.groupby("variant", sort=False):
        coefficient_benchmark.append(_fold_summary(
            frame, ["variant", "feature", "benchmark"], "standardized_effect",
        ))
        within_fold = frame.groupby(
            ["variant", "feature", "fold"], as_index=False,
        ).standardized_effect.mean()
        coefficient_overall.append(_fold_summary(
            within_fold, ["variant", "feature"], "standardized_effect",
        ))
        all_features = sorted(frame.feature.astype(str).unique())
        selected = selections[selections.variant == variant]
        selected_counts = selected.groupby("feature").fold.nunique()
        stability.extend({
            "variant": variant, "feature": feature,
            "selected_folds": int(selected_counts.get(feature, 0)),
            "total_folds": 16,
            "selection_share": float(selected_counts.get(feature, 0) / 16),
        } for feature in all_features)

    source_rows = []
    for record in composites:
        for leaf in record["leaf_members"]:
            for benchmark in BENCHMARKS:
                source_rows.append({
                    "composite": record["output_name"], "block": record["block"],
                    "source_blocks": json.dumps(record["source_blocks"]),
                    "leaf": leaf,
                    "leaf_label": "minimal non-answer prose" if leaf == "bare_answer"
                    else str(leaf).replace("_", " "),
                    "global_weight": record["leaf_weights"].get(leaf, 0.0),
                    "benchmark": benchmark,
                    "benchmark_weight": record["leaf_weights_by_benchmark"].get(
                        benchmark, {},
                    ).get(leaf, 0.0),
                })
    loading_table = loadings.merge(
        raw_dictionary[["feature", "block"]], on="feature", how="left",
        validate="many_to_one",
    )
    loading_table["label"] = loading_table.feature.map(labels).fillna(
        loading_table.feature.str.replace("_", " ")
    )
    block_mass = (loading_table.assign(mass=loading_table.loading.pow(2))
                  .groupby(["component", "block"], as_index=False).mass.sum())
    top_loadings = pd.concat([
        pd.concat([
            frame.nlargest(10, "loading").assign(direction="positive", rank=range(1, 11)),
            frame.nsmallest(10, "loading").assign(direction="negative", rank=range(1, 11)),
        ])
        for _, frame in loading_table.groupby("component", sort=False)
    ], ignore_index=True)
    return {
        "capability_coefficients": capability_coefficients,
        "judge_coefficients": judge_coefficients,
        "selections": selections,
        "lesions": pd.concat(lesion_tables, ignore_index=True),
        "feature_effects": pd.concat(feature_tables, ignore_index=True),
        "group_effects": pd.concat(group_tables, ignore_index=True),
        "benchmark_effects": pd.concat(benchmark_tables, ignore_index=True),
        "coefficient_overall": pd.concat(coefficient_overall, ignore_index=True),
        "coefficient_benchmark": pd.concat(coefficient_benchmark, ignore_index=True),
        "selection_stability": pd.DataFrame(stability),
        "variant_metrics": variant_metrics,
        "benchmark_metrics": benchmark_metrics,
        "fit_diagnostics": pd.DataFrame(fit_diagnostics),
        "composite_sources": pd.DataFrame(source_rows),
        "loadings": loading_table,
        "explained_variance": variance,
        "pc_block_mass": block_mass,
        "pc_top_loadings": top_loadings,
    }


def refit_matches_frozen(values: dict[str, object], frozen: dict[str, object]) -> bool:
    """Return whether a refit diagnostic preserves every frozen hyperparameter."""
    return all(key in values and values[key] == expected for key, expected in frozen.items())


def validate_fold_manifest(fold: Path, expected_sha256: str) -> dict[str, object]:
    """Validate one completed lesion fold against its summary predecessor hash."""
    manifest_path = fold / "manifest.json"
    if (not manifest_path.is_file()
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_sha256):
        raise RuntimeError(f"lesion fold manifest hash mismatch: {fold}")
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.get("artifacts")
    files = {
        str(path.relative_to(fold))
        for path in fold.rglob("*") if path.is_file() and path != manifest_path
    }
    if (manifest.get("state") != "completed" or not isinstance(hashes, dict)
            or set(hashes) != files
            or any(hashlib.sha256((fold / name).read_bytes()).hexdigest() != digest
                   for name, digest in hashes.items())):
        raise RuntimeError(f"invalid completed fold artifacts: {fold}")
    return manifest


def validate_ablation_download(ablation_root: Path) -> pd.DataFrame:
    rows = []
    for variant, (fit_name, _, space, _) in VARIANTS.items():
        directory = ablation_root / fit_name
        summary = json.loads((directory / "summary-manifest.json").read_text())
        if summary.get("state") != "completed" or len(summary["settings"]["fold_manifests"]) != 16:
            raise RuntimeError(f"invalid 16-fold summary manifest: {variant}")
        for name, digest in summary["artifact_sha256"].items():
            path = directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"summary artifact hash mismatch: {path}")
        folds = sorted((directory / "folds").glob("model-fold-*__item-*"))
        if len(folds) != 16:
            raise RuntimeError(f"expected 16 downloaded folds for {variant}, found {len(folds)}")
        expected_fold_hashes = summary["settings"]["fold_manifests"]
        expected_specs = 250 if space == "raw" else 60
        converged, diagnostic_count = 0, 0
        for fold, expected_fold_hash in zip(folds, expected_fold_hashes, strict=True):
            manifest = validate_fold_manifest(fold, expected_fold_hash)
            if (manifest.get("state") != "completed"
                    or manifest.get("unique_specifications") != expected_specs):
                raise RuntimeError(f"invalid completed fold manifest: {fold}")
            frozen = json.loads((fold / "frozen-hyperparameters.json").read_text())
            if set(frozen) != {"C", "l1_ratio", "benchmark_deviation_penalty"}:
                raise RuntimeError(f"invalid frozen capability hyperparameters: {fold}")
            diagnostics = sorted((fold / "diagnostics").glob("*.json"))
            diagnostic_count += len(diagnostics)
            for path in diagnostics:
                if path.stem in {"full", "null"}:
                    continue
                values = json.loads(path.read_text())
                if not refit_matches_frozen(values, frozen):
                    raise RuntimeError(f"refit hyperparameters changed: {path}")
                converged += int(values.get("converged") is True)
        expected_diagnostics = 16 * (expected_specs + 2)
        if diagnostic_count != expected_diagnostics or converged != 16 * expected_specs:
            raise RuntimeError(
                f"incomplete/nonconverged diagnostics for {variant}: "
                f"{diagnostic_count}/{expected_diagnostics}, {converged}/{16 * expected_specs}"
            )
        lesions = pd.read_csv(directory / "lesions.csv")
        if not lesions.fold_count.eq(16).all():
            raise RuntimeError(f"lesion summary fold count is not 16: {variant}")
        rows.append({
            "variant": variant, "backend": fit_name, "folds": len(folds),
            "labelled_specifications": len(lesions), "unique_refits": expected_specs,
            "diagnostics": diagnostic_count, "converged_refits": converged,
            "summary_manifest_sha256": hashlib.sha256(
                (directory / "summary-manifest.json").read_bytes()
            ).hexdigest(),
        })
    return pd.DataFrame(rows)


def _axis_limit(values: np.ndarray, *, symmetric: bool = True) -> tuple[float, float]:
    values = np.asarray(values, float)
    maximum = max(0.05, float(np.nanmax(np.abs(values))))
    step = 10 ** math.floor(math.log10(maximum))
    limit = math.ceil(maximum / step) * step
    return (-limit, limit) if symmetric else (min(0.0, float(np.nanmin(values))), limit)


def _importance_frame(tables: dict[str, pd.DataFrame], variant: str) -> pd.DataFrame:
    effects = tables["feature_effects"].query("variant == @variant")
    coefficients = tables["coefficient_overall"].query("variant == @variant").rename(columns={
        "estimate": "coefficient", "lower": "coefficient_lower", "upper": "coefficient_upper",
    })
    stability = tables["selection_stability"].query("variant == @variant")
    return (effects.merge(coefficients, on=["variant", "feature"], validate="one_to_one")
            .merge(stability, on=["variant", "feature"], validate="one_to_one"))


def _importance_plot(frame: pd.DataFrame, directory: Path, name: str, *, top: int | None) -> None:
    values = frame.copy()
    values["rank_score"] = values[["singular", "unique", "coefficient"]].abs().max(axis=1)
    if top is not None:
        values = values.nlargest(min(top, len(values)), "rank_score")
    values = values.sort_values("rank_score")
    height = min(3.15, max(1.35, 0.17 * len(values) + 0.65))
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(3.35, height), sharey=True, layout="constrained")
    colors = {"singular": "#2a78d6", "unique": "#eb6834", "coefficient": "#4a3aa7"}
    labels = {"singular": "Gain (pp)", "unique": "Loss (pp)",
              "coefficient": "Std. β"}
    y = np.arange(len(values))
    for index, metric in enumerate(("singular", "unique", "coefficient")):
        ax = axes[index]
        lower, upper = values[f"{metric}_lower"], values[f"{metric}_upper"]
        ax.errorbar(values[metric], y, xerr=np.vstack((values[metric] - lower, upper - values[metric])),
                    fmt="none", color=colors[metric], lw=1, capsize=0)
        ax.scatter(values[metric], y, s=8, color=colors[metric], zorder=3)
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        limits = _axis_limit(np.r_[lower, upper], symmetric=True)
        ax.set_xlim(limits); ax.set_xticks([limits[0], 0, limits[1]])
        ax.set_xlabel(labels[metric])
        ax.text(0, 1.02, metric.capitalize(), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7, color="#666666")
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels([textwrap.fill(value, 20) for value in values.label], fontsize=6)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="y", length=0)
        _ordinary(ax)
    axes[0].set_ylim(-0.5, len(values) - 0.5)
    _save(fig, directory, name, values.drop(columns="rank_score"))


def _group_plot(frame: pd.DataFrame, directory: Path, name: str) -> None:
    values = frame.assign(rank=frame[["singular", "unique"]].abs().max(axis=1)).sort_values("rank")
    height = min(3.1, max(1.5, 0.18 * len(values) + 0.7))
    _style(); fig, ax = plt.subplots(figsize=(2.0, height), layout="constrained")
    y = np.arange(len(values))
    for metric, dodge, color in (("singular", -0.1, "#2a78d6"), ("unique", 0.1, "#eb6834")):
        ax.errorbar(values[metric], y + dodge,
                    xerr=np.vstack((values[metric] - values[f"{metric}_lower"],
                                    values[f"{metric}_upper"] - values[metric])),
                    fmt="none", color=color, lw=1, capsize=0)
        ax.scatter(values[metric], y + dodge, s=8, color=color, zorder=3)
    limits = _axis_limit(values[["singular_lower", "singular_upper",
                                 "unique_lower", "unique_upper"]].to_numpy())
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(limits); ax.set_xticks([limits[0], 0, limits[1]])
    ax.set_yticks(y, [f"{level} · {group}".replace("-", " ")
                      for level, group in zip(values.level, values.group)], fontsize=6)
    ax.set_xlabel("Accuracy change (pp)"); _ordinary(ax)
    fig.legend(handles=[
        Line2D([], [], marker="o", lw=0, color="#2a78d6", label="Singular"),
        Line2D([], [], marker="o", lw=0, color="#eb6834", label="Unique removal"),
    ], loc="outside upper center", ncol=2, frameon=False)
    _save(fig, directory, name, values.drop(columns="rank"))


def _benchmark_plot(
    effects: pd.DataFrame, coefficients: pd.DataFrame, features: list[str],
    directory: Path, name: str,
) -> None:
    values = effects[effects.feature.isin(features)].merge(
        coefficients[coefficients.feature.isin(features)].rename(columns={
            "estimate": "coefficient", "lower": "coefficient_lower",
            "upper": "coefficient_upper",
        }), on=["variant", "feature", "benchmark"], validate="one_to_one",
    )
    order = list(reversed(features))
    _style(); fig, axes = plt.subplots(4, 3, figsize=(3.35, 3.15),
                                       sharex="col", sharey=True, layout="constrained")
    all_columns = {metric: [] for metric in ("singular", "unique", "coefficient")}
    for row, benchmark in enumerate(BENCHMARKS):
        frame = values[values.benchmark == benchmark].set_index("feature").reindex(order)
        for column, metric in enumerate(("singular", "unique", "coefficient")):
            ax = axes[row, column]; y = np.arange(len(order))
            ax.errorbar(frame[metric], y,
                        xerr=np.vstack((frame[metric] - frame[f"{metric}_lower"],
                                        frame[f"{metric}_upper"] - frame[metric])),
                        fmt="none", color=("#2a78d6", "#eb6834", "#4a3aa7")[column],
                        lw=0.8, capsize=0)
            ax.scatter(frame[metric], y, s=5,
                       color=("#2a78d6", "#eb6834", "#4a3aa7")[column])
            ax.axvline(0, color="#999999", lw=0.6, ls="--"); _ordinary(ax)
            ax.set_yticks(y)
            if column == 0:
                ax.set_yticklabels([textwrap.fill(label, 16) for label in frame.label], fontsize=5)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="y", length=0)
            all_columns[metric].extend([*frame[f"{metric}_lower"], *frame[f"{metric}_upper"]])
        axes[row, 0].text(-0.04, 1.02, BENCHMARK_LABELS[benchmark],
                          transform=axes[row, 0].transAxes, ha="right", va="bottom",
                          fontsize=7, color="#666666")
    for column, metric in enumerate(("singular", "unique", "coefficient")):
        limits = _axis_limit(np.array(all_columns[metric]))
        for ax in axes[:, column]:
            ax.set_xlim(limits); ax.set_xticks([limits[0], 0, limits[1]])
        axes[-1, column].set_xlabel(("Gain (pp)", "Loss (pp)", "Std. β")[column])
    axes[0, 0].set_ylim(-0.5, len(order) - 0.5)
    _save(fig, directory, name, values)


def _heatmap(
    frame: pd.DataFrame, metric: str, features: list[str], directory: Path, name: str,
) -> None:
    values = frame[frame.feature.isin(features)].pivot(
        index="feature", columns="benchmark", values=metric,
    ).reindex(index=features, columns=BENCHMARKS)
    limit = max(0.05, float(np.nanmax(np.abs(values.to_numpy(float)))))
    _style(); fig, ax = plt.subplots(figsize=(1.7, min(3.0, max(1.6, 0.17 * len(features) + 0.55))),
                                     layout="constrained")
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), [BENCHMARK_LABELS[value] for value in BENCHMARKS],
                  rotation=90, ha="center")
    labels = frame.drop_duplicates("feature").set_index("feature").label
    ax.set_yticks(range(len(features)), [textwrap.fill(labels[value], 18) for value in features],
                  fontsize=6)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.05, shrink=0.75,
                            ticks=[-limit, 0, limit])
    colorbar.ax.set_xticklabels([f"{-limit:.2g}", "0", f"{limit:.2g}"])
    source = values.rename_axis(index="feature").reset_index().melt(
        id_vars="feature", var_name="benchmark", value_name=metric,
    )
    _save(fig, directory, name, source)


def plot_capability_importance(tables: dict[str, pd.DataFrame], root: Path) -> None:
    for variant in VARIANTS:
        directory = root / "figures/capability" / variant
        frame = _importance_frame(tables, variant)
        _importance_plot(frame, directory, "feature-effects-overall", top=12)
        if variant.startswith("raw"):
            for block, block_rows in frame.groupby("block", sort=False):
                ranked = block_rows.assign(score=block_rows[
                    ["singular", "unique", "coefficient"]
                ].abs().max(axis=1)).sort_values("score", ascending=False)
                chunks = [ranked.iloc[index:index + 12].drop(columns="score")
                          for index in range(0, len(ranked), 12)]
                for index, chunk in enumerate(chunks, 1):
                    suffix = "" if len(chunks) == 1 else f"-{index}"
                    _importance_plot(chunk, directory / "by-block",
                                     f"feature-effects-{block}{suffix}", top=None)
            _group_plot(tables["group_effects"].query("variant == @variant"),
                        directory, "block-family-effects")
        effects = tables["benchmark_effects"].query("variant == @variant")
        coefficients = tables["coefficient_benchmark"].query("variant == @variant")
        rank = frame.assign(score=frame[["singular", "unique", "coefficient"]].abs().max(axis=1))
        features = rank.nlargest(4, "score").feature.tolist()
        _benchmark_plot(effects, coefficients, features, directory,
                        "feature-effects-by-benchmark")
        for metric in ("singular", "unique"):
            _heatmap(effects, metric, features, directory,
                     f"{metric}-by-benchmark")
        coefficient_heat = coefficients.merge(
            frame[["feature", "label"]], on="feature", validate="many_to_one",
        )
        _heatmap(coefficient_heat, "estimate", features, directory,
                 "coefficient-by-benchmark")

    pc_stability = tables["selection_stability"].query("variant in ['pc-l1', 'pc-l2']")
    pc_stability = pc_stability.assign(number=lambda frame: frame.feature.str.extract(r"(\d+)").astype(int))
    _style(); fig, ax = plt.subplots(figsize=(2.1, 1.45), layout="constrained")
    for variant, color, marker in (("pc-l1", "#2a78d6", "o"), ("pc-l2", "#eb6834", "s")):
        frame = pc_stability[pc_stability.variant == variant]
        ax.plot(frame.number, frame.selection_share, marker=marker, ms=2.5, color=color,
                label=VARIANTS[variant][3])
    ax.set_xlim(1, 30); ax.set_xticks([1, 15, 30]); ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1])
    ax.set_xlabel("PC"); ax.set_ylabel("Selected fold share"); _ordinary(ax)
    fig.legend(loc="outside upper center", ncol=2, frameon=False)
    _save(fig, root / "figures/capability/pc-comparison", "selection-stability", pc_stability)

    comparisons = []
    for metric, table_name, column in (
        ("singular", "feature_effects", "singular"),
        ("unique", "feature_effects", "unique"),
        ("coefficient", "coefficient_overall", "estimate"),
    ):
        table = tables[table_name]
        left = table[table.variant == "pc-l1"][["feature", column]].rename(columns={column: "l1"})
        right = table[table.variant == "pc-l2"][["feature", column]].rename(columns={column: "l2"})
        comparisons.append(left.merge(right, on="feature", validate="one_to_one").assign(metric=metric))
    comparison = pd.concat(comparisons, ignore_index=True)
    _style(); fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.8), layout="constrained")
    for ax, metric in zip(axes, ("singular", "unique", "coefficient")):
        frame = comparison[comparison.metric == metric]
        limit = max(0.05, float(np.abs(frame[["l1", "l2"]]).to_numpy().max()))
        ax.scatter(frame.l1, frame.l2, s=6, color="#4a3aa7", alpha=.55)
        ax.plot([-limit, limit], [-limit, limit], color="#999999", lw=.8, ls="--")
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        tick_labels = [f"{-limit:.2g}", f"{limit:.2g}"]
        ax.set_xticks([-limit, limit], tick_labels)
        ax.set_yticks([-limit, limit], tick_labels)
        ax.set_xlabel("L1"); ax.set_ylabel("L2")
        ax.text(0, 1.02, metric.capitalize(), transform=ax.transAxes,
                fontsize=7, color="#666666"); _ordinary(ax)
    _save(fig, root / "figures/capability/pc-comparison", "l1-vs-l2-effects", comparison)


def select_important_pcs(
    tables: dict[str, pd.DataFrame], root: Path, top: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """Apply the fixed union-of-top-five rule declared for the final report."""
    candidates = []
    capability = tables["feature_effects"].query("space == 'pc'")
    for metric in ("singular", "unique"):
        scores = capability.groupby("feature", as_index=False)[metric].mean()
        candidates.append(scores.nlargest(top, metric).assign(source=f"capability-{metric}")
                          .rename(columns={metric: "score"}))
    coefficients = (tables["coefficient_overall"].query("variant in ['pc-l1', 'pc-l2']")
                    .groupby("feature", as_index=False).estimate.apply(lambda values: values.abs().mean()))
    candidates.append(coefficients.nlargest(top, "estimate").assign(source="capability-coefficient")
                      .rename(columns={"estimate": "score"}))
    judge_rows = []
    for variant in ("pc-l1", "pc-l2"):
        rows = pd.read_csv(root / "canonical-judge-ablations" / variant / "lesions.csv")
        rows["variant"] = variant
        judge_rows.append(rows)
    judge = pd.concat(judge_rows, ignore_index=True)
    for kind, name in (("singular-feature", "judge-singular"),
                       ("lesion-feature", "judge-unique")):
        scores = (judge[judge.kind == kind].groupby("variable", as_index=False)
                  .delta_accuracy.mean().rename(columns={"variable": "feature",
                                                         "delta_accuracy": "score"}))
        candidates.append(scores.nlargest(top, "score").assign(source=name))
    selection = pd.concat(candidates, ignore_index=True)
    selection["rank_within_source"] = selection.groupby("source").score.rank(
        ascending=False, method="first",
    ).astype(int)
    important = sorted(selection.feature.unique(), key=lambda value: int(value.split("_")[-1]))
    return important, selection.sort_values(["source", "rank_within_source"])


def _pc_definition_figure(
    pc: str, tables: dict[str, pd.DataFrame], directory: Path,
) -> pd.DataFrame:
    loading = tables["loadings"].query("component == @pc")
    loading = pd.concat([loading.nlargest(3, "loading"), loading.nsmallest(3, "loading")])
    loading = loading.drop_duplicates("feature").sort_values("loading")
    mass = tables["pc_block_mass"].query("component == @pc").sort_values("mass")
    capability = tables["capability_coefficients"].query("feature == @pc")
    judge = tables["judge_coefficients"].query("feature == @pc")
    benchmark = pd.concat([
        capability.groupby(["penalty", "benchmark"], as_index=False).standardized_effect.mean()
        .assign(task="Capability"),
        judge.groupby(["penalty", "benchmark"], as_index=False).standardized_effect.mean()
        .assign(task="Judge mean"),
    ], ignore_index=True)
    cap_overall = capability.groupby("penalty").standardized_effect.mean()
    comparison = (judge.groupby(["penalty", "judge"], as_index=False).standardized_effect.mean()
                  .rename(columns={"standardized_effect": "judge_effect"}))
    comparison["capability_effect"] = comparison.penalty.map(cap_overall)

    _style(); fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), layout="constrained")
    ax = axes[0]; y = np.arange(len(loading))
    colors = [BLOCK_COLORS.get(block, "#777777") for block in loading.block]
    ax.hlines(y, 0, loading.loading, color=colors, lw=1)
    ax.scatter(loading.loading, y, color=colors, s=8)
    limit = max(.05, float(loading.loading.abs().max()))
    ax.set_xlim(-limit, limit)
    ax.set_xticks([-limit, limit], [f"{-limit:.1g}", f"{limit:.1g}"])
    ax.set_yticks(y, [textwrap.shorten(value, width=17, placeholder="…")
                      for value in loading.label], fontsize=5.5)
    ax.set_xlabel("Signed loading"); ax.axvline(0, color="#999999", lw=.8, ls="--"); _ordinary(ax)
    ax.text(0, 1.03, "Loading definition", transform=ax.transAxes, fontsize=7, color="#666666")

    ax = axes[1]; y = np.arange(len(mass))
    ax.barh(y, mass.mass, color=[BLOCK_COLORS.get(block, "#777777") for block in mass.block])
    upper = max(.1, math.ceil(float(mass.mass.max()) * 10) / 10)
    ax.set_xlim(0, upper); ax.set_xticks([0, upper]); ax.set_yticks(y, mass.block.str.replace("-", " "), fontsize=5.5)
    ax.set_xlabel("Mass"); _ordinary(ax)
    ax.text(0, 1.03, "By block", transform=ax.transAxes, fontsize=7, color="#666666")
    loading_source = pd.concat([
        loading.assign(section="loadings"), mass.assign(section="block-mass"),
    ], ignore_index=True, sort=False)
    _save(fig, directory, f"{pc}-loadings", loading_source)

    _style(); fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), layout="constrained")
    ax = axes[0]; x = np.arange(4)
    for task, color in (("Capability", "#222222"), ("Judge mean", "#999999")):
        for penalty, marker, linestyle in (("L1", "o", "-"), ("L2", "s", "--")):
            frame = benchmark[(benchmark.task == task) & (benchmark.penalty == penalty)].set_index(
                "benchmark"
            ).reindex(BENCHMARKS)
            ax.plot(x, frame.standardized_effect, marker=marker, ms=3, color=color,
                    ls=linestyle, label=f"{task} · {penalty}")
    values = benchmark.standardized_effect.to_numpy(float); limits = _axis_limit(values)
    ax.set_ylim(limits); ax.set_yticks([limits[0], 0, limits[1]])
    ax.set_xticks(x, ["A", "G", "IC", "SWE"])
    ax.set_ylabel("Standardized β"); _ordinary(ax)
    ax.text(0, 1.03, "Benchmark effects", transform=ax.transAxes, fontsize=7, color="#666666")

    ax = axes[1]
    for row in comparison.itertuples():
        ax.scatter(row.capability_effect, row.judge_effect, s=10,
                   color=JUDGE_COLORS[row.judge], marker="o" if row.penalty == "L1" else "s")
    limit = max(.05, float(comparison[["capability_effect", "judge_effect"]].abs().max().max()))
    ax.plot([-limit, limit], [-limit, limit], color="#999999", lw=.8, ls="--")
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ticks = [f"{-limit:.2g}", f"{limit:.2g}"]
    ax.set_xticks([-limit, limit], ticks); ax.set_yticks([-limit, limit], ticks)
    ax.set_xlabel("Capability β"); ax.set_ylabel("Judge β"); _ordinary(ax)
    ax.text(0, 1.03, "Task comparison", transform=ax.transAxes, fontsize=7, color="#666666")
    handles = [Line2D([], [], marker="o", lw=0, color=color, label=JUDGE_LABELS[judge])
               for judge, color in JUDGE_COLORS.items()]
    handles += [Line2D([], [], color="#222222", label="Capability"),
                Line2D([], [], color="#999999", label="Judge mean"),
                Line2D([], [], marker="o", lw=0, color="#555555", label="L1"),
                Line2D([], [], marker="s", lw=0, color="#555555", label="L2")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=5.2)
    effects_source = pd.concat([
        benchmark.assign(section="benchmark-effects"), comparison.assign(section="task-comparison"),
    ], ignore_index=True, sort=False)
    _save(fig, directory, f"{pc}-task-effects", effects_source)
    return pd.concat([loading_source, effects_source], ignore_index=True, sort=False)


def plot_pc_definitions(
    tables: dict[str, pd.DataFrame], source_root: Path, output_root: Path,
) -> list[str]:
    important, selection = select_important_pcs(tables, source_root)
    selection.to_csv(output_root / "tables/important-pc-selection.csv", index=False)
    directory = output_root / "figures/pc-definition"
    variance = tables["explained_variance"].head(30)
    _style(); fig, ax = plt.subplots(figsize=(1.75, 1.35), layout="constrained")
    ax.plot(range(1, 31), variance.explained_variance_ratio, color="#2a78d6", marker="o", ms=2,
            label="Per PC")
    ax.plot(range(1, 31), variance.cumulative_explained_variance_ratio, color="#555555", ls="--",
            label="Cumulative")
    ax.set_xlim(1, 30); ax.set_xticks([1, 15, 30]); ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1])
    ax.set_xlabel("PC"); ax.set_ylabel("Variance share"); _ordinary(ax)
    fig.legend(loc="outside upper center", ncol=2, frameon=False)
    _save(fig, directory, "variance-explained", variance)

    mass = tables["pc_block_mass"].pivot(index="component", columns="block", values="mass").fillna(0)
    _style(); fig, ax = plt.subplots(figsize=(3.3, 2.15), layout="constrained")
    image = ax.imshow(mass.T, aspect="auto", cmap="Blues", vmin=0, vmax=float(mass.max().max()))
    ax.set_xticks([0, 14, 29], ["PC1", "PC15", "PC30"])
    ax.set_yticks(range(len(mass.columns)), mass.columns.str.replace("-", " "), fontsize=6)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    maximum = float(mass.max().max())
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=.05, shrink=.6,
                            ticks=[0, maximum])
    colorbar.ax.set_xticklabels(["0", f"{maximum:.2g}"])
    _save(fig, directory, "pc-by-source-block", mass.rename_axis(index="component").reset_index())

    loadings = tables["loadings"]
    top_features = (loadings.assign(magnitude=loadings.loading.abs())
                    .groupby(["feature", "label"], as_index=False).magnitude.max()
                    .nlargest(12, "magnitude").feature.tolist())
    heat = loadings[loadings.feature.isin(top_features)].pivot(
        index="feature", columns="component", values="loading",
    ).reindex(index=top_features, columns=variance.component)
    labels = loadings.drop_duplicates("feature").set_index("feature").label
    limit = float(heat.abs().max().max())
    _style(); fig, ax = plt.subplots(figsize=(3.3, 2.1), layout="constrained")
    image = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks([0, 14, 29], ["PC1", "PC15", "PC30"])
    ax.set_yticks(range(len(heat)), [textwrap.fill(labels[value], 18) for value in heat.index], fontsize=5.5)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=.05, shrink=.6,
                            ticks=[-limit, 0, limit])
    colorbar.ax.set_xticklabels([f"{-limit:.2g}", "0", f"{limit:.2g}"])
    _save(fig, directory, "pc-by-top-feature", heat.rename_axis(index="feature").reset_index())

    for pc in important:
        _pc_definition_figure(pc, tables, directory / "selected")
    descriptions = []
    for pc in variance.component:
        frame = loadings[loadings.component == pc]
        positive = frame.nlargest(2, "loading").label.tolist()
        negative = frame.nsmallest(2, "loading").label.tolist()
        descriptions.append({
            "component": pc,
            "description": f"Positive: {', '.join(positive)}; negative: {', '.join(negative)}",
            "important": pc in important,
        })
    pd.DataFrame(descriptions).to_csv(
        output_root / "tables/pc-semantic-descriptions.csv", index=False,
    )
    return important


def build_backprojection_tables(
    tables: dict[str, pd.DataFrame], provenance_path: Path,
) -> dict[str, pd.DataFrame]:
    provenance = json.loads(provenance_path.read_text())
    composites = provenance["final_composites"]
    source_blocks = provenance["source_blocks"]
    loadings = tables["loadings"][["component", "feature", "loading"]]
    variance_scale = pc_score_scales(tables["explained_variance"])
    leaves = []
    for task, coefficients in (("capability", tables["capability_coefficients"]),
                               ("judge", tables["judge_coefficients"])):
        dimensions = ["variant", "fold", "benchmark"]
        if task == "judge":
            dimensions.append("judge")
        for variant, (_, _, space, penalty) in VARIANTS.items():
            frame = coefficients[coefficients.variant == variant][
                [*dimensions, "feature", "effective"]
            ].copy()
            if task == "capability":
                frame["judge"] = "all"
            composite = (frame if space == "raw" else
                         backproject_pc_coefficients(frame, loadings, variance_scale))
            expanded = expand_leaf_coefficients(composite, composites, source_blocks)
            expanded["task"], expanded["variant"] = task, variant
            expanded["space"], expanded["penalty"] = space, penalty
            leaves.append(expanded)
    leaves = pd.concat(leaves, ignore_index=True)
    summary = _fold_summary(
        leaves, ["task", "variant", "space", "penalty", "judge", "benchmark",
                 "feature", "block"], "effective",
    )

    points, statistics, ranked = [], [], []
    for task in ("capability", "judge"):
        task_rows = leaves[leaves.task == task]
        for penalty in ("L1", "L2"):
            raw = task_rows[(task_rows.penalty == penalty) & (task_rows.space == "raw")]
            pc = task_rows[(task_rows.penalty == penalty) & (task_rows.space == "pc")]
            keys = ["fold", "judge", "benchmark", "feature", "block"]
            joined = raw[keys + ["effective"]].rename(columns={"effective": "raw"}).merge(
                pc[keys + ["effective"]].rename(columns={"effective": "pc"}),
                on=keys, how="inner", validate="one_to_one",
            ).assign(task=task, penalty=penalty)
            points.append(joined)
            for (judge, benchmark), frame in pd.concat([
                joined,
                joined.assign(benchmark="all"),
            ], ignore_index=True).groupby(["judge", "benchmark"], sort=False):
                statistics.append({
                    "task": task, "penalty": penalty, "judge": judge,
                    "benchmark": benchmark, **_coefficient_stats(frame.raw, frame.pc),
                })
            means = joined.groupby(["judge", "feature", "block"], as_index=False)[["raw", "pc"]].mean()
            means["agreement"] = means.raw * means.pc
            for judge, frame in means.groupby("judge", sort=False):
                ranked.extend(frame.nlargest(10, "agreement").assign(
                    task=task, penalty=penalty, direction="agree",
                    rank=range(1, min(10, len(frame)) + 1),
                ).to_dict("records"))
                ranked.extend(frame.nsmallest(10, "agreement").assign(
                    task=task, penalty=penalty, direction="oppose",
                    rank=range(1, min(10, len(frame)) + 1),
                ).to_dict("records"))
    points = pd.concat(points, ignore_index=True)

    mass_rows = leaves.assign(mass=leaves.effective.abs()).groupby(
        ["task", "variant", "space", "penalty", "fold", "judge", "benchmark", "block"],
        as_index=False,
    ).mass.sum()
    totals = mass_rows.groupby(
        ["task", "variant", "fold", "judge", "benchmark"],
    ).mass.transform("sum")
    mass_rows["normalized_mass"] = mass_rows.mass / totals
    mass_summary = mass_rows.groupby(
        ["task", "variant", "space", "penalty", "judge", "block"],
        as_index=False,
    ).normalized_mass.mean()
    return {
        "leaf_coefficients": leaves,
        "leaf_coefficient_summary": summary,
        "raw_pc_points": points,
        "raw_pc_statistics": pd.DataFrame(statistics),
        "raw_pc_top_agreement": pd.DataFrame(ranked),
        "block_coefficient_mass": mass_summary,
    }


def _scatter_grid(
    points: pd.DataFrame, statistics: pd.DataFrame, task: str, penalty: str,
    directory: Path, name: str,
) -> None:
    judges = ["all"] if task == "capability" else list(JUDGE_LABELS)
    rows, columns = math.ceil(len(judges) / 2), min(2, len(judges))
    _style(); fig, axes = plt.subplots(rows, columns, figsize=(3.5, .92 * rows + .44),
                                       squeeze=False, layout="constrained")
    for index, (ax, judge) in enumerate(zip(axes.flat, judges)):
        frame = points[(points.task == task) & (points.penalty == penalty) & (points.judge == judge)]
        stats = statistics[(statistics.task == task) & (statistics.penalty == penalty)
                           & (statistics.judge == judge) & (statistics.benchmark == "all")].iloc[0]
        limit = max(.05, float(frame[["raw", "pc"]].abs().max().max()))
        ax.scatter(frame.raw, frame.pc, s=2, color="#4a3aa7", alpha=.12)
        ax.plot([-limit, limit], [-limit, limit], color="#999999", lw=.8, ls="--")
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        ax.set_xticks([-limit, 0, limit]); ax.set_yticks([-limit, 0, limit])
        ax.set_xlabel("Raw β" if index // columns == rows - 1 else "")
        ax.set_ylabel("PC-projected β" if index % columns == 0 else "")
        _ordinary(ax)
        label = "Capability" if judge == "all" else JUDGE_LABELS[judge]
        ax.text(0, 1.02, label, transform=ax.transAxes, fontsize=7, color="#666666")
        ax.text(.03, .97, _quadrant_text(stats), transform=ax.transAxes,
                ha="left", va="top", fontsize=5, color="#555555")
    for ax in axes.flat[len(judges):]:
        ax.remove()
    source = points[(points.task == task) & (points.penalty == penalty)].merge(
        statistics[(statistics.task == task) & (statistics.penalty == penalty)
                   & (statistics.benchmark == "all")],
        on=["task", "penalty", "judge"], how="left", validate="many_to_one",
    )
    _save(fig, directory, name, source)


def plot_backprojections(back: dict[str, pd.DataFrame], root: Path) -> None:
    directory = root / "figures/pc-definition/back-projection"
    for task in ("capability", "judge"):
        for penalty in ("L1", "L2"):
            _scatter_grid(back["raw_pc_points"], back["raw_pc_statistics"], task, penalty,
                          directory, f"{task}-raw-vs-pc-{penalty.lower()}")
    mass = back["block_coefficient_mass"]
    capability = mass[(mass.task == "capability") & (mass.judge == "all")]
    order = capability.groupby("block").normalized_mass.mean().sort_values().index.tolist()
    _style(); fig, ax = plt.subplots(figsize=(2.35, 2.1), layout="constrained")
    y = np.arange(len(order)); dodges = np.linspace(-.24, .24, 4)
    for dodge, (variant, (_, _, _, penalty)) in zip(dodges, VARIANTS.items()):
        frame = capability[capability.variant == variant].set_index("block").reindex(order)
        ax.scatter(frame.normalized_mass, y + dodge, s=8,
                   color="#2a78d6" if penalty == "L1" else "#eb6834",
                   marker="o" if variant.startswith("raw") else "s")
    ax.set_xlim(0, max(.1, float(capability.normalized_mass.max()) * 1.05))
    upper = ax.get_xlim()[1]
    ax.set_xticks([0, upper], ["0", f"{upper:.2g}"])
    ax.set_yticks(y, [value.replace("-", " ") for value in order], fontsize=6)
    ax.set_xlabel("Normalized |β| mass"); _ordinary(ax)
    fig.legend(handles=[
        Line2D([], [], marker="o", lw=0, color="#555555", label="Raw"),
        Line2D([], [], marker="s", lw=0, color="#555555", label="PC projected"),
        Line2D([], [], color="#2a78d6", label="L1"),
        Line2D([], [], color="#eb6834", label="L2"),
    ], loc="outside upper center", ncol=2, frameon=False)
    _save(fig, directory, "capability-block-coefficient-mass", capability)


def _load_capability_predictions(root: Path, fit_name: str, variant: str) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "fits" / fit_name).glob("model-fold-*__item-*/predictions.parquet")):
        frames.append(pd.read_parquet(path).assign(fold=path.parent.name, variant=variant))
    if len(frames) != 16:
        raise ValueError(f"expected 16 capability prediction folds for {variant}")
    return pd.concat(frames, ignore_index=True)


def judge_capability_overview(
    accuracy: pd.DataFrame, statistics: pd.DataFrame,
    cosines: pd.DataFrame, eci: pd.DataFrame,
) -> pd.DataFrame:
    """Join matched predictive accuracy and coefficient-alignment summaries."""
    coefficient = statistics[statistics.benchmark == "all"][
        ["variant", "judge", "pearson", "spearman", "same_sign_share"]
    ]
    cosine = cosines[cosines.benchmark == "all"][
        ["variant", "judge", "estimate", "lower", "upper"]
    ].rename(columns={
        "estimate": "cosine", "lower": "cosine_lower", "upper": "cosine_upper",
    })
    output = (accuracy.merge(
        coefficient, on=["variant", "judge"], how="left", validate="one_to_one",
    ).merge(
        cosine, on=["variant", "judge"], how="left", validate="one_to_one",
    ).merge(
        eci[["judge", "eci", "eci_ci_low", "eci_ci_high"]],
        on="judge", how="left", validate="many_to_one",
    ))
    output.insert(
        output.columns.get_loc("judge_accuracy") + 1, "accuracy_difference",
        output.capability_accuracy - output.judge_accuracy,
    )
    return output.sort_values(["variant", "judge"], kind="stable").reset_index(drop=True)


def build_judge_comparison_tables(
    tables: dict[str, pd.DataFrame], back: dict[str, pd.DataFrame],
    root: Path, eci_path: Path,
) -> dict[str, pd.DataFrame]:
    points, statistics, cosines, contributions = [], [], [], []
    for variant in VARIANTS:
        capability = tables["capability_coefficients"].query("variant == @variant")
        judge = tables["judge_coefficients"].query("variant == @variant")
        keys = ["fold", "benchmark", "feature"]
        joined = judge.merge(
            capability[keys + ["standardized_effect"]].rename(
                columns={"standardized_effect": "capability"}
            ), on=keys, how="inner", validate="many_to_one",
        ).rename(columns={"standardized_effect": "judge_effect"})
        joined = joined[["variant", "fold", "judge", "benchmark", "feature",
                         "capability", "judge_effect"]]
        points.append(joined)
        for (judge_name, benchmark), frame in pd.concat([
            joined, joined.assign(benchmark="all"),
        ], ignore_index=True).groupby(["judge", "benchmark"], sort=False):
            statistics.append({
                "variant": variant, "judge": judge_name, "benchmark": benchmark,
                **_coefficient_stats(frame.capability, frame.judge_effect),
            })
        for (fold, judge_name), frame in joined.groupby(["fold", "judge"], sort=False):
            for benchmark in ("all", *BENCHMARKS):
                values = frame if benchmark == "all" else frame[frame.benchmark == benchmark]
                left, right = values.capability.to_numpy(float), values.judge_effect.to_numpy(float)
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                cosine = float(left @ right / denominator) if denominator else np.nan
                cosines.append({
                    "variant": variant, "fold": fold, "judge": judge_name,
                    "benchmark": benchmark, "cosine": cosine,
                })
                if benchmark == "all" and denominator:
                    driver = values.assign(contribution=left * right / denominator).groupby(
                        "feature", as_index=False,
                    ).contribution.sum()
                    contributions.extend(driver.assign(
                        variant=variant, fold=fold, judge=judge_name,
                    ).to_dict("records"))
    points = pd.concat(points, ignore_index=True)
    cosines = pd.DataFrame(cosines)
    cosine_summary = _fold_summary(
        cosines, ["variant", "judge", "benchmark"], "cosine",
    )
    contribution_summary = (pd.DataFrame(contributions).groupby(
        ["variant", "judge", "feature"], as_index=False,
    ).contribution.mean())

    leaf_points, leaf_statistics, leaf_cosines, leaf_contributions = [], [], [], []
    leaves = back["leaf_coefficients"]
    for variant in VARIANTS:
        capability = leaves[(leaves.task == "capability") & (leaves.variant == variant)]
        judge = leaves[(leaves.task == "judge") & (leaves.variant == variant)]
        keys = ["fold", "benchmark", "feature", "block"]
        joined = judge.merge(
            capability[keys + ["effective"]].rename(columns={"effective": "capability"}),
            on=keys, how="inner", validate="many_to_one",
        ).rename(columns={"effective": "judge_effect"})
        joined["variant"] = variant
        leaf_points.append(joined)
        for (judge_name, benchmark), frame in pd.concat([
            joined, joined.assign(benchmark="all"),
        ], ignore_index=True).groupby(["judge", "benchmark"], sort=False):
            leaf_statistics.append({
                "variant": variant, "judge": judge_name, "benchmark": benchmark,
                **_coefficient_stats(frame.capability, frame.judge_effect),
            })
        for (fold, judge_name), frame in joined.groupby(["fold", "judge"], sort=False):
            for benchmark in ("all", *BENCHMARKS):
                values = frame if benchmark == "all" else frame[frame.benchmark == benchmark]
                left, right = values.capability.to_numpy(float), values.judge_effect.to_numpy(float)
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                leaf_cosines.append({
                    "variant": variant, "fold": fold, "judge": judge_name,
                    "benchmark": benchmark,
                    "cosine": float(left @ right / denominator) if denominator else np.nan,
                })
                if benchmark == "all" and denominator:
                    driver = values.assign(contribution=left * right / denominator).groupby(
                        ["feature", "block"], as_index=False,
                    ).contribution.sum()
                    leaf_contributions.extend(driver.assign(
                        variant=variant, fold=fold, judge=judge_name,
                    ).to_dict("records"))
    leaf_points = pd.concat(leaf_points, ignore_index=True)
    leaf_cosines = pd.DataFrame(leaf_cosines)
    leaf_contributions = (pd.DataFrame(leaf_contributions).groupby(
        ["variant", "judge", "feature", "block"], as_index=False,
    ).contribution.mean())

    accuracy_rows = []
    for variant in VARIANTS:
        fit_name, judge_name, _, _ = VARIANTS[variant]
        capability = _load_capability_predictions(root, fit_name, variant)
        judge = pd.read_parquet(root / "canonical-judge-fits" / judge_name / "predictions.parquet")
        keys = ["fold", "bench", "item", "model_left", "model_right", "log_left", "log_right"]
        joined = judge.merge(
            capability[keys + ["observed", "probability"]].rename(columns={
                "observed": "left_stronger", "probability": "capability_probability",
            }), on=keys, how="inner", validate="many_to_one",
        )
        judge_target = joined.left_stronger_x.gt(0).astype(int)
        joined["legacy_target_agrees"] = (
            joined.left_stronger_x.notna() & judge_target.eq(joined.left_stronger_y)
        )
        joined["capability_correct"] = (
            (joined.capability_probability >= .5) == joined.left_stronger_y
        )
        joined["judge_correct"] = joined.observed == joined.left_stronger_y
        accuracy_rows.append(joined.groupby(
            ["fold", "judge", "bench"], as_index=False,
        ).agg(capability_accuracy=("capability_correct", "mean"),
              judge_accuracy=("judge_correct", "mean"),
              legacy_target_agreement=("legacy_target_agrees", "mean"),
              pair_count=("judge_correct", "size"))
          .assign(variant=variant))
    accuracy = pd.concat(accuracy_rows, ignore_index=True)
    accuracy_summary = accuracy.groupby(["variant", "judge"], as_index=False).apply(
        lambda frame: pd.Series({
            "capability_accuracy": np.average(frame.capability_accuracy, weights=frame.pair_count),
            "judge_accuracy": np.average(frame.judge_accuracy, weights=frame.pair_count),
            "pair_count": int(frame.pair_count.sum()),
        }), include_groups=False,
    ).reset_index(drop=True)

    eci = pd.read_csv(eci_path)
    eci = pd.DataFrame([
        {"judge": judge, "eci_model": model,
         **eci.loc[eci.Model == model, ["eci", "eci_ci_low", "eci_ci_high"]].iloc[0].to_dict()}
        for judge, model in ECI_MODELS.items()
        if (eci.Model == model).any()
    ])
    coefficient_statistics = pd.DataFrame(statistics)
    overview = judge_capability_overview(
        accuracy_summary, coefficient_statistics, cosine_summary, eci,
    )
    relations = cosine_summary[cosine_summary.benchmark == "all"].merge(
        accuracy_summary.groupby("judge", as_index=False).judge_accuracy.mean(),
        on="judge", how="left", validate="many_to_one",
    ).merge(eci, on="judge", how="left", validate="many_to_one")
    return {
        "judge_coefficient_points": points,
        "judge_coefficient_statistics": coefficient_statistics,
        "judge_cosines": cosines,
        "judge_cosine_summary": cosine_summary,
        "judge_cosine_contributions": contribution_summary,
        "leaf_judge_points": leaf_points,
        "leaf_judge_statistics": pd.DataFrame(leaf_statistics),
        "leaf_judge_cosines": leaf_cosines,
        "leaf_judge_contributions": leaf_contributions,
        "stronger_model_accuracy": accuracy,
        "stronger_model_accuracy_summary": accuracy_summary,
        "judge_capability_overview": overview,
        "judge_eci": eci,
        "judge_relations": relations,
    }


def _judge_scatter(
    points: pd.DataFrame, statistics: pd.DataFrame, variant: str, directory: Path,
    *, suffix: str = "coefficient",
) -> None:
    _style(); fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.2), layout="constrained")
    for index, (ax, judge) in enumerate(zip(axes.flat, JUDGE_LABELS)):
        frame = points[(points.variant == variant) & (points.judge == judge)]
        stats = statistics[(statistics.variant == variant) & (statistics.judge == judge)
                           & (statistics.benchmark == "all")].iloc[0]
        limit = max(.05, float(frame[["capability", "judge_effect"]].abs().max().max()))
        ax.scatter(frame.capability, frame.judge_effect, s=2, color=JUDGE_COLORS[judge], alpha=.12)
        ax.plot([-limit, limit], [-limit, limit], color="#999999", lw=.8, ls="--")
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
        tick_labels = [f"{-limit:.2g}", f"{limit:.2g}"]
        ax.set_xticks([-limit, limit], tick_labels); ax.set_yticks([-limit, limit], tick_labels)
        ax.set_xlabel("Capability β" if index // 2 == 2 else "")
        ax.set_ylabel("Judge β" if index % 2 == 0 else "")
        _ordinary(ax)
        ax.text(0, 1.02, JUDGE_LABELS[judge], transform=ax.transAxes, fontsize=7, color="#666666")
        ax.text(.03, .97, _quadrant_text(stats), transform=ax.transAxes,
                ha="left", va="top", fontsize=5, color="#555555")
    source = points[points.variant == variant].merge(
        statistics[(statistics.variant == variant) & (statistics.benchmark == "all")],
        on=["variant", "judge"], how="left", validate="many_to_one",
    )
    _save(fig, directory, f"{variant}-{suffix}-scatter", source)


def plot_judge_comparisons(comparison: dict[str, pd.DataFrame], root: Path) -> None:
    directory = root / "figures/judge-comparison"
    for variant in VARIANTS:
        _judge_scatter(comparison["judge_coefficient_points"],
                       comparison["judge_coefficient_statistics"], variant, directory)
        _judge_scatter(comparison["leaf_judge_points"],
                       comparison["leaf_judge_statistics"], variant, directory,
                       suffix="original-leaf")
        frame = comparison["judge_cosine_summary"].query("variant == @variant")
        _style(); fig, ax = plt.subplots(figsize=(2.45, 2.05), layout="constrained")
        y = np.arange(len(JUDGE_LABELS)); dodges = np.linspace(-.24, .24, 5)
        for dodge, benchmark, color in zip(dodges, ("all", *BENCHMARKS),
                                            ("#333333", "#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
            values = frame[frame.benchmark == benchmark].set_index("judge").reindex(JUDGE_LABELS)
            ax.errorbar(values.estimate, y + dodge,
                        xerr=np.vstack((values.estimate - values.lower, values.upper - values.estimate)),
                        fmt="none", color=color, lw=.8, capsize=0)
            ax.scatter(values.estimate, y + dodge, s=7, color=color,
                       label="Overall" if benchmark == "all" else BENCHMARK_LABELS[benchmark])
        ax.set_xlim(-1, 1); ax.set_xticks([-1, 0, 1])
        ax.set_yticks(y, [JUDGE_LABELS[value] for value in JUDGE_LABELS], fontsize=6)
        ax.set_xlabel("Coefficient cosine"); _ordinary(ax)
        fig.legend(loc="outside upper center", ncol=3, frameon=False, fontsize=6)
        _save(fig, directory, f"{variant}-cosine", frame)

        drivers = comparison["judge_cosine_contributions"].query("variant == @variant")
        drivers = drivers.groupby("feature", as_index=False).contribution.mean()
        drivers = pd.concat([drivers.nlargest(6, "contribution"),
                             drivers.nsmallest(6, "contribution")]).sort_values("contribution")
        _style(); fig, ax = plt.subplots(figsize=(1.8, 3.0), layout="constrained")
        y = np.arange(len(drivers)); colors = np.where(drivers.contribution >= 0, "#1baf7a", "#eb6834")
        ax.barh(y, drivers.contribution, color=colors)
        ax.set_yticks(y, [textwrap.fill(value.replace("_", " "), 18)
                          for value in drivers.feature], fontsize=5.5)
        limit = max(.01, float(drivers.contribution.abs().max()))
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-limit, limit], [f"{-limit:.2g}", f"{limit:.2g}"])
        ax.axvline(0, color="#999999", lw=.8, ls="--")
        ax.set_xlabel("Contribution"); _ordinary(ax)
        _save(fig, directory, f"{variant}-cosine-drivers", drivers)

        leaf_drivers = comparison["leaf_judge_contributions"].query("variant == @variant")
        leaf_drivers = leaf_drivers.groupby(["feature", "block"], as_index=False).contribution.mean()
        leaf_drivers = pd.concat([leaf_drivers.nlargest(6, "contribution"),
                                  leaf_drivers.nsmallest(6, "contribution")]).sort_values("contribution")
        _style(); fig, ax = plt.subplots(figsize=(1.8, 3.0), layout="constrained")
        y = np.arange(len(leaf_drivers)); colors = np.where(
            leaf_drivers.contribution >= 0, "#1baf7a", "#eb6834",
        )
        ax.barh(y, leaf_drivers.contribution, color=colors)
        ax.set_yticks(y, [textwrap.fill(value.replace("_", " "), 18)
                          for value in leaf_drivers.feature], fontsize=5.5)
        limit = max(.01, float(leaf_drivers.contribution.abs().max()))
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-limit, limit], [f"{-limit:.2g}", f"{limit:.2g}"])
        ax.axvline(0, color="#999999", lw=.8, ls="--")
        ax.set_xlabel("Leaf contribution"); _ordinary(ax)
        _save(fig, directory, f"{variant}-original-leaf-cosine-drivers", leaf_drivers)

    accuracy = comparison["stronger_model_accuracy_summary"]
    _style(); fig, axes = plt.subplots(2, 2, figsize=(3.0, 2.6), sharex=True, sharey=True,
                                      layout="constrained")
    for ax, variant in zip(axes.flat, VARIANTS):
        frame = accuracy[accuracy.variant == variant].set_index("judge").reindex(JUDGE_LABELS)
        y = np.arange(len(frame))
        ax.scatter(frame.capability_accuracy, y - .08, color="#2a78d6", s=9, label="Capability GLM")
        ax.scatter(frame.judge_accuracy, y + .08, color="#eb6834", s=9, label="Judge decision")
        ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_yticks(y, [JUDGE_LABELS[value]
                    for value in frame.index], fontsize=7)
        ax.set_xlabel(""); _ordinary(ax)
        ax.text(0, 1.02, f"{VARIANTS[variant][2].upper()} {VARIANTS[variant][3]}",
                transform=ax.transAxes, fontsize=7, color="#666666")
    fig.supxlabel("Stronger-model accuracy", fontsize=8)
    fig.legend(loc="outside upper center", ncol=2, frameon=False)
    _save(fig, directory, "capability-vs-judge-accuracy", accuracy)

    relations = comparison["judge_relations"].dropna(subset=["eci"])
    _style(); fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.5), layout="constrained")
    for ax, x, label in zip(axes, ("eci", "judge_accuracy"),
                            ("Judge ECI", "Judge accuracy")):
        for row in relations.itertuples():
            ax.scatter(getattr(row, x), row.estimate, s=10, color=JUDGE_COLORS[row.judge],
                       marker="o" if row.variant.endswith("l1") else "s")
        xmin, xmax = float(relations[x].min()), float(relations[x].max())
        step = 5.0 if x == "eci" else .05
        lower, upper = math.floor(xmin / step) * step, math.ceil(xmax / step) * step
        ax.set_xlim(lower, upper); ax.set_xticks([lower, upper])
        ax.set_ylim(-1, 1); ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(label); ax.set_ylabel("Coefficient cosine"); _ordinary(ax)
    handles = [Line2D([], [], marker="o", lw=0, color=color, label=JUDGE_LABELS[judge])
               for judge, color in JUDGE_COLORS.items() if judge in set(relations.judge)]
    handles += [Line2D([], [], marker="o", lw=0, color="#555555", label="L1"),
                Line2D([], [], marker="s", lw=0, color="#555555", label="L2")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=6)
    _save(fig, directory, "cosine-vs-judge-capability", relations)


def plot_deterministic_views(
    tables: dict[str, pd.DataFrame], comparison: dict[str, pd.DataFrame], root: Path,
) -> None:
    """Write the requested non-annotation subset views without refitting."""
    dictionary = tables["composite_sources"][["composite", "block"]].drop_duplicates()
    presentation = set(tables["composite_sources"].loc[
        tables["composite_sources"].leaf == "bare_answer", "composite",
    ])
    names = set(dictionary.loc[
        ~dictionary.block.isin(DETERMINISTIC_EXCLUDED_BLOCKS)
        & ~dictionary.composite.isin(presentation), "composite",
    ])
    if len(names) != 90:
        raise ValueError(f"expected 90 final deterministic composites, found {len(names)}")
    output = root / "figures/deterministic-only"
    for variant in ("raw-l1", "raw-l2"):
        frame = _importance_frame(tables, variant)
        frame = frame[frame.feature.isin(names)]
        directory = output / "capability" / variant
        _importance_plot(frame, directory, "feature-effects-overall", top=12)
        for block, block_rows in frame.groupby("block", sort=False):
            ranked = block_rows.assign(score=block_rows[
                ["singular", "unique", "coefficient"]
            ].abs().max(axis=1)).sort_values("score", ascending=False)
            chunks = [ranked.iloc[index:index + 12].drop(columns="score")
                      for index in range(0, len(ranked), 12)]
            for index, chunk in enumerate(chunks, 1):
                suffix = "" if len(chunks) == 1 else f"-{index}"
                _importance_plot(chunk, directory / "by-block",
                                 f"feature-effects-{block}{suffix}", top=None)
        groups = tables["group_effects"].query("variant == @variant")
        groups = groups[
            ~groups.group.isin(DETERMINISTIC_EXCLUDED_BLOCKS)
            & ~groups.group.eq("annotations")
        ]
        _group_plot(groups, directory, "block-family-effects")
        effects = tables["benchmark_effects"].query("variant == @variant and feature in @names")
        coefficients = tables["coefficient_benchmark"].query(
            "variant == @variant and feature in @names"
        )
        selected = frame.assign(score=frame[["singular", "unique", "coefficient"]].abs().max(axis=1))
        features = selected.nlargest(4, "score").feature.tolist()
        _benchmark_plot(effects, coefficients, features, directory, "feature-effects-by-benchmark")

        points = comparison["judge_coefficient_points"].query(
            "variant == @variant and feature in @names"
        )
        stats = []
        cosine_rows = []
        for judge, judge_rows in points.groupby("judge", sort=False):
            for benchmark, values in pd.concat([
                judge_rows, judge_rows.assign(benchmark="all"),
            ], ignore_index=True).groupby("benchmark", sort=False):
                stats.append({
                    "variant": variant, "judge": judge, "benchmark": benchmark,
                    **_coefficient_stats(values.capability, values.judge_effect),
                })
            for fold, values in judge_rows.groupby("fold", sort=False):
                for benchmark in ("all", *BENCHMARKS):
                    frame_values = values if benchmark == "all" else values[values.benchmark == benchmark]
                    left, right = frame_values.capability.to_numpy(float), frame_values.judge_effect.to_numpy(float)
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    cosine_rows.append({
                        "variant": variant, "judge": judge, "fold": fold,
                        "benchmark": benchmark,
                        "cosine": float(left @ right / denominator) if denominator else np.nan,
                    })
        stats = pd.DataFrame(stats)
        _judge_scatter(points, stats, variant, output / "judge-comparison")
        cosine = _fold_summary(pd.DataFrame(cosine_rows),
                               ["variant", "judge", "benchmark"], "cosine")
        cosine.to_csv(root / "tables" / f"deterministic-{variant}-judge-cosines.csv", index=False)
    pd.DataFrame({"feature": sorted(names)}).to_csv(
        root / "tables/deterministic-feature-set.csv", index=False,
    )
    (root / "tables/deterministic-scope.json").write_text(json.dumps({
        "mode": "subset views of the completed all-readable fits; no refitting",
        "feature_count": len(names),
        "excluded_blocks": sorted(DETERMINISTIC_EXCLUDED_BLOCKS),
        "excluded_features": ["bare_answer"],
        "excluded_composites": sorted(presentation),
    }, indent=2, sort_keys=True) + "\n")


def _write_scientific_summary(
    source_root: Path, output_root: Path, tables: dict[str, pd.DataFrame],
    comparison: dict[str, pd.DataFrame],
    important_pcs: list[str],
) -> None:
    metrics = tables["variant_metrics"]
    top_features = (tables["feature_effects"].query("variant == 'raw-l1'")
                    .nlargest(5, "singular")[["label", "singular", "unique"]])
    top_pcs = (tables["feature_effects"].query("variant == 'pc-l2'")
               .nlargest(5, "singular")[["feature", "singular", "unique"]])
    cosines = comparison["judge_cosine_summary"]
    overview = comparison["judge_capability_overview"]
    best = cosines[cosines.benchmark == "all"].sort_values("estimate", ascending=False).iloc[0]
    lines = [
        "# Pairwise capability ablations and judge comparison",
        "",
        "## Predictive accuracy",
        "",
        "Held-out accuracy and log loss are properties of the complete predictive models; "
        "they are not feature attribution measures.",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Judge–capability prediction and representation comparison",
        "",
        "The matched accuracy columns compare prediction of the same held-out stronger-model "
        "target. Correlations, sign agreement, and coefficient cosine instead compare fitted "
        "representations; they are not predictive-performance measures.",
        "",
        overview.to_markdown(index=False),
        "",
        "## Singular contribution",
        "",
        "Singular effects are held-out accuracy gains over the fixed 0.5 capability null. "
        "The leading raw-L1 singular predictors were:",
        "",
        top_features.to_markdown(index=False),
        "",
        "The leading PC-L2 singular predictors were:",
        "",
        top_pcs.to_markdown(index=False),
        "",
        "## Unique/drop-one contribution",
        "",
        "Unique effects are held-out accuracy losses after removing one feature, block, or "
        "family from the complete model while keeping the outer fold's penalty frozen. They "
        "measure conditional predictive necessity and can differ sharply from singular effects.",
        "",
        "## Coefficient direction",
        "",
        "Effective coefficients describe conditional direction, not unique attribution. PC "
        "coefficients are already per unit-variance PC score. Original-feature comparisons "
        "divide by the saved PC standard deviation before back-projecting through the shared "
        "loadings and benchmark-specific composite weights.",
        "",
        "## PC variance and loading definition",
        "",
        f"The predeclared top-five union rule selected {len(important_pcs)} PCs: "
        f"{', '.join(important_pcs)}. Each selected-PC figure retains its signed loading values, "
        "block mass, benchmark effects, and judge-task comparison; semantic descriptions are "
        "only shorthand.",
        "",
        "## Benchmark-specific judge–capability similarity",
        "",
        f"The largest mean coefficient cosine was {best.estimate:.3f} for "
        f"{JUDGE_LABELS.get(best.judge, best.judge)} under {best.variant}. Cosines, raw/PC leaf "
        "consistency, ECI relations, and empirical stronger-model accuracy are descriptive; the "
        "judge count is small.",
        "",
        "## Deterministic-only view",
        "",
        "The deterministic-only deliverable is a 90-composite subset view of the same complete "
        "raw fits: annotation blocks and the composite containing `bare_answer` are excluded. "
        "It is deliberately not a separate refit, so its coefficients remain conditional on the "
        "full feature set and its model-level accuracy is unchanged.",
        "",
        "All figures have PNG/SVG partners and a same-stem source CSV. Intervals for capability "
        "lesions use item clustering overall and conservative crossed outer-fold summaries by "
        "benchmark; coefficient intervals show the observed 16-fold range.",
    ]
    (output_root / "tables/scientific-summary.md").write_text("\n".join(lines) + "\n")


def _write_manifest(root: Path) -> None:
    manifest_path = root / "tables/report-manifest.json"
    paths = sorted(path for folder in (root / "tables", root / "figures")
                   for path in folder.rglob("*") if path.is_file() and path != manifest_path)
    figures = sorted((root / "figures").rglob("*.png"))
    for png in figures:
        if not png.with_suffix(".svg").is_file() or not png.with_suffix(".csv").is_file():
            raise RuntimeError(f"figure lacks PNG/SVG/CSV partners: {png}")
        if "<text" not in png.with_suffix(".svg").read_text():
            raise RuntimeError(f"SVG lacks editable text: {png.with_suffix('.svg')}")
    manifest = {
        "state": "completed",
        "figure_count": len(figures),
        "all_figures_have_png_svg_csv": True,
        "artifact_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--basis-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--eci", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_root
    (output / "tables").mkdir(parents=True, exist_ok=True)
    validate_ablation_download(args.ablations).to_csv(
        output / "tables/ablation-validation.csv", index=False,
    )
    tables = build_core_tables(args.root, args.basis_root, args.provenance, args.ablations)
    for name, table in tables.items():
        table.to_csv(output / "tables" / f"{name.replace('_', '-')}.csv", index=False)
    plot_capability_importance(tables, output)
    important_pcs = plot_pc_definitions(tables, args.root, output)
    back = build_backprojection_tables(tables, args.provenance)
    for name, table in back.items():
        table.to_csv(output / "tables" / f"{name.replace('_', '-')}.csv", index=False)
    plot_backprojections(back, output)
    comparison = build_judge_comparison_tables(tables, back, args.root, args.eci)
    for name, table in comparison.items():
        table.to_csv(output / "tables" / f"{name.replace('_', '-')}.csv", index=False)
    plot_judge_comparisons(comparison, output)
    plot_deterministic_views(tables, comparison, output)
    _write_scientific_summary(args.root, output, tables, comparison, important_pcs)
    _write_manifest(output)


if __name__ == "__main__":
    main()
