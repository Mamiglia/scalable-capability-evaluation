"""Shared strict folds and crossed judge-by-benchmark design matrices."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


_PENALTIES = {
    "judge": "judge_deviation_penalty",
    "benchmark": "benchmark_deviation_penalty",
    "cell": "cell_deviation_penalty",
    "item": "item_intercept_penalty",
}
_MAP_COLUMNS = ["column", "block", "feature", "judge", "benchmark", "item", "multiplier"]


def outer_roles(
    frame: pd.DataFrame,
    model_fold: Iterable[str],
    item_fold: int,
    item_folds: int,
) -> np.ndarray:
    """Return strict train/test/drop roles for one new-model/new-item fold."""
    _require_columns(frame, ("bench", "item", "model_left", "model_right"))
    if item_folds <= 0 or not 0 <= item_fold < item_folds:
        raise ValueError("item_fold must be in range(item_folds)")
    items = frame[["bench", "item"]].drop_duplicates().sort_values(["bench", "item"])
    items["fold"] = items.groupby("bench", sort=False).cumcount() % item_folds
    folds = frame[["bench", "item"]].merge(items, on=["bench", "item"], how="left", validate="many_to_one")["fold"].to_numpy()
    model_out = frame.model_left.isin(model_fold).to_numpy() | frame.model_right.isin(model_fold).to_numpy()
    item_out = folds == item_fold
    return np.select([model_out & item_out, ~model_out & ~item_out], ["test", "train"], default="drop")


@dataclass(frozen=True)
class HierarchyLevels:
    """Sorted categories learned only from an outer fold's training rows."""

    judges: tuple[str, ...]
    benchmarks: tuple[str, ...]
    cells: tuple[tuple[str, str], ...]
    items: tuple[tuple[str, str], ...]

    @classmethod
    def from_training(cls, frame: pd.DataFrame) -> "HierarchyLevels":
        _require_columns(frame, ("judge", "bench", "item"))
        judges = tuple(sorted(frame.judge.drop_duplicates()))
        benchmarks = tuple(sorted(frame.bench.drop_duplicates()))
        cells = tuple((judge, benchmark) for judge in judges for benchmark in benchmarks)
        items = tuple(map(tuple, frame[["bench", "item"]].drop_duplicates().sort_values(["bench", "item"]).to_numpy()))
        return cls(judges, benchmarks, cells, items)


def hierarchical_design(
    frame: pd.DataFrame,
    features: Sequence[str],
    levels: HierarchyLevels,
    penalties: Mapping[str, float],
    include_items: bool = True,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Build the sparse intercept-and-slope hierarchy and its coefficient map."""
    _require_columns(frame, ("judge", "bench", "item", *features))
    multipliers = _multipliers(penalties)
    n_rows = len(frame)
    rows = np.arange(n_rows)
    blocks: list[sparse.csr_matrix] = []
    records: list[dict[str, object]] = []
    column = 0

    def add(block: str, matrix: sparse.csr_matrix, *, feature: str | None = None,
            judge: str | None = None, benchmark: str | None = None, item: str | None = None,
            multiplier: float = 1.0) -> None:
        nonlocal column
        blocks.append(matrix)
        for offset in range(matrix.shape[1]):
            records.append({
                "column": column + offset, "block": block, "feature": feature,
                "judge": judge if matrix.shape[1] == 1 else None,
                "benchmark": benchmark if matrix.shape[1] == 1 else None,
                "item": item if matrix.shape[1] == 1 else None,
                "multiplier": multiplier,
            })
        column += matrix.shape[1]

    # Add multi-category blocks one column at a time so every map row names its category.
    add("global_intercept", sparse.csr_matrix(np.ones((n_rows, 1))))
    _add_categories(add, "judge_intercept", frame.judge, levels.judges, rows, n_rows, multipliers["judge"], "judge")
    _add_categories(add, "benchmark_intercept", frame.bench, levels.benchmarks, rows, n_rows, multipliers["benchmark"], "benchmark")
    _add_pairs(add, "cell_intercept", frame.judge, frame.bench, levels.cells, rows, n_rows, multipliers["cell"])
    if include_items:
        _add_pairs(add, "item_intercept", frame.bench, frame.item, levels.items, rows, n_rows, multipliers["item"], first_name="benchmark", second_name="item")
    for feature in features:
        values = _feature_values(frame, feature)
        add("population_slope", sparse.csr_matrix(values[:, None]), feature=feature)
        _add_categories(add, "judge_slope", frame.judge, levels.judges, rows, n_rows, multipliers["judge"], "judge", feature, values)
        _add_categories(add, "benchmark_slope", frame.bench, levels.benchmarks, rows, n_rows, multipliers["benchmark"], "benchmark", feature, values)
        _add_pairs(add, "cell_slope", frame.judge, frame.bench, levels.cells, rows, n_rows, multipliers["cell"], feature=feature, values=values)
    return sparse.hstack(blocks, format="csr"), pd.DataFrame.from_records(records, columns=_MAP_COLUMNS)


def effective_coefficients(raw_coefficients: np.ndarray | pd.DataFrame, column_map: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct each feature's natural-scale slope for every judge-benchmark cell."""
    coefficients = _coefficients(raw_coefficients, column_map)
    mapped = column_map.copy()
    mapped["natural"] = coefficients * mapped.multiplier.to_numpy()
    output: list[dict[str, object]] = []
    for feature in mapped.loc[mapped.block == "population_slope", "feature"].drop_duplicates():
        population = _term(mapped, "population_slope", feature)
        cells = mapped[(mapped.block == "cell_slope") & (mapped.feature == feature)][["judge", "benchmark"]].drop_duplicates()
        for judge, benchmark in cells.itertuples(index=False):
            judge_deviation = _term(mapped, "judge_slope", feature, judge=judge)
            benchmark_deviation = _term(mapped, "benchmark_slope", feature, benchmark=benchmark)
            cell_deviation = _term(mapped, "cell_slope", feature, judge=judge, benchmark=benchmark)
            output.append({
                "feature": feature, "judge": judge, "benchmark": benchmark,
                "population": population, "judge_deviation": judge_deviation,
                "benchmark_deviation": benchmark_deviation, "cell_deviation": cell_deviation,
                "effective": population + judge_deviation + benchmark_deviation + cell_deviation,
            })
    return pd.DataFrame(output, columns=[
        "feature", "judge", "benchmark", "population", "judge_deviation",
        "benchmark_deviation", "cell_deviation", "effective",
    ])


def _add_categories(add, block, values, categories, rows, n_rows, multiplier, category_name, feature=None, slope=None) -> None:
    for category in categories:
        active = values.to_numpy() == category
        data = np.ones(active.sum()) if slope is None else slope[active]
        matrix = sparse.csr_matrix((data * multiplier, (rows[active], np.zeros(active.sum()))), shape=(n_rows, 1))
        add(block, matrix, feature=feature, multiplier=multiplier, **{category_name: category})


def _add_pairs(add, block, first, second, categories, rows, n_rows, multiplier, *, first_name="judge", second_name="benchmark", feature=None, values=None) -> None:
    first_values, second_values = first.to_numpy(), second.to_numpy()
    for first_category, second_category in categories:
        active = (first_values == first_category) & (second_values == second_category)
        data = np.ones(active.sum()) if values is None else values[active]
        matrix = sparse.csr_matrix((data * multiplier, (rows[active], np.zeros(active.sum()))), shape=(n_rows, 1))
        add(block, matrix, feature=feature, multiplier=multiplier,
            **{first_name: first_category, second_name: second_category})


def _multipliers(penalties: Mapping[str, float]) -> dict[str, float]:
    missing = set(_PENALTIES.values()) - set(penalties)
    if missing:
        raise ValueError(f"missing hierarchy penalties: {sorted(missing)}")
    result = {name: 1 / np.sqrt(float(penalties[key])) for name, key in _PENALTIES.items()}
    if any(not np.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("hierarchy penalties must be finite and positive")
    return result


def _feature_values(frame: pd.DataFrame, feature: str) -> np.ndarray:
    values = frame[feature].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"feature {feature!r} contains non-finite values")
    return values


def _coefficients(raw: np.ndarray | pd.DataFrame, column_map: pd.DataFrame) -> np.ndarray:
    if isinstance(raw, pd.DataFrame):
        if "coefficient" not in raw:
            raise ValueError("raw coefficient table needs a 'coefficient' column")
        if "column" in raw:
            raw = column_map[["column"]].merge(raw[["column", "coefficient"]], on="column", how="left", validate="one_to_one")
        values = raw.coefficient.to_numpy(dtype=float)
    else:
        values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) != len(column_map):
        raise ValueError("raw coefficient count must match the column map")
    if not np.isfinite(values).all():
        raise ValueError("raw coefficients must be finite")
    return values


def _term(mapped: pd.DataFrame, block: str, feature: str, *, judge: str | None = None, benchmark: str | None = None) -> float:
    selected = (mapped.block == block) & (mapped.feature == feature)
    if judge is not None:
        selected &= mapped.judge == judge
    if benchmark is not None:
        selected &= mapped.benchmark == benchmark
    return float(mapped.loc[selected, "natural"].item())


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"frame missing columns: {missing}")
