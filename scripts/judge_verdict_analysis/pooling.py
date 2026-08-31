"""Relative-penalty scaling and coordinate selection for sparse fits."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


POOLING_PARAMETERS = (
    "judge_deviation_penalty",
    "benchmark_deviation_penalty",
    "cell_deviation_penalty",
    "item_intercept_penalty",
)
Score = tuple[float, int, bool]


def penalty_multiplier(relative_penalty: float, penalty: str) -> float:
    """Return the design-column multiplier for one relative penalty."""
    ratio = float(relative_penalty)
    if penalty not in {"l1", "ridge"}:
        raise ValueError("penalty must be 'l1' or 'ridge'")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("relative penalties must be finite and positive")
    return 1 / ratio if penalty == "l1" else 1 / np.sqrt(ratio)


def coordinate_search(
    cs: Sequence[float],
    ratios: Sequence[float],
    evaluate: Callable[[list[dict[str, float]]], list[Score]],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Alternate pooling-ratio coordinates with C for at most two passes."""
    cs = tuple(map(float, cs))
    ratios = tuple(map(float, ratios))
    if not cs or not ratios or 1.0 not in ratios:
        raise ValueError("C and pooling grids must be non-empty and pooling must contain 1")
    if any(not np.isfinite(value) or value <= 0 for value in (*cs, *ratios)):
        raise ValueError("C and pooling grids must be finite and positive")
    state = {"C": cs[0], **dict.fromkeys(POOLING_PARAMETERS, 1.0)}
    records: dict[tuple[float, ...], dict[str, object]] = {}

    def select(parameter: str, values: Sequence[float], pass_index: int) -> None:
        nonlocal state
        candidates = [state | {parameter: value} for value in values]
        scores = evaluate(candidates)
        if len(scores) != len(candidates):
            raise ValueError("pooling evaluator returned the wrong number of scores")
        if any(not np.isfinite(loss) for loss, _, _ in scores):
            raise ValueError("pooling evaluator returned a non-finite loss")
        for candidate, (loss, max_n_iter, converged) in zip(candidates, scores):
            key = tuple(candidate[name] for name in ("C", *POOLING_PARAMETERS))
            records.setdefault(key, {
                **candidate, "pass": pass_index, "coordinate": parameter,
                "mean_log_loss": float(loss), "max_n_iter": int(max_n_iter),
                "converged": bool(converged),
            })
        current = state[parameter]
        state = min(
            zip(candidates, scores),
            key=lambda result: (not result[1][2], result[1][0],
                                result[0][parameter] != current,
                                result[0][parameter]),
        )[0]

    for pass_index in (1, 2):
        previous = state.copy()
        select("C", cs, pass_index)
        for parameter in POOLING_PARAMETERS:
            select(parameter, ratios, pass_index)
            select("C", cs, pass_index)
        if state == previous:
            break
    selected = tuple(state[name] for name in ("C", *POOLING_PARAMETERS))
    rows = list(records.values())
    for row in rows:
        row["selected"] = tuple(
            row[name] for name in ("C", *POOLING_PARAMETERS)
        ) == selected
    return state, rows
