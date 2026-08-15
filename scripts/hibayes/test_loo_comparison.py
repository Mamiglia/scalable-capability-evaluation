"""The paired elpd difference is only meaningful if both runs saw the same rows."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from loo_comparison import check_same_rows


def run_dir(root: Path, name: str, frame: pd.DataFrame) -> Path:
    path = root / name
    path.mkdir()
    frame.to_parquet(path / "data.parquet")
    return path


def frame(**overrides) -> pd.DataFrame:
    return pd.DataFrame({"chosen_left": [1, 0, 1], "length_difference_sd": [0.5, -0.5, 0.0]} | overrides)


def test_identical_rows_pass(tmp_path):
    a = run_dir(tmp_path, "a", frame())
    b = run_dir(tmp_path, "b", frame())
    assert check_same_rows(a, b) == 2


def test_extra_columns_are_ignored(tmp_path):
    """The candidate legitimately carries predictors the baseline does not."""
    a = run_dir(tmp_path, "a", frame(hedging_difference_sd=[1.0, 2.0, 3.0]))
    b = run_dir(tmp_path, "b", frame())
    assert check_same_rows(a, b) == 2


def test_different_rows_are_rejected(tmp_path):
    a = run_dir(tmp_path, "a", frame())
    b = run_dir(tmp_path, "b", frame(chosen_left=[0, 0, 1]))
    with pytest.raises(ValueError, match="different rows"):
        check_same_rows(a, b)


def test_reordered_rows_are_rejected(tmp_path):
    """Same rows, wrong order: the pointwise pairing would be silently scrambled."""
    a = run_dir(tmp_path, "a", frame())
    b = run_dir(tmp_path, "b", frame().iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="different rows"):
        check_same_rows(a, b)
