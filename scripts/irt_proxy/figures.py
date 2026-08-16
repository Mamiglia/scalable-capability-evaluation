"""Render the current easiest-slice prediction figure set."""

from pathlib import Path

import pandas as pd

from judge_figures import render_judge_figures
from partition_figures import render_partition_figures


def render_figures(tables: dict[str, pd.DataFrame], output_directory: Path) -> None:
    """Render only the split-specific figures; old suites are not regenerated."""
    render_partition_figures(
        tables["partition_predictions.csv"],
        tables["partition_metrics.csv"],
        tables["null_metrics.csv"],
        output_directory,
    )
    render_judge_figures(
        tables["partition_predictions.csv"],
        tables["null_metrics.csv"],
        output_directory,
    )
