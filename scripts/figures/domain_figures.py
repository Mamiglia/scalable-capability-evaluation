"""Figures of a frozen domain: theta by semester and by release date, item difficulty per bench, theta-vs-b map per
semester, accuracy heatmap.

    uv run python scripts/figures/domain_figures.py science
    uv run python scripts/figures/domain_figures.py science math coding cyber   # + joined analysis/{benchmarks,theta_release}

Reads data/meta/<domain>/{models,items,responses}.csv (release dates: configs/models/*/*.yml), writes
analysis/<domain>/{theta,theta_release,difficulty,wright,accuracy}.{png,svg}. The figures: plots/domains.py.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from plots.domains import (SEMESTERS, accuracy_figure, benchmarks_figure, difficulty_figure, release_figure,
                           theta_figure, wright_figure)


def release_dates() -> pd.Series:
    """Model short name -> release date, from the `name` / `release_date` fields of configs/models/*/*.yml."""
    configs = [yaml.safe_load(p.read_text()) or {} for p in Path("configs/models").glob("*/*.yml")]
    raw = {c["name"].split("/")[-1].lower(): c.get("release_date") for c in configs if "name" in c}
    # a few configs hold "2023-02" or a range "2024-04-23/2024-05-21": take the first date
    return pd.to_datetime(pd.Series(raw, dtype=str).str.split("/").str[0], format="mixed")


def main(domains: list[str]) -> int:
    loaded, dates = {}, release_dates()
    for domain in domains:
        root = Path("data/meta") / domain
        models = pd.read_csv(root / "models.csv", index_col="model")
        items = pd.read_csv(root / "items.csv", dtype={"item": str})
        responses = pd.read_csv(root / "responses.csv", dtype={"item": str})
        semesters = [s for s in SEMESTERS if s in set(models.semester.dropna())]
        out = Path("analysis") / domain
        out.mkdir(parents=True, exist_ok=True)
        theta_figure(models, semesters, out)
        release_figure({domain: models}, dates, out)
        difficulty_figure(items, out)
        wright_figure(models, items, out)
        accuracy_figure(models, responses, out)
        loaded[domain] = (models, responses)
    if len(loaded) > 1:  # joined across domains
        release_figure({d: m for d, (m, _) in loaded.items()}, dates, Path("analysis"))
        benchmarks_figure(loaded, Path("analysis"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
