"""The frozen tables of a domain (data/meta/<domain>/, written by `uv run reference`) and the roster.

    models.csv  index model; semester; global_theta; path_<bench> (repeat-0 log, relative to data/);
                per task <sem>_role (reference / target / blank), <sem>_theta, <sem>_theta_sd
    items.csv   bench, item, guess, a, b (+ _sd); per task <sem>_set (easy / hard / ""), <sem>_train
"""

from pathlib import Path

import pandas as pd
import yaml

from configs.domain import DomainConfig

SEMESTERS = Path("configs/semesters.yml")
DATA = Path("data")
META = DATA / "meta"


def frozen_dir(domain: DomainConfig) -> Path:
    return META / domain.name


def load(domain: DomainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen tables of a domain: (models indexed by model, items)."""
    directory = frozen_dir(domain)
    if not (directory / "models.csv").exists():
        raise FileNotFoundError(f"{directory} not frozen — run `uv run reference {domain.source or domain.name}`")
    models = pd.read_csv(directory / "models.csv", index_col="model")
    items = pd.read_csv(directory / "items.csv", dtype={"item": str})
    return models, items


def load_semesters(path: Path = SEMESTERS) -> dict[str, list[str]]:
    """configs/semesters.yml as {semester: [model shortname, ...]}, in file order."""
    return {str(k): list(v) for k, v in yaml.safe_load(path.read_text()).items()}
