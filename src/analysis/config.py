"""The analysis configuration (scripts/analysis/config.yaml): loading it, and pointing the data readers at its paths."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "scripts/analysis/config.yaml") -> dict[str, Any]:
    """The configuration as nested dicts and lists, keys as in the file.

    path: the YAML file, relative to the repository root.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def apply_paths(config: dict[str, Any]) -> None:
    """Point the readers of transcripts, features and embeddings at the configured checkouts.

    config: the loaded configuration; uses paths.data, paths.features and paths.embeddings. Judge logs are read from
        the directories the judge configs name (logs/...), so paths.logs is documentation only.
    """
    import os

    from protocols import embeddings
    from reference import tables

    tables.DATA = Path(config["paths"]["data"])
    tables.FEATURES = Path(config["paths"]["features"])
    embeddings.EMBEDDINGS = Path(config["paths"]["embeddings"])
    # worker processes re-import these modules: they read the paths from the environment
    os.environ["SMALL_JUDGE_DATA"] = str(tables.DATA)
    os.environ["SMALL_JUDGE_EMBEDDINGS"] = str(embeddings.EMBEDDINGS)
