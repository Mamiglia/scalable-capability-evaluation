"""
Loads the analysis configuration (scripts/analysis/config.yaml) and points the readers of transcripts, features and
embeddings at the paths it names.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "scripts/analysis/config.yaml") -> dict[str, Any]:
    """
    Reads the analysis configuration from a YAML file.

    Inputs:
        path: the YAML file, relative to the repository root (default scripts/analysis/config.yaml).
    Outputs:
        the configuration as nested dicts and lists, keys as in the file.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def apply_paths(config: dict[str, Any]) -> None:
    """
    Points the readers of the frozen tables, transcripts, features and embeddings at the folders the configuration
    names, in this process and (through environment variables) in the worker processes it starts. paths.logs is not
    used here: the judge runs are read from the log folders the judge configs name (logs/...), which is why paths.logs
    must stay logs/.

    Inputs:
        config: the loaded configuration; uses paths.data, paths.features and paths.embeddings.
    Outputs:
        none returned; sets reference.tables.DATA, META (<paths.data>/meta) and FEATURES,
        protocols.embeddings.EMBEDDINGS, and the environment variables SMALL_JUDGE_DATA and SMALL_JUDGE_EMBEDDINGS.
    """
    import os

    from protocols import embeddings
    from reference import tables

    tables.DATA = Path(config["paths"]["data"])
    tables.META = tables.DATA / "meta"  # the frozen tables, fetched with the transcripts
    tables.FEATURES = Path(config["paths"]["features"])
    embeddings.EMBEDDINGS = Path(config["paths"]["embeddings"])
    # worker processes re-import these modules: they read the paths from the environment
    os.environ["SMALL_JUDGE_DATA"] = str(tables.DATA)
    os.environ["SMALL_JUDGE_EMBEDDINGS"] = str(embeddings.EMBEDDINGS)
