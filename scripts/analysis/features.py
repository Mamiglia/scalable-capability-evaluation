"""Deterministic transcript features of two transcript views not in generality-labs/glm_features: the reasoning alone
(channels) and each third removed or kept alone (thirds). The transcript_parts step reads them.

Writes to paths.cache (config.yaml): features_reasoning_only.csv and features_thirds.csv, one row per correct first
attempt of the main-analysis tasks' references and targets (and per thirds condition). Reuses a file already there;
delete it to recompute. Reads transcripts from paths.data. Run from the repository root:
    uv run python scripts/analysis/features.py [--config scripts/analysis/config.yaml]
"""
import argparse
from pathlib import Path

from analysis.config import apply_paths, load_config
from challenges.channels.features import feature_rows


def main(config: dict) -> None:
    apply_paths(config)
    cache = Path(config["paths"]["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    for view in ("reasoning_only", "thirds"):
        out = cache / f"features_{view}.csv"
        if out.exists():
            print("reusing", out)
            continue
        feature_rows(view, config["tasks"]["domains"], config["tasks"]["semesters"]).to_csv(out, index=False)
        print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    main(load_config(parser.parse_args().config))
