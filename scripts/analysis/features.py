"""
Computes the deterministic transcript features of two transcript views that generality-labs/glm_features does not
hold: the reasoning alone (for the channel analyses) and the transcript with each third removed or kept alone (for the
thirds analyses). transcript_parts.py reads them.

Writes to <paths.cache> (config.yaml):
    features_reasoning_only.csv   one row per first attempt on a train item by a reference or target of the tasks of
                                  tasks.domains and tasks.semesters: path (the transcript log), item, correct and the
                                  features
    features_thirds.csv           the same rows once per thirds condition, with a condition column (full, thirds-23,
                                  thirds-13, thirds-12, thirds-1, thirds-2, thirds-3)
A file already there is reused; delete it to recompute. Reads the transcripts in paths.data.
Run from the repository root:
    uv run python scripts/analysis/features.py [--config scripts/analysis/config.yaml]
"""
import argparse
from pathlib import Path

from analysis.config import apply_paths, load_config
from challenges.channels.features import feature_rows


def main(config: dict) -> None:
    """
    Writes each view's feature table unless it is already in the cache.

    Inputs:
        config: the analysis configuration (scripts/analysis/config.yaml); uses paths, tasks.domains and
            tasks.semesters.
    Outputs:
        <paths.cache>/features_reasoning_only.csv and <paths.cache>/features_thirds.csv, each only when absent.
    """
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
