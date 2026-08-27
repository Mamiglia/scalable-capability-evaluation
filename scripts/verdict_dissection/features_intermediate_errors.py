"""Intermediate-error features from the Codex annotations (annotate/intermediate_errors_codex.py).

    uv run python scripts/verdict_dissection/features_intermediate_errors.py [--annotations <jsonl>]

Per transcript: error count in total and per kind, errors per 100 words, and whether any error was
flagged. Transcripts without an annotation (other benchmarks, or unparsed replies) get NaN, which
the design-matrix builder fills with 0 for the block. Writes <features_root>/raw/intermediate-errors.parquet.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
KINDS = ["nomenclature", "notation", "constant_or_formula", "arithmetic", "contradiction", "definition", "other"]
DEFAULT = ROOT / "judge_results/verdict_dissection/annotate_outputs/intermediate_errors_codex.jsonl"


def error_features(records, words):
    """records: parsed annotation dicts keyed by content_hash; words: Series of word counts by content_hash."""
    rows = {}
    for hash_, record in records.items():
        if not record["parsed"]:
            continue
        errors = record["errors"]
        kinds = [e.get("kind") if e.get("kind") in KINDS else "other" for e in errors]
        row = {"error_count": len(errors), "any_error": int(bool(errors)),
               "errors_per_100_words": 100 * len(errors) / max(words.get(hash_, 1), 1)}
        row |= {f"errors_{k}": kinds.count(k) for k in KINDS}
        rows[hash_] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    surface = pd.read_parquet(features_root / "raw/surface-textual.parquet", columns=KEYS + ["n_words"])
    records = {}
    for line in args.annotations.open():
        if line.strip():
            record = json.loads(line)
            records[record["content_hash"]] = record          # last annotation of a hash wins
    words = surface.drop_duplicates("content_hash").set_index("content_hash").n_words
    features = error_features(records, words)
    table = surface[KEYS].merge(features, left_on="content_hash", right_index=True, how="left")
    table.to_parquet(features_root / "raw/intermediate-errors.parquet", index=False)
    annotated = table.error_count.notna()
    print(f"{int(annotated.sum())} of {len(table)} transcripts annotated; "
          f"{int((table.any_error == 1).sum())} with at least one error")
    print(table[annotated].groupby("bench")[["error_count", "any_error", "errors_per_100_words"] + [f"errors_{k}" for k in KINDS]].mean().round(2).T.to_string())


if __name__ == "__main__":
    main()
