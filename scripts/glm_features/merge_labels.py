"""Fold RST labels into glm_features/: per-log label files and the 11 structure columns.

    uv run python scripts/glm_features/merge_labels.py [--labels glm_features/rst/labels]

Reads every shard-*.jsonl under --labels (written by rst_labels.py on the parser machine),
writes <log path>.labels.jsonl next to each log's CSV (one line per transcript: item, epoch,
labels as [type, in_reasoning] pairs, failed_parts), and adds the STRUCTURE_FEATURES columns
to that CSV: the visible view uses every label, the hidden view drops reasoning labels.
Rows without labels keep NaN. Re-running replaces the columns.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from protocols.transcript_features.features import STRUCTURE_FEATURES, transition_features

FEATURES_DIR = Path("glm_features")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=FEATURES_DIR / "rst" / "labels")
    args = parser.parse_args()
    by_log: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    for shard in sorted(args.labels.rglob("shard-*.jsonl")):
        with shard.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_log[row["log"]][(row["item"], row["epoch"])] = row
    n_rows = 0
    for log, rows in sorted(by_log.items()):
        csv = FEATURES_DIR / Path(log).with_suffix(".csv")
        with (csv.with_suffix(".labels.jsonl")).open("w") as f:
            for (item, epoch), row in sorted(rows.items()):
                f.write(json.dumps({"item": item, "epoch": epoch, "labels": row["labels"],
                                    "failed_parts": row["failed_parts"]}) + "\n")
        table = pd.read_csv(csv, dtype={"item": str})
        values = pd.DataFrame(np.nan, index=table.index, columns=STRUCTURE_FEATURES)
        for i, (item, epoch, view) in enumerate(zip(table.item, table.epoch, table.reasoning)):
            row = rows.get((item, int(epoch)))
            if row is None:
                continue
            types = [t for t, in_reasoning in row["labels"] if view == "visible" or not in_reasoning]
            values.iloc[i] = pd.Series(transition_features(types))
            n_rows += 1
        table = table.drop(columns=[c for c in STRUCTURE_FEATURES if c in table], errors="ignore")
        pd.concat([table, values], axis=1).to_csv(csv, index=False)
    print(f"{len(by_log)} logs, {sum(map(len, by_log.values()))} transcripts labelled, {n_rows} CSV rows updated")


if __name__ == "__main__":
    main()
