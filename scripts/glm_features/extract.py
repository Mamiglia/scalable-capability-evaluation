"""Per-transcript GLM features for every sample of every data/ log -> glm_features/.

    uv run python scripts/glm_features/extract.py [bench ...] [--data data] [--out glm_features]

glm_features/ is a clone of https://huggingface.co/datasets/LASR-G5/glm_features and mirrors
data/'s layout: <bench>/<task_args_hash>/<model>/<repeat>/<eval id>.csv, one row per sample
and reasoning view (visible: text + raw CoT; hidden: text only). Values are per-transcript
rates; averaging over a model's selected items happens at fit time. Only logs newer than
their CSV are re-extracted; `legacy/` and `judge/` are skipped.
"""

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd
from inspect_ai.log import EvalSample

from evals.sample import extract_score
from protocols.transcript_features.reader import sample_features

MEMBER = re.compile(r"samples/(.+)_epoch_(\d+)\.json$")
SKIP = {"legacy", "judge"}


def extract_log(path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not (member := MEMBER.match(name)):
                continue
            sample = EvalSample.model_validate_json(archive.read(name))
            for view in ("visible", "hidden"):
                rows.append({"item": member[1], "epoch": int(member[2]), "correct": extract_score(sample),
                             "reasoning": view, **sample_features(sample, include_reasoning=view == "visible")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benches", nargs="*", help="default: every bench under --data")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="glm_features", type=Path)
    args = parser.parse_args()
    for log in sorted(args.data.glob("*/*/*/*/*.eval")):
        relative = log.relative_to(args.data)
        if relative.parts[0] in SKIP or (args.benches and relative.parts[0] not in args.benches):
            continue
        target = args.out / relative.with_suffix(".csv")
        if target.exists() and target.stat().st_mtime >= log.stat().st_mtime:
            continue
        if log.stat().st_size < 1024 and log.read_bytes().startswith(b"version https://git-lfs"):
            print(f"skip (LFS pointer, not fetched): {log}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        extract_log(log).to_csv(target, index=False)
        print(target)


if __name__ == "__main__":
    main()
