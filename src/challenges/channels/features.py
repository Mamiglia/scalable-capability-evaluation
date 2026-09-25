"""Deterministic transcript features of the reasoning channel alone, for the channel ablation of the
Deterministic features protocol (scripts/analyses/other_protocol_channels.py).

    SMALL_JUDGE_DATA=/Volumes/T7/small-judge-remote/data uv run python src/challenges/channels/features.py [--out CSV]

glm_features/ holds two views per transcript, "visible" (messages + reasoning) and "hidden" (messages
only); the other channel views derive from them except the reasoning alone, extracted here: every
assistant message reduced to its reasoning blocks, no tool calls, then protocols.transcript_features
.features.extract_features. Rows: the correct first attempts of the references' and targets' train
items of 2025-II and 2026-I in every domain; columns path (the data/ log), item, correct, FEATURES.
"""

import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from inspect_ai.log import EvalSample

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from analysis import loading  # noqa: E402
from evals.sample import extract_score  # noqa: E402
from protocols.transcript_features.features import FEATURES, extract_features  # noqa: E402
from protocols.transcript_features.reader import MEMBER  # noqa: E402
from reference.tables import DATA  # noqa: E402

DOMAINS, SEMESTERS = ("science", "math", "coding", "cyber"), ("2025-II", "2026-I")
OUT = ROOT / "analysis/results/figures/channels/features_only_reasoning.csv"


def reasoning_only(sample: EvalSample) -> list:
    """The sample's messages with every assistant message cut to its reasoning blocks and no tool calls."""
    out = []
    for message in sample.messages:
        if message.role != "assistant":
            out.append(message)
            continue
        blocks = [] if isinstance(message.content, str) else [b for b in message.content if b.type == "reasoning"]
        out.append(message.model_copy(update={"content": blocks, "tool_calls": None}))
    return out


def extract(job: tuple[str, set[str]]) -> list[dict]:
    """Feature rows of the first attempts on `items` in the log at `path` (relative to data/)."""
    path, items = job
    rows = []
    with zipfile.ZipFile(DATA / path) as archive:
        for name in archive.namelist():
            if not (m := MEMBER.match(name)) or int(m[2]) != 1 or m[1] not in items:
                continue
            sample = EvalSample.model_validate_json(archive.read(name))
            if not sample.messages:
                continue
            rows.append({"path": path, "item": m[1], "correct": extract_score(sample),
                         **extract_features(reasoning_only(sample), include_reasoning=True)})
    return rows


def jobs() -> list[tuple[str, set[str]]]:
    """(log path, train item ids) per log of every reference and target of the two semesters."""
    wanted: dict[str, set[str]] = {}
    for domain in DOMAINS:
        for semester in SEMESTERS:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            for cohort in (task.references, task.test):
                for bench, ids in cohort.items.groupby("bench").item:
                    for model in cohort.models.index:
                        if isinstance(path := cohort.models.loc[model, f"path_{bench}"], str):
                            wanted.setdefault(path, set()).update(map(str, ids))
    return sorted(wanted.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=OUT, type=Path)
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    todo = jobs()
    print(f"{len(todo)} logs", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, part in enumerate(pool.map(extract, todo), start=1):
            rows.extend(part)
            if i % 25 == 0:
                print(f"{i}/{len(todo)} logs, {len(rows)} rows", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["path", "item", "correct", *FEATURES]).to_csv(args.out, index=False)
    print(args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
