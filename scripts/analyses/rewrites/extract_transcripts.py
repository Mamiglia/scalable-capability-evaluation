"""Extract the unique (model, item) transcripts from the Task 8 manipulation subset.

The subset (50 GPQA pairs x 15 items) is fixed and identical across every config
in scripts/pairwise-newjudges/configs/manipulations/ - any one of them names it.
A model's answer to an item appears in many pairs; the rewrite unit is the
transcript, so each (model, item) is extracted exactly once here, not once per
pair slot.

Reruns (e.g. the subset changing) only append rows for newly required keys.
Existing rows are never rewritten or dropped, even if the subset no longer
needs them - a rewrite campaign may already be reading the file.

Usage:
  uv run python scripts/analyses/rewrites/extract_transcripts.py [config.yaml]
"""
import json
import sys
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401  epoch .eval zips use zstd; py<3.14 zipfile needs this shim
except ImportError:
    pass

import yaml
from inspect_ai.log import read_eval_log

from judge.util import extract_transcript, first_correct_by_id

ROOT = Path(__file__).resolve().parents[3]
SUBSET_CONFIG = ROOT / "scripts/pairwise-newjudges/configs/manipulations/trunc_first_10.yaml"
OUT_PATH = ROOT / "judge_results/rewrites/originals.jsonl"


def unique_keys(comparisons: list, sample_ids: dict) -> set[tuple[str, str]]:
    """(model, item_id) pairs used anywhere in the subset - the rewrite unit."""
    keys = set()
    for major, minor in comparisons:
        items = sample_ids[f"{major}-vs-{minor}"]
        keys.update((major, item) for item in items)
        keys.update((minor, item) for item in items)
    return keys


def extract_row(sample, model: str, item_id: str) -> dict:
    transcript = extract_transcript(sample, include_reasoning=False)
    return {
        "model": model,
        "item_id": item_id,
        "question": sample.input,
        "transcript": transcript,
        "n_words": len(transcript.split()),
        "n_chars": len(transcript),
    }


def load_existing(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {(r["model"], r["item_id"]): r for r in rows}


def main(config: Path = SUBSET_CONFIG) -> None:
    cfg = yaml.safe_load(config.read_text())
    assert cfg["include_reasoning"] is False, "subset must be final-text-only"

    keys = unique_keys(cfg["comparisons"], cfg["sample_ids"])
    n_slots = sum(len(v) for v in cfg["sample_ids"].values()) * 2  # x2: both models in each pair
    existing = load_existing(OUT_PATH)

    items_by_model: dict[str, list[str]] = {}
    for model, item_id in keys:
        items_by_model.setdefault(model, []).append(item_id)

    fresh: dict[tuple[str, str], dict] = {}
    for model, item_ids in sorted(items_by_model.items()):
        log = read_eval_log(str(ROOT / cfg["transcripts"][model]))
        correct = first_correct_by_id(log)  # id -> correct samples, earliest epoch first
        for item_id in sorted(item_ids):
            group = correct.get(item_id)
            if not group:
                raise ValueError(f"no correct epoch for {model}/{item_id} in {cfg['transcripts'][model]}")
            fresh[(model, item_id)] = extract_row(group[0], model, item_id)

    # each key is extracted exactly once - if this fails, some (model, item) got
    # written twice (or dropped), which breaks the within-item comparison design
    assert len(fresh) == len(keys), f"{len(fresh)} rows but {len(keys)} unique keys"

    # a key already on disk must extract to the same transcript every time - if
    # not, something upstream moved and must not be papered over by silently
    # rewriting a row a live rewrite campaign may already be reading
    for key, row in fresh.items():
        old = existing.get(key)
        if old is not None and old["transcript"] != row["transcript"]:
            raise ValueError(f"transcript changed for {key} since the last extraction - stopping")

    new_rows = [row for key, row in fresh.items() if key not in existing]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")

    n_surplus = len(existing) - len(existing.keys() & keys)
    print(f"{n_slots} pair-model-item slots -> {len(keys)} unique transcripts needed")
    print(f"{len(new_rows)} new rows appended; {len(existing)} rows already on disk"
          f" ({n_surplus} no longer in the subset, left in place)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SUBSET_CONFIG)
