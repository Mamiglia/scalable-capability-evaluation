#!/usr/bin/env python3
"""Merge, QC, and build the style_standardised_gpt rewrite condition."""
import json
import re
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/hibayes"))

from normalise_prefix import strip_role_prefix  # noqa: E402
from style_features import (  # noqa: E402
    STANDALONE_CAPITAL_I,
    STANDALONE_I_RE,
    _term_pattern,
    feature_rates,
    load_lexicon,
)


CONDITION = "style_standardised_gpt"
EXPECTED_ROWS = 1997
REWRITE_DIR = ROOT / "judge_results/rewrites"
MANIFEST = REWRITE_DIR / "paraphrase_gpt_v2.jsonl"
PARTS = tuple(REWRITE_DIR / f"{CONDITION}.part-{index}.jsonl" for index in range(3))
MERGED = REWRITE_DIR / f"{CONDITION}.jsonl"
ORIGINALS = REWRITE_DIR / "originals.jsonl"
QC_OUT = REWRITE_DIR / f"qc_{CONDITION}.json"
EXCLUDED_OUT = REWRITE_DIR / f"excluded_{CONDITION}.json"
FEATURES_OUT = REWRITE_DIR / f"style_features_{CONDITION}.jsonl"
LEXICON = ROOT / "scripts/hibayes/style_lexicon.yaml"
EVAL_DIR = ROOT / f"data/rewrites/{CONDITION}"
CAMPAIGN_CONFIG = ROOT / f"scripts/pairwise-newjudges/configs/manipulations/{CONDITION}.yaml"
EXPECTED_STORES = 42
EXPECTED_PAIR_CELLS = 1500

TARGETED_CATEGORIES = ("personal_pronouns", "hedging", "backtracking")
# Preserve actual answer labels, not prose such as "why B is the correct answer:".
# Uppercase ANSWER is also accepted inside the corpus's LaTeX wrappers.
ANSWER_LINE_RE = re.compile(
    r"(?i:^\s*(?:\\\[\s*)?(?:\\boxed\{\s*)?\\text\{\s*answer\s*:)"
    r"|(?i:^\s*(?:\[assistant\]:\s*)?(?:#{1,6}\s*)?(?:✅\s*)?"
    r"(?:\*\*)?(?:\d+\.\s*)?(?:\*\*)?"
    r"(?:(?:final|correct)\s+)?answer\s*:\s*(?:\*\*)?)",
)
DELIMITER_RE = re.compile(r"<<<\s*(?:REWRITE|END|TRANSCRIPT)?\s*\d*\s*>>>|<<<|>>>")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(row: dict) -> tuple[str, str]:
    try:
        return row["model"], row["item_id"]
    except KeyError as exc:
        raise ValueError(f"row missing {exc.args[0]!r}: {row}") from exc


def _unique_map(rows: list[dict], label: str) -> dict[tuple[str, str], dict]:
    out = {}
    duplicates = []
    for row in rows:
        key = _key(row)
        if key in out:
            duplicates.append(key)
        out[key] = row
    if duplicates:
        raise ValueError(f"{label} has duplicate keys (e.g. {sorted(set(duplicates))[:3]})")
    return out


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.replace(path)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def merge_parts(manifest_path: Path, part_paths: list[Path] | tuple[Path, ...],
                merged_path: Path, expected_rows: int = EXPECTED_ROWS) -> list[dict]:
    """Merge exactly three complete shards in manifest order, or write nothing."""
    if len(part_paths) != 3:
        raise ValueError(f"expected exactly 3 part files, got {len(part_paths)}")
    manifest_rows = _read_jsonl(manifest_path)
    manifest = _unique_map(manifest_rows, "manifest")
    if len(manifest) != expected_rows:
        raise ValueError(f"manifest has {len(manifest)} keys, expected {expected_rows}")

    part_rows = [row for path in part_paths for row in _read_jsonl(path)]
    parts = _unique_map(part_rows, "part files")
    extra = parts.keys() - manifest.keys()
    missing = manifest.keys() - parts.keys()
    problems = []
    if extra:
        problems.append(f"{len(extra)} extra keys (e.g. {sorted(extra)[:3]})")
    if missing:
        problems.append(f"{len(missing)} missing keys (e.g. {sorted(missing)[:3]})")
    if problems:
        raise ValueError("; ".join(problems))

    merged = [parts[_key(row)] for row in manifest_rows]
    _write_jsonl_atomic(merged_path, merged)
    return merged


def _answer_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if ANSWER_LINE_RE.search(line)]


def _has_content_after_final_answer(text: str) -> bool:
    lines = text.splitlines()
    answer_indexes = [index for index, line in enumerate(lines) if ANSWER_LINE_RE.search(line)]
    return bool(answer_indexes and any(line.strip() for line in lines[answer_indexes[-1] + 1:]))


def structural_failure(original: str, rewritten: str) -> str | None:
    if not rewritten.strip():
        return "empty"
    if DELIMITER_RE.search(rewritten):
        return "delimiter_leak"
    wanted = _answer_lines(original)
    got = _answer_lines(rewritten)
    if len(wanted) != len(got):
        return "answer_line_count_changed"
    if any(a.rstrip().encode("utf-8") != b.rstrip().encode("utf-8")
           for a, b in zip(wanted, got)):
        return "answer_lines_changed"
    if _has_content_after_final_answer(original) != _has_content_after_final_answer(rewritten):
        return "final_answer_trailing_content_changed"
    return None


def _category_matches(text: str, terms: list[str]) -> list[tuple[int, int, str]]:
    """Canonical style-feature matches with spans for human-readable contexts."""
    matches = []
    if STANDALONE_CAPITAL_I in terms:
        matches.extend((m.start(), m.end(), m.group()) for m in STANDALONE_I_RE.finditer(text))
    other = [term for term in terms if term != STANDALONE_CAPITAL_I]
    if other:
        pattern = "|".join(_term_pattern(term) for term in sorted(other, key=len, reverse=True))
        matches.extend((m.start(), m.end(), text[m.start():m.end()])
                       for m in re.finditer(pattern, text.lower()))
    return sorted(matches)


def qc_rows(rows: list[dict], originals: dict[tuple[str, str], dict], lexicon: dict) -> dict:
    missing_categories = set(TARGETED_CATEGORIES) - lexicon.keys()
    if missing_categories:
        raise ValueError(f"lexicon missing targeted categories: {sorted(missing_categories)}")

    failures = []
    features = []
    residuals = {category: {"count": 0, "contexts": []}
                 for category in TARGETED_CATEGORIES}
    for row in rows:
        key = _key(row)
        if key not in originals:
            raise ValueError(f"no original transcript for {key}")
        original_raw = originals[key]["transcript"]
        rewritten_raw = row["rewritten"]
        reason = structural_failure(original_raw, rewritten_raw)
        if reason:
            failures.append({"model": key[0], "item_id": key[1], "reason": reason})

        original = strip_role_prefix(original_raw)
        rewritten = strip_role_prefix(rewritten_raw)
        before = feature_rates(original, lexicon)
        after = feature_rates(rewritten, lexicon)
        features.append({
            "model": key[0],
            "item_id": key[1],
            "n_words_original": len(original.split()),
            "n_words_rewritten": len(rewritten.split()),
            "original": before,
            "rewritten": after,
            "delta": {category: after[category] - before[category] for category in lexicon},
        })
        for category in TARGETED_CATEGORIES:
            matches = _category_matches(rewritten, lexicon[category])
            residuals[category]["count"] += len(matches)
            for start, end, matched in matches:
                residuals[category]["contexts"].append({
                    "model": key[0],
                    "item_id": key[1],
                    "match": matched,
                    "context": rewritten[max(0, start - 60):min(len(rewritten), end + 60)],
                })
    return {"failures": failures, "features": features, "residual_targeted": residuals}


def _build_evals(rows: list[dict], out_dir: Path) -> list[dict]:
    from build_rewrite_evals import build_model_store

    rewrites = {_key(row): strip_role_prefix(row["rewritten"]) for row in rows}
    by_model = {}
    for model, item_id in rewrites:
        by_model.setdefault(model, []).append(item_id)

    bad = []
    for model, items in sorted(by_model.items()):
        built, model_bad = build_model_store(model, items, rewrites, out_dir)
        bad.extend({"model": cell["model"], "item_id": cell["item"],
                    "reason": cell["reason"]} for cell in model_bad)
        print(f"  {model}: {len(built)} items -> {out_dir / f'{model}.eval'}")
    return bad


def _eval_store_keys(directory: Path) -> set[tuple[str, str]]:
    from inspect_ai.log import read_eval_log

    keys = set()
    for path in sorted(directory.glob("*.eval")):
        for sample in read_eval_log(str(path)).samples:
            key = path.stem, str(sample.id)
            if key in keys:
                raise ValueError(f"duplicate built eval key: {key}")
            keys.add(key)
    return keys


def _validate_campaign(directory: Path, manifest_keys: set[tuple[str, str]]) -> None:
    cfg = yaml.safe_load(CAMPAIGN_CONFIG.read_text())
    comparisons = cfg["comparisons"]
    if len(comparisons) != 50:
        raise ValueError(f"campaign has {len(comparisons)} comparisons, expected 50")

    required = set()
    pair_cells = 0
    for major, minor in comparisons:
        items = cfg["sample_ids"][f"{major}-vs-{minor}"]
        if len(items) != 30 or len(set(items)) != 30:
            raise ValueError(f"{major}-vs-{minor} does not have 30 unique items")
        pair_cells += len(items)
        required.update((major, item) for item in items)
        required.update((minor, item) for item in items)
    if pair_cells != EXPECTED_PAIR_CELLS:
        raise ValueError(f"campaign has {pair_cells} pair-cells, expected {EXPECTED_PAIR_CELLS}")
    if required != manifest_keys:
        raise ValueError("campaign sides do not exactly match the manifest keys")

    expected_models = set(cfg["transcripts"])
    stores = {path.stem for path in directory.glob("*.eval")}
    if len(expected_models) != EXPECTED_STORES or stores != expected_models:
        raise ValueError(
            f"built eval stores do not match the {EXPECTED_STORES} campaign models"
        )
    built_keys = _eval_store_keys(directory)
    if built_keys != manifest_keys:
        raise ValueError(
            f"built eval keys do not match manifest: {len(built_keys)} vs {len(manifest_keys)}"
        )


def main() -> int:
    rows = merge_parts(MANIFEST, PARTS, MERGED)
    originals = _unique_map(_read_jsonl(ORIGINALS), "originals")
    report = qc_rows(rows, originals, load_lexicon(LEXICON))
    structural = report["failures"]
    scorer = []
    coverage_failure = None
    if not structural:
        EVAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{CONDITION}-", dir=EVAL_DIR.parent) as stage:
            stage_dir = Path(stage)
            scorer = _build_evals(rows, stage_dir)
            if not scorer:
                try:
                    _validate_campaign(stage_dir, {_key(row) for row in rows})
                except ValueError as exc:
                    coverage_failure = str(exc)
                if not coverage_failure:
                    staged = {path.name for path in stage_dir.glob("*.eval")}
                    existing = {path.name for path in EVAL_DIR.glob("*.eval")}
                    if existing - staged:
                        coverage_failure = "production eval directory has unexpected stores"
                    else:
                        EVAL_DIR.mkdir(parents=True, exist_ok=True)
                        for path in stage_dir.glob("*.eval"):
                            path.replace(EVAL_DIR / path.name)
    excluded = structural + scorer

    _write_jsonl_atomic(FEATURES_OUT, report.pop("features"))
    _write_json_atomic(QC_OUT, {
        "condition": CONDITION,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "n_rows": len(rows),
        "n_structural_failures": len(structural),
        "n_scorer_failures": len(scorer),
        "coverage_failure": coverage_failure,
        "structural_failures": structural,
        "scorer_failures": scorer,
        "residual_targeted": report["residual_targeted"],
        "style_features": str(FEATURES_OUT.relative_to(ROOT)),
    })
    _write_json_atomic(EXCLUDED_OUT, {
        "condition": CONDITION,
        "why": "Only structural transcript failures and failures from the real Inspect scorer are excluded.",
        "n_excluded": len(excluded),
        "cells": excluded,
        "coverage_failure": coverage_failure,
    })
    print(f"merged {len(rows)} rows -> {MERGED}")
    print(f"excluded {len(excluded)} rows -> {EXCLUDED_OUT}")
    print(f"canonical style features -> {FEATURES_OUT}")
    print(f"QC, including non-excluding residual target contexts -> {QC_OUT}")
    return 1 if excluded or coverage_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
