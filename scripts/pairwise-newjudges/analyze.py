"""Convert new-judge pairwise logs -> verdict rows (sidecar to judging_variations/verdicts.csv).

Layout here is bench/judge/pair (no condition dir); all rows are condition a_baseline.
Row semantics + schema are copied verbatim from judging-variations analyze.py so these
rows concatenate cleanly with the 6 canonical judges. Writes a SIDECAR csv (not the
canonical verdicts.csv) so the analysis session's regen of that file won't wipe these.
Manipulation mode defaults to its condition/judge/pair root and writes only after the
complete three-judge campaign grid has been collected.

Usage:
  uv run python scripts/pairwise-newjudges/analyze.py            # ling only -> verdicts_newjudges.csv
  uv run python scripts/pairwise-newjudges/analyze.py --judges ling-2.6-flash
  uv run python scripts/pairwise-newjudges/analyze.py --conditions style_standardised_gpt
"""
import argparse
import csv
import functools
import os
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inspect_ai.log import read_eval_log
from judge.util import extract_transcript, first_correct_by_id
from judge.scorer import parse_judge_output

FIELDS = ["bench", "condition", "judge", "major", "minor", "item_id", "order",
          "picked", "picked_major", "confidence", "len_major", "len_minor",
          "picked_longer", "parse_ok"]
CONDITION = "a_baseline"
CAMPAIGN_JUDGES = ("qwen3-8b", "qwen3-32b", "gpt-oss-20b")
MANIPULATION_LOG_ROOT = ROOT / "logs" / "pairwise-newjudges" / "manipulations"
STYLE_REWRITE_VERDICTS = (
    ROOT / "judge_results" / "robustness" / "style_standardised_gpt"
    / "verdicts_rewrite.csv"
)


@functools.lru_cache(maxsize=None)
def _load_config(bench):
    return yaml.safe_load((HERE / "configs" / f"{bench}.yaml").read_text())


@functools.lru_cache(maxsize=None)
def _transcript_lengths(log_path, include_reasoning, max_length, epoch_slot=0):
    """item -> length of the transcript the judge saw.

    epoch_slot picks which correct epoch, as build_dataset does: 0 for both sides
    of a normal pair, 1 for the minor side of a self-pair. Items without that many
    correct epochs are absent, matching the pairs build_dataset skips.
    """
    groups = first_correct_by_id(read_eval_log(str(ROOT / log_path)))
    return {item_id: len(extract_transcript(group[epoch_slot],
                                            include_reasoning=include_reasoning)[:max_length])
            for item_id, group in groups.items() if len(group) > epoch_slot}


def rows_from_pair_dir(pair_dir, bench, judge, cfg, condition=CONDITION):
    major, minor = pair_dir.name.split("-vs-")
    log = next((read_eval_log(str(p)) for p in pair_dir.glob("*.eval")
                if read_eval_log(str(p), header_only=True).status == "success"), None)
    if log is None:
        print(f"  no success log: {pair_dir}")
        return
    invert = bool(cfg.get("invert_at_analysis"))
    len_major = _transcript_lengths(cfg["transcripts"][major],
                                    cfg["include_reasoning"], cfg["max_length"])
    # a self-pair judges one model's two earliest correct epochs against each other
    len_minor = _transcript_lengths(cfg["transcripts"][minor],
                                    cfg["include_reasoning"], cfg["max_length"],
                                    1 if minor == major else 0)
    for s in log.samples:
        if not s.scores:
            continue  # errored sample, no verdict
        item_id, target = s.id.rsplit("_", 1)
        sc = list(s.scores.values())[0]
        picked = sc.answer if sc.answer in ("1", "2") else ""
        if not picked and s.output and s.output.message:
            # verdict may sit in an unclosed think block -> empty completion; recover it
            content = s.output.message.content
            reasoning = " ".join(getattr(c, "reasoning", "") or "" for c in content
                                 if getattr(c, "type", "") == "reasoning") \
                if isinstance(content, list) else ""
            if reasoning:
                ans, conf, _ = parse_judge_output(reasoning)
                if ans in ("1", "2"):
                    picked = ans
                    if conf is not None and not (sc.metadata or {}).get("confidence"):
                        sc.metadata = dict(sc.metadata or {}, confidence=conf)
        picked_major = ""
        if picked:
            picked_major = (picked == target)
            if invert:
                picked_major = not picked_major
        lm, ln = len_major.get(item_id), len_minor.get(item_id)
        picked_longer = ""
        if picked and lm is not None and ln is not None and lm != ln:
            longer_pos = target if lm > ln else ("2" if target == "1" else "1")
            picked_longer = (picked == longer_pos)
        yield dict(bench=bench, condition=condition, judge=judge, major=major,
                   minor=minor, item_id=item_id, order=target, picked=picked,
                   picked_major=picked_major,
                   confidence=(sc.metadata or {}).get("confidence", ""),
                   len_major=lm, len_minor=ln, picked_longer=picked_longer,
                   parse_ok=bool(picked))


def collect(root, judges):
    rows = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = _load_config(bench_dir.name)
        for judge_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            if judge_dir.name not in judges:
                print(f"  skip judge {bench_dir.name}/{judge_dir.name} (not in --judges)")
                continue
            for pair_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
                rows += list(rows_from_pair_dir(pair_dir, bench_dir.name, judge_dir.name, cfg))
    return rows


def collect_manipulations(root, configs, conditions, judges):
    """Collect condition/judge/pair manipulation logs in verdicts.csv schema."""
    rows = []
    for condition in conditions:
        cfg = yaml.safe_load((configs / f"{condition}.yaml").read_text())
        if cfg["condition"] != condition:
            raise ValueError(f"condition mismatch in {condition}.yaml")
        condition_dir = root / condition
        if not condition_dir.exists():
            continue
        for judge_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
            if judge_dir.name not in judges:
                continue
            for pair_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
                rows += list(rows_from_pair_dir(
                    pair_dir, "gpqa", judge_dir.name, cfg, condition=condition
                ))
    return rows


def write_manipulation_verdicts(rows, config, out):
    """Atomically write only one exact 50-pair campaign grid."""
    pairs = [f"{major}-vs-{minor}" for major, minor in config["comparisons"]]
    sample_ids = config["sample_ids"]
    valid_config = (
        len(pairs) == 50
        and len(sample_ids) == 50
        and set(pairs) == set(sample_ids)
        and all(len(items) == 30 and len(set(items)) == 30
                for items in sample_ids.values())
    )
    if not valid_config:
        raise ValueError("Manipulation config must contain exactly 50 pairs × 30 unique items")

    expected = {
        ("gpqa", config["condition"], judge, major, minor, str(item), str(order))
        for judge in CAMPAIGN_JUDGES
        for major, minor in config["comparisons"]
        for item in sample_ids[f"{major}-vs-{minor}"]
        for order in (1, 2)
    }
    actual = [
        (row.get("bench"), row.get("condition"), row.get("judge"),
         row.get("major"), row.get("minor"), str(row.get("item_id")),
         str(row.get("order")))
        for row in rows
    ]
    unique = set(actual)
    missing, extra = expected - unique, unique - expected
    duplicates = len(actual) - len(unique)
    if len(expected) != 9000 or len(actual) != 9000 or missing or extra or duplicates:
        raise ValueError(
            "Refusing to write manipulation verdicts: expected exactly 9,000 unique rows "
            f"(got {len(actual)} rows/{len(unique)} unique; {len(missing)} missing, "
            f"{len(extra)} extra, {duplicates} duplicate)"
        )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", dir=out.parent, prefix=f".{out.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root")
    ap.add_argument("--configs", type=Path,
                    default=HERE / "configs" / "manipulations")
    ap.add_argument("--conditions", nargs="+",
                    help="collect manipulation logs laid out as condition/judge/pair")
    ap.add_argument("--judges", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.conditions:
        if len(args.conditions) != 1:
            raise ValueError("Manipulation collection requires exactly one condition")
        condition = args.conditions[0]
        root = Path(args.root) if args.root else MANIPULATION_LOG_ROOT
        judges = set(args.judges or CAMPAIGN_JUDGES)
        out = Path(args.out) if args.out else (
            STYLE_REWRITE_VERDICTS if condition == "style_standardised_gpt"
            else ROOT / "judge_results" / "robustness" / condition / "verdicts_rewrite.csv"
        )
        rows = collect_manipulations(
            root, args.configs, args.conditions, judges
        )
        config = yaml.safe_load((args.configs / f"{condition}.yaml").read_text())
        write_manipulation_verdicts(rows, config, out)
    else:
        root = Path(args.root) if args.root else ROOT / "logs" / "pairwise-newjudges"
        judges = set(args.judges or ["ling-2.6-flash"])
        out = Path(args.out) if args.out else (
            ROOT / "judge_results" / "judging_variations" / "verdicts_newjudges.csv"
        )
        rows = collect(root, judges)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    # quick per bench x judge sanity readout
    from collections import Counter
    n = Counter((r["bench"], r["judge"]) for r in rows)
    parsed = Counter((r["bench"], r["judge"]) for r in rows if r["parse_ok"])
    print(f"{len(rows)} rows -> {out}")
    for k in sorted(n):
        print(f"  {k}: {n[k]} rows, parse_rate={parsed[k]/n[k]:.3f}")


if __name__ == "__main__":
    main()
