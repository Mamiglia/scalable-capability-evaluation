"""API reasoning-ablation analysis: with-CoT vs baseline judging, cheap-6 judges.

Walks the finished API-judge logs (both arms), emits rows in analyze.py's
verdicts schema with condition api_baseline / api_with_reasoning, restricted to
pairs judged in BOTH arms, then reuses analyze.summarize.

Pair-dir naming in these logs is mixed vs capability; rows are oriented so
major = higher-ECI member (same convention the viewer export uses).

Usage: uv run python ablation_api.py
"""
import csv
import functools
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log

from analyze import ROOT, OUT_DIR, FIELDS, summarize
from judge.util import extract_score, extract_transcript

# ECIs per configs/comparisons/model_set.json (qwen3-32b from the epoch import)
ECI = {"qwen3-8b": 127.9, "qwen3-32b": 130.0, "qwen3-vl-30b-a3b-instruct": 131.27,
       "glm-4.6": 140.86, "ling-2.6-flash": 131.8, "deepseek-v4-flash-0731": 150.22}

ARMS = {
    "api_baseline": dict(root="logs/data-expansion", include_reasoning=False,
                         config="configs/judge_cheap6_{bench}.yaml"),
    "api_with_reasoning": dict(root="logs/data-expansion-reasoning-ablation",
                               include_reasoning=True,
                               config="configs/judge_reasoning_ablation_{bench}.yaml"),
}
BENCHES = ["aime2025", "aime2026"]
MAX_LENGTH = 65536


@functools.lru_cache(maxsize=None)
def _lengths(log_path, include_reasoning):
    """item -> length of its first correct epoch, the epoch the judge saw."""
    log = read_eval_log(str(ROOT / log_path))
    grouped = {}
    for s in log.samples:
        if extract_score(s):
            grouped.setdefault(s.id, []).append(s)
    for group in grouped.values():
        group.sort(key=lambda s: getattr(s, "epoch", 0))
    return {item_id: len(extract_transcript(group[0], include_reasoning=include_reasoning)[:MAX_LENGTH])
            for item_id, group in grouped.items()}


def success_samples(pair_dir):
    """All samples from success logs under the pair dir (phase subdirs included),
    deduped by sample id."""
    seen = {}
    for p in sorted(pair_dir.rglob("*.eval")):
        try:
            if read_eval_log(str(p), header_only=True).status != "success":
                continue
        except Exception:
            continue
        for s in read_eval_log(str(p)).samples:
            seen.setdefault(s.id, s)
    return list(seen.values())


def pair_rows(pair_dir, bench, condition, judge, transcripts, include_reasoning):
    first, second = pair_dir.name.split("-vs-")
    if first not in ECI or second not in ECI:
        return
    flipped = ECI[second] > ECI[first]
    major, minor = (second, first) if flipped else (first, second)
    len_major = _lengths(transcripts[major], include_reasoning)
    len_minor = _lengths(transcripts[minor], include_reasoning)
    for s in success_samples(pair_dir):
        item_id, target = s.id.rsplit("_", 1)
        sc = list(s.scores.values())[0]
        picked = sc.answer if sc.answer in ("1", "2") else ""
        picked_major = ""
        if picked:
            picked_first = picked == target
            picked_major = (not picked_first) if flipped else picked_first
        lm, ln = len_major.get(item_id), len_minor.get(item_id)
        picked_longer = ""
        if picked and lm is not None and ln is not None and lm != ln:
            # target marks FIRST-NAMED's position; map to major's position
            major_pos = target if not flipped else ("2" if target == "1" else "1")
            longer_pos = major_pos if lm > ln else ("2" if major_pos == "1" else "1")
            picked_longer = (picked == longer_pos)
        yield dict(bench=bench, condition=condition, judge=judge, major=major,
                   minor=minor, item_id=item_id, order=target, picked=picked,
                   picked_major=picked_major,
                   confidence=(sc.metadata or {}).get("confidence", ""),
                   len_major=lm, len_minor=ln, picked_longer=picked_longer,
                   parse_ok=bool(picked))


def main():
    # pair dirs present in both arms, keyed (bench, judge, unordered pair)
    dirs = {}
    for cond, arm in ARMS.items():
        for bench in BENCHES:
            root = ROOT / arm["root"] / bench
            for judge_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for pair_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
                    key = (bench, judge_dir.name, frozenset(pair_dir.name.split("-vs-")))
                    dirs.setdefault(key, {})[cond] = pair_dir
    both = {k: v for k, v in dirs.items() if len(v) == 2}
    print(f"pairs judged in both arms: {len(both)} (of {len(dirs)} seen)")

    rows = []
    for cond, arm in ARMS.items():
        for bench in BENCHES:
            cfg = yaml.safe_load((ROOT / arm["config"].format(bench=bench)).read_text())
            for (b, judge, _), v in sorted(both.items()):
                if b != bench:
                    continue
                rows += list(pair_rows(v[cond], bench, cond, judge,
                                       cfg["transcripts"], arm["include_reasoning"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cot_inclusion" / "cheap_judges_aime_verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(OUT_DIR / "cot_inclusion" / "cheap_judges_aime_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"{len(rows)} rows -> {OUT_DIR}/cot_inclusion/cheap_judges_aime_verdicts.csv")
    for s in summary:
        print("  ", s)


if __name__ == "__main__":
    main()
