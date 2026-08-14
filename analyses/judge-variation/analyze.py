"""Collect judging-variation logs into verdicts.csv + summary.csv.

Sample semantics (from src/judge): dataset id <item>_<target>; target = which
transcript position holds MAJOR (1 normal, 2 reversed). Scorer answer = the
position the judge picked. picked_major = (answer == target). Condition d asks
for the WEAKER model, so picked_major is inverted there (raw pick kept in
`picked`). NOANSWER rows keep picked empty and are excluded from rates.

Lengths are recomputed from member logs with the condition's include_reasoning
and max_length, so picked_longer reflects what the judge actually saw.

Usage:
  uv run python analyze.py                # logs/judging-variations -> judge_results/judging_variations/
  uv run python analyze.py --roots logs/judging-variations
"""
import argparse
import csv
import functools
from collections import defaultdict
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log

from judge.util import extract_transcript

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT_DIR = ROOT / "judge_results" / "judging_variations"

FIELDS = ["bench", "condition", "judge", "major", "minor", "item_id", "order",
          "picked", "picked_major", "confidence", "len_major", "len_minor",
          "picked_longer", "parse_ok"]


@functools.lru_cache(maxsize=None)
def _load_config(bench, condition):
    p = HERE / "configs" / f"{bench}_{condition}.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None


@functools.lru_cache(maxsize=None)
def _transcript_lengths(log_path, include_reasoning, max_length):
    log = read_eval_log(str(ROOT / log_path))
    return {s.id: len(extract_transcript(s, include_reasoning=include_reasoning)[:max_length])
            for s in log.samples}


def rows_from_pair_dir(pair_dir, bench, condition, judge, cfg):
    """One verdict row per judged sample in the pair dir's success log."""
    major, minor = pair_dir.name.split("-vs-")
    log = next((read_eval_log(str(p)) for p in pair_dir.glob("*.eval")
                if read_eval_log(str(p), header_only=True).status == "success"), None)
    if log is None:
        print(f"  no success log: {pair_dir}")
        return
    invert = bool(cfg.get("invert_at_analysis"))
    len_major = _transcript_lengths(cfg["transcripts"][major],
                                    cfg["include_reasoning"], cfg["max_length"])
    len_minor = _transcript_lengths(cfg["transcripts"][minor],
                                    cfg["include_reasoning"], cfg["max_length"])
    from judge.scorer import parse_judge_output
    for s in log.samples:
        if not s.scores:
            continue  # errored sample (context overflow etc.), no verdict
        item_id, target = s.id.rsplit("_", 1)
        sc = list(s.scores.values())[0]
        picked = sc.answer if sc.answer in ("1", "2") else ""
        if not picked and s.output and s.output.message:
            # small judges sometimes answer inside an unclosed think block; the
            # serving reasoning-parser then leaves the completion empty. The
            # verdict is still in the reasoning field - recover it.
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


def collect(root: Path):
    rows = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cond_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            if cond_dir.name == "attribution":
                continue
            cond = cond_dir.name.removesuffix("_native")
            cfg = _load_config(bench_dir.name, cond)
            if cfg is None:
                print(f"  no config for {bench_dir.name}/{cond_dir.name}, skipped")
                continue
            for judge_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                for pair_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
                    rows += list(rows_from_pair_dir(
                        pair_dir, bench_dir.name, cond_dir.name, judge_dir.name, cfg))
    return rows


def summarize(rows):
    """Per bench x condition x judge: n, parse rate, and the three rates."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["bench"], r["condition"], r["judge"])].append(r)
    out = []
    for (bench, cond, judge), rs in sorted(groups.items()):
        parsed = [r for r in rs if r["parse_ok"]]
        def rate(key):
            vals = [r[key] for r in parsed if r[key] != ""]
            return round(sum(vals) / len(vals), 4) if vals else ""
        out.append(dict(
            bench=bench, condition=cond, judge=judge, n=len(rs),
            parse_rate=round(len(parsed) / len(rs), 4) if rs else "",
            p_pick_stronger=rate("picked_major"),
            p_pick_longer=rate("picked_longer"),
            p_pick_pos1=round(sum(r["picked"] == "1" for r in parsed) / len(parsed), 4) if parsed else "",
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=[str(ROOT / "logs" / "judging-variations")])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        rows += collect(Path(root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else
                           ["bench", "condition", "judge", "n"])
        w.writeheader()
        w.writerows(summary)
    print(f"{len(rows)} verdict rows -> {out_dir}/verdicts.csv")
    for s in summary:
        print("  ", s)


if __name__ == "__main__":
    main()
