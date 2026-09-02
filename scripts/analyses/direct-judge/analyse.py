# scripts/analyses/direct-judge/analyse.py
"""Collect direct-judge logs into scores.csv + picks.csv + summary.csv.

scores.csv: one row per judged transcript (bench, judge, model, item).
picks.csv: induced pair picks — for every pair and item of the baseline
pairwise comparisons, the pick is whichever transcript the same judge scored
higher; equal scores count 0.5. Columns follow the pairwise verdicts.csv where
they overlap so downstream joins are trivial.
summary.csv: per bench x judge rates, plus mean-score-vs-ECI Kendall tau.

Usage:
  uv run python analyse.py                # logs/direct-judge -> judge_results/direct_judge/
  uv run python analyse.py --roots <dir>  # e.g. a smoke-test log root
"""
import argparse
import csv
import functools
from collections import defaultdict
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log
from scipy.stats import kendalltau

from judge.util import extract_transcript, first_correct_by_id
from score_task import parse_score_output

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
OUT_DIR = ROOT / "judge_results" / "direct_judge"

SCORE_FIELDS = ["bench", "judge", "model", "item_id", "score", "confidence",
                "transcript_len", "parse_ok"]
PICK_FIELDS = ["bench", "judge", "major", "minor", "item_id", "score_major",
               "score_minor", "picked_major", "tie", "len_major", "len_minor",
               "picked_longer"]


@functools.lru_cache(maxsize=None)
def _load_config(bench):
    p = HERE / "configs" / f"{bench}.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None


@functools.lru_cache(maxsize=None)
def _transcript_lengths(log_path, include_reasoning, max_length):
    """item -> length of its first correct epoch, the epoch the judge scored."""
    groups = first_correct_by_id(read_eval_log(str(ROOT / log_path)))
    return {item_id: len(extract_transcript(group[0], include_reasoning=include_reasoning)[:max_length])
            for item_id, group in groups.items()}


def rows_from_model_dir(model_dir, bench, judge, cfg):
    """One score row per judged transcript in the model dir's success log."""
    model = model_dir.name
    log = next((read_eval_log(str(p)) for p in model_dir.glob("*.eval")
                if read_eval_log(str(p), header_only=True).status == "success"), None)
    if log is None:
        print(f"  no success log: {model_dir}")
        return
    lengths = _transcript_lengths(cfg["transcripts"][model],
                                  cfg["include_reasoning"], cfg["max_length"])
    for s in log.samples:
        if not s.scores:
            continue  # errored sample (context overflow etc.), no score
        sc = list(s.scores.values())[0]
        score = int(sc.answer) if sc.answer not in (None, "") else None
        if score is None and s.output and s.output.message:
            # small judges sometimes answer inside an unclosed think block; the
            # serving reasoning-parser then leaves the completion empty. The
            # rating is still in the reasoning field - recover it.
            content = s.output.message.content
            reasoning = " ".join(getattr(c, "reasoning", "") or "" for c in content
                                 if getattr(c, "type", "") == "reasoning") \
                if isinstance(content, list) else ""
            if reasoning:
                val, conf, _ = parse_score_output(reasoning)
                if val is not None:
                    score = val
                    if conf is not None and not (sc.metadata or {}).get("confidence"):
                        sc.metadata = dict(sc.metadata or {}, confidence=conf)
        yield dict(bench=bench, judge=judge, model=model, item_id=s.id,
                   score="" if score is None else score,
                   confidence=(sc.metadata or {}).get("confidence", ""),
                   transcript_len=lengths.get(s.id, ""),
                   parse_ok=score is not None)


def collect(root: Path):
    rows = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = _load_config(bench_dir.name)
        if cfg is None:
            print(f"  no config for {bench_dir.name}, skipped")
            continue
        for judge_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
                rows += list(rows_from_model_dir(
                    model_dir, bench_dir.name, judge_dir.name, cfg))
    return rows


def induced_picks(scores, comparisons, pair_sample_ids, lengths=None):
    """Pair picks induced from per-transcript scores of one (bench, judge).

    scores: (model, item_id) -> int. Items missing either score are dropped.
    """
    rows = []
    for major, minor in comparisons:
        for item in pair_sample_ids[f"{major}-vs-{minor}"]:
            a, b = scores.get((major, item)), scores.get((minor, item))
            if a is None or b is None:
                continue
            tie = a == b
            picked_major = 0.5 if tie else float(a > b)
            lm = lengths.get((major, item), "") if lengths else ""
            ln = lengths.get((minor, item), "") if lengths else ""
            picked_longer = ""
            if not tie and lm != "" and ln != "" and lm != ln:
                picked_longer = (a > b) == (lm > ln)
            rows.append(dict(major=major, minor=minor, item_id=item,
                             score_major=a, score_minor=b,
                             picked_major=picked_major, tie=tie,
                             len_major=lm, len_minor=ln,
                             picked_longer=picked_longer))
    return rows


def build_picks(score_rows):
    picks = []
    by_group = defaultdict(dict)
    len_by_group = defaultdict(dict)
    for r in score_rows:
        if r["parse_ok"]:
            by_group[(r["bench"], r["judge"])][(r["model"], r["item_id"])] = r["score"]
            len_by_group[(r["bench"], r["judge"])][(r["model"], r["item_id"])] = r["transcript_len"]
    for (bench, judge), scores in sorted(by_group.items()):
        cfg = _load_config(bench)
        for row in induced_picks(scores, cfg["comparisons"], cfg["pair_sample_ids"],
                                 len_by_group[(bench, judge)]):
            picks.append(dict(bench=bench, judge=judge, **row))
    return picks


def summarize(score_rows, pick_rows):
    """Per bench x judge: parse rate, tie fraction, oriented accuracy, ECI Kendall tau."""
    out = []
    scores_g = defaultdict(list)
    for r in score_rows:
        scores_g[(r["bench"], r["judge"])].append(r)
    picks_g = defaultdict(list)
    for r in pick_rows:
        picks_g[(r["bench"], r["judge"])].append(r)
    for (bench, judge), rs in sorted(scores_g.items()):
        parsed = [r for r in rs if r["parse_ok"]]
        ps = picks_g[(bench, judge)]
        longer = [r["picked_longer"] for r in ps if r["picked_longer"] != ""]
        # model-level: mean score vs ECI
        eci = _load_config(bench)["eci"]
        by_model = defaultdict(list)
        for r in parsed:
            by_model[r["model"]].append(r["score"])
        mean_scores = {m: sum(v) / len(v) for m, v in by_model.items()}
        both = [m for m in mean_scores if m in eci]
        if len(both) < len(mean_scores):
            print(f"  {bench}/{judge}: {len(mean_scores) - len(both)} models "
                  f"missing ECI, dropped from tau")
        tau = ""
        if len(both) >= 3:
            tau = round(kendalltau([mean_scores[m] for m in both],
                                   [eci[m] for m in both]).statistic, 4)
        out.append(dict(
            bench=bench, judge=judge, n_scored=len(rs),
            parse_rate=round(len(parsed) / len(rs), 4) if rs else "",
            n_picks=len(ps),
            tie_frac=round(sum(r["tie"] for r in ps) / len(ps), 4) if ps else "",
            p_pick_stronger=round(sum(r["picked_major"] for r in ps) / len(ps), 4) if ps else "",
            p_pick_longer=round(sum(longer) / len(longer), 4) if longer else "",
            n_models=len(mean_scores),
            kendall_mean_score_vs_eci=tau,
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=[str(ROOT / "logs" / "direct-judge")])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    score_rows = []
    for root in args.roots:
        score_rows += collect(Path(root))
    pick_rows = build_picks(score_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (("scores.csv", score_rows, SCORE_FIELDS),
                               ("picks.csv", pick_rows, PICK_FIELDS)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    summary = summarize(score_rows, pick_rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else
                           ["bench", "judge", "n_scored"])
        w.writeheader()
        w.writerows(summary)
    print(f"{len(score_rows)} score rows, {len(pick_rows)} pick rows -> {out_dir}")
    for s in summary:
        print("  ", s)


if __name__ == "__main__":
    main()
