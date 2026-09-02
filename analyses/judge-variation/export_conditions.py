"""Export judging-variation conditions as transcript-viewer datasets.

One dataset per (bench, condition):
interactive_dashboards/jv/data/jv-<bench>-<letter>/
index.json, pairs/ (per-item verdicts with the judge's explanation) and
transcripts/ (member text exactly as that condition's judge saw it -
include_reasoning and truncation from the condition config), in the same shape
scripts/judge/export_viewer.py emits. Datasets are registered in
data/datasets.json so the viewer lists them as projects. Built from
judge_results/judging_variations/verdicts.csv (pairs there are already
stronger-first by ECI) plus the raw logs under logs/judging-variations/.

Usage: uv run python export_conditions.py
"""
import csv
import functools
import json
from collections import defaultdict
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log
from scipy.stats import beta as beta_dist

from judge.util import extract_transcript
from pool import load_members

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT = ROOT / "interactive_dashboards" / "jv" / "data"
VERDICTS = ROOT / "judge_results" / "judging_variations" / "verdicts.csv"
LOG_ROOT = ROOT / "logs" / "judging-variations"

COND_PLAIN = {"baseline": "final text only", "cot": "reasoning shown, long context",
              "cot-native": "reasoning shown, 32k context",
              "ignore-style": "ignore superficial style",
              "ignore-style-conf": "ignore style + confidence",
              "pick-weaker": "pick weaker (inverted)"}
BENCH_PLAIN = {"gpqa": "GPQA", "otis-aime": "OTIS AIME"}


@functools.lru_cache(maxsize=None)
def _cfg(bench, cond):
    return yaml.safe_load((HERE / "configs" / f"{bench}_{cond.removesuffix('_native')}.yaml").read_text())


def _question(sample):
    x = sample.input
    if isinstance(x, str):
        return x
    try:
        c = x[0].content
        return c if isinstance(c, str) else " ".join(getattr(p, "text", "") for p in c)
    except Exception:
        return ""


def export_payloads(bench, cond, out):
    """pairs/ + transcripts/ + questions from the raw logs, viewer shapes."""
    cfg = _cfg(bench, cond)
    invert = bool(cfg.get("invert_at_analysis"))
    (out / "pairs").mkdir(exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    used_models = set()
    for judge_dir in sorted(p for p in (LOG_ROOT / bench / cond).iterdir() if p.is_dir()):
        for pair_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
            log = next((read_eval_log(str(p)) for p in pair_dir.glob("*.eval")
                        if read_eval_log(str(p), header_only=True).status == "success"), None)
            if log is None:
                continue
            major, minor = pair_dir.name.split("-vs-")
            used_models |= {major, minor}
            fp = out / "pairs" / f"{major}__{minor}__full.json"
            payload = json.load(open(fp)) if fp.exists() else {}
            for smp in log.samples:
                sc = list(smp.scores.values())[0] if smp.scores else None
                if sc is None:
                    continue
                item, order = str(smp.id).rsplit("_", 1)
                ans = sc.answer if sc.answer in ("1", "2") else None
                pm = None if ans is None else int((ans == smp.target) != invert)
                slot = payload.setdefault(item, {}).setdefault(
                    judge_dir.name, {"orders": [None, None]})
                slot["orders"][int(order) - 1] = {
                    "slot_of_major": smp.target, "pick": ans, "picked_major": pm,
                    "confidence": (sc.metadata or {}).get("confidence"),
                    "explanation": sc.explanation}
            json.dump(payload, open(fp, "w"))
    questions = {}
    for model in sorted(used_models):
        log = read_eval_log(str(ROOT / cfg["transcripts"][model]))
        json.dump({smp.id: extract_transcript(smp, include_reasoning=cfg["include_reasoning"])
                   [:cfg["max_length"]] for smp in log.samples},
                  open(out / "transcripts" / f"{model}.json", "w"))
        if not questions:
            questions = {smp.id: _question(smp) for smp in log.samples}
    return questions

COND_LETTER = {"a_baseline": "baseline", "b_ignore_superficial": "ignore-style",
               "c_ignore_superficial_confidence": "ignore-style-conf",
               "d_pick_weaker": "pick-weaker",
               "e_with_reasoning": "cot", "e_with_reasoning_native": "cot-native"}


def jeffreys(c, n):
    if n == 0:
        return None, None
    return (float(beta_dist.ppf(0.025, c + 0.5, n - c + 0.5)),
            float(beta_dist.ppf(0.975, c + 0.5, n - c + 0.5)))


def bench_models(bench):
    """short -> viewer model meta (accuracy from the epoch log, eci/release from pool)."""
    from pool import match_name, _correct_ids
    from inspect_ai.log import read_eval_log
    pool = [m for m in json.load(open(ROOT / "model-sampler" / "model_pool.json"))
            if m["eci"] is not None]
    members = load_members(ROOT / "external_data" / f"epoch-{bench}",
                           ROOT / "model-sampler" / "model_pool.json")
    meta = {}
    for m in members:
        rec = match_name(pool, m["short"]) or {}
        log = read_eval_log(str(m["path"]))
        n = len(log.samples)
        meta[m["short"]] = {"label": m["short"], "eci": m["eci"],
                            "release": rec.get("release"), "arm": "A",
                            "accuracy": len(_correct_ids(m["path"])) / n if n else None}
    return meta


def main():
    rows = list(csv.DictReader(open(VERDICTS)))
    datasets = defaultdict(list)
    for r in rows:
        if r["condition"] == "attribution":
            continue
        datasets[(r["bench"], r["condition"])].append(r)

    model_meta = {bench: bench_models(bench) for bench in {b for b, _ in datasets}}

    for (bench, cond), rs in sorted(datasets.items()):
        name = f"jv-{bench}-{COND_LETTER.get(cond, cond)}"
        used = {r["major"] for r in rs} | {r["minor"] for r in rs}
        models = {m: v for m, v in model_meta[bench].items() if m in used}
        runs = []
        grp = defaultdict(list)
        for r in rs:
            grp[(r["judge"], r["major"], r["minor"])].append(r)
        for (judge, major, minor), g in sorted(grp.items()):
            ok = [r for r in g if r["parse_ok"] == "True"]
            n_ok = len(ok)
            n_c = sum(r["picked_major"] == "True" for r in ok)
            lo, hi = jeffreys(n_c, n_ok)
            per_item = defaultdict(dict)
            for r in ok:
                per_item[r["item_id"]][r["order"]] = r["picked_major"] == "True"
            both = [v for v in per_item.values() if len(v) == 2]
            disagree = [v for v in both if v["1"] != v["2"]]
            acc = lambda m: models.get(m, {}).get("accuracy") or 0
            runs.append({
                "judge": judge, "major": major, "minor": minor,
                "d_acc": abs(acc(major) - acc(minor)),
                "score": (n_c / n_ok) if n_ok else None,
                "score_lo": lo, "score_hi": hi,
                "n_items": len(per_item),
                "slot_a": (sum(r["picked"] == "1" for r in ok) / n_ok) if n_ok else None,
                "disagree_rate": len(disagree) / max(len(both), 1),
                "noanswer_rate": 1 - n_ok / max(len(g), 1),
                "is_self": major == minor, "weak_judge": False, "old_judge": False,
                "arm_a": True, "frac": "full",
            })
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        longer = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in rs:
            if r["parse_ok"] == "True" and r["picked_longer"] != "":
                cell = longer[r["judge"]][f"{r['major']}-vs-{r['minor']}"]
                cell[0] += r["picked_longer"] == "True"
                cell[1] += 1
        json.dump({j: dict(v) for j, v in longer.items()},
                  open(out / "lengths.json", "w"))
        questions = export_payloads(bench, cond, out)
        json.dump({"models": models, "judges": sorted({r["judge"] for r in runs}),
                   "colors": {}, "questions": questions, "fits": {}, "runs": runs},
                  open(out / "index.json", "w"))
        print(f"{name}: {len(runs)} runs, {len(models)} models")

    reg_path = OUT / "datasets.json"
    reg = json.load(open(reg_path))
    have = {d["id"] for d in reg}
    for (bench, cond) in sorted(datasets):
        name = f"jv-{bench}-{COND_LETTER.get(cond, cond)}"
        if name not in have:
            letter = COND_LETTER.get(cond, cond)
            reg.append({"id": name,
                        "label": f"{BENCH_PLAIN.get(bench, bench)} variations - "
                                 f"{COND_PLAIN.get(letter, letter)}"})
            have.add(name)
    json.dump(reg, open(reg_path, "w"))
    print(f"datasets.json: {len(reg)} projects")


if __name__ == "__main__":
    main()
