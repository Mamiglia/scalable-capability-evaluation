"""Bake the gpqa-pilot-rd2 dashboard dataset into interactive_viewer/data/.

Writes the three tiers the static viewer reads:
  index.json                          plot points, model/judge metadata, fits,
                                      colours, GPQA questions
  transcripts/{model}.json            item -> full transcript, deduped across runs
  transcripts/{model}__ep1.json       self-pair second-epoch (slot-B) transcripts
  pairs/{major}__{minor}__{frac}.json item -> judge -> both counterbalanced orders

Points and stats come from the already-computed CSVs (summary/models/judge_fits);
the judge logs supply pick/confidence/explanation and the source model logs the
transcripts. Mirrors digest_verdicts.digest's log loop. Re-run when logs change.

Usage:
  uv run python scripts/analyses/export_dashboard.py
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log
from matplotlib.colors import to_hex

from digest_verdicts import pair_items
from comparisons import JUDGE_ALIASES, MODELS
from figures import JCOLOR, JUDGES, release_year

JUDGE_ONLY = ("gpt-4.1-nano", "qwen-2.5-72b-instruct")


def num(x):
    """float or None for NaN; keeps the JSON valid (no bare NaN tokens)."""
    return None if pd.isna(x) else float(x)


def frac_str(tf):
    return "full" if pd.isna(tf) else str(int(round(tf * 100)))


def build_models(models_csv: pd.DataFrame) -> dict:
    """model -> {label, accuracy, release, arm}; judge-only models from MODELS."""
    out = {}
    for _, r in models_csv.iterrows():
        out[r.model] = {"label": r.label, "accuracy": float(r.accuracy),
                        "release": r.release, "arm": r.arm}
    for m in JUDGE_ONLY:
        meta = MODELS[m]
        out[m] = {"label": meta["label"], "accuracy": float(meta["accuracy"]),
                  "release": meta["release"], "arm": "judge"}
    return out


def build_runs(summary: pd.DataFrame, acc: dict, arm: dict) -> list:
    runs = []
    for _, r in summary.iterrows():
        frac = frac_str(r.truncate_frac)
        is_self = r.major == r.minor
        runs.append(dict(
            run_id=f"{r.judge}__{r.major}__{r.minor}__{frac}",
            judge=r.judge, major=r.major, minor=r.minor, frac=frac,
            d_acc=num(r.d_acc), score=num(r.score),
            score_lo=num(r.score_lo), score_hi=num(r.score_hi),
            n_items=int(r.n_items), slot_a=num(r.slot_a),
            disagree_rate=num(r.disagree_rate), noanswer_rate=num(r.noanswer_rate),
            is_self=bool(is_self),
            weak_judge=bool(acc[r.judge] < min(acc[r.major], acc[r.minor])),
            old_judge=bool(release_year(r.judge)
                           < min(release_year(r.major), release_year(r.minor))),
            arm_a=bool(arm[r.major] == "A" and arm[r.minor] == "A"),
        ))
    return runs


def collect_logs(log_root: Path, config: Path):
    """Mirror digest's loop: return (transcripts, questions, verdict_rows).

    transcripts: model -> {item: text}; self-pair minor side under {model}__ep1.
    verdict_rows: (judge, major, minor, item, slot, frac) -> order dict, newer
    logs overwriting older (filename-sorted ascending).
    """
    cfg = yaml.safe_load(config.read_text())
    by_path = {Path(p).resolve(): name for name, p in cfg["transcripts"].items()}

    def judge_short(j):
        s = j.split("/")[-1]
        return JUDGE_ALIASES.get(s, s)

    configured = {(judge_short(j), maj, mnr) for j, maj, mnr in cfg["comparisons"]}

    transcripts: dict[str, dict] = {}
    questions: dict[str, str] = {}
    verdict_rows: dict[tuple, dict] = {}
    selection: set[tuple] = set()

    for f in sorted(log_root.rglob("*.eval")):
        log = read_eval_log(str(f))
        if log.status != "success":
            continue
        meta = log.eval.metadata or {}
        judge_raw = log.eval.model.split("/")[-1]
        judge = JUDGE_ALIASES.get(judge_raw, judge_raw)
        major = by_path.get(Path(meta["major_path"]).resolve())
        minor = by_path.get(Path(meta["minor_path"]).resolve())
        if major is None or minor is None:
            continue
        if (judge, major, minor) not in configured:
            continue

        ir, rbc = meta["include_reasoning"], meta["require_both_correct"]
        selection.add((ir, rbc))
        self_pair = major == minor
        frac = frac_str(meta.get("truncate_frac"))

        items = pair_items(meta["major_path"], meta["minor_path"], ir, rbc,
                           truncate_frac=None)
        minor_key = f"{minor}__ep1" if self_pair else minor
        for item, (question, t_major, t_minor, _, _) in items.items():
            transcripts.setdefault(major, {})[item] = t_major
            transcripts.setdefault(minor_key, {})[item] = t_minor
            questions[item] = question

        for s in log.samples or []:
            item, slot = str(s.id).rsplit("_", 1)
            score = next(iter((s.scores or {}).values()), None)
            pick = getattr(score, "answer", None)
            if self_pair:
                picked_major = None
            elif pick == slot:
                picked_major = 1
            elif pick in ("1", "2"):
                picked_major = 0
            else:
                picked_major = None
            conf = (score.metadata or {}).get("confidence") if score and score.metadata else None
            expl = getattr(score, "explanation", None) if score else None
            verdict_rows[(judge, major, minor, item, slot, frac)] = dict(
                slot_of_major=slot, pick=pick, picked_major=picked_major,
                confidence=conf, explanation=expl)

    # Transcript dedup keys on (model, item) only, so a model's judged text must
    # not depend on the run's selection flags. Constant today; assert to catch a
    # future config that varies them (then the key would need (ir, rbc) too).
    assert len(selection) <= 1, \
        f"include_reasoning/require_both_correct vary across runs: {selection}"
    return transcripts, questions, verdict_rows


def assemble_pairs(verdict_rows: dict) -> dict:
    """(major, minor, frac) -> {item: {judge: {orders: [A, B]}}}."""
    pairs: dict[tuple, dict] = {}
    for (judge, major, minor, item, _slot, frac), order in verdict_rows.items():
        item_map = pairs.setdefault((major, minor, frac), {}).setdefault(item, {})
        item_map.setdefault(judge, {"orders": []})["orders"].append(order)
    for item_map in pairs.values():
        for judges in item_map.values():
            for jd in judges.values():
                jd["orders"].sort(key=lambda o: o["slot_of_major"])
    return pairs


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/judge_pilot.yaml"))
    ap.add_argument("--log-root", type=Path, default=Path("logs/gpqa-pilot-rd2"))
    ap.add_argument("--results", type=Path, default=Path("judge_results/gpqa-pilot-rd2"))
    ap.add_argument("--out", type=Path, default=Path("interactive_viewer/data"))
    args = ap.parse_args()

    summary = pd.read_csv(args.results / "summary.csv")
    models_csv = pd.read_csv(args.results / "models.csv")
    fits_csv = pd.read_csv(args.results / "judge_fits.csv")

    models = build_models(models_csv)
    acc = {m: v["accuracy"] for m, v in models.items()}
    arm = {m: v["arm"] for m, v in models.items()}

    present = set(summary.judge)
    judges = [j for j in JUDGES if j in present]
    colors = {j: to_hex(JCOLOR[j]) for j in judges}
    fits = {r.judge: {"a": float(r.a), "d": float(r.d),
                      "acc_at_gap0": float(r.acc_at_gap0)}
            for _, r in fits_csv.iterrows()}

    transcripts, questions, verdict_rows = collect_logs(args.log_root, args.config)
    pairs = assemble_pairs(verdict_rows)

    index = dict(models=models, judges=judges, colors=colors, questions=questions,
                 fits=fits, runs=build_runs(summary, acc, arm))

    # Clear stale exports, keeping the viewer's html/js/css/plotly untouched.
    for sub in ("transcripts", "pairs"):
        if (args.out / sub).exists():
            shutil.rmtree(args.out / sub)
    (args.out / "index.json").unlink(missing_ok=True)
    (args.out / "transcripts").mkdir(parents=True, exist_ok=True)
    (args.out / "pairs").mkdir(parents=True, exist_ok=True)

    dump(index, args.out / "index.json")
    for model, items in transcripts.items():
        dump(items, args.out / "transcripts" / f"{model}.json")
    for (major, minor, frac), data in pairs.items():
        dump(data, args.out / "pairs" / f"{major}__{minor}__{frac}.json")

    print(f"index.json: {len(index['runs'])} runs, {len(questions)} questions, "
          f"{len(models)} models, {len(judges)} judges")
    print(f"transcripts/: {len(transcripts)} files")
    print(f"pairs/: {len(pairs)} files")


if __name__ == "__main__":
    main()
