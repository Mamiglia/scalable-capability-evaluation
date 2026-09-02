"""Export the SWE-bench-Mini and Intercode-CTF pilot judging to the transcript viewer.

Reads the same judge logs as swe_intercode_gap_curves.py plus the benchmark
rollouts, and writes one viewer dataset per bench (index.json, transcripts/,
pairs/) into interactive_dashboards/transcript_viewer/data/, registering both
in datasets.json. Items carry an __ep<n> suffix so both rollout epochs share
one dataset. Pairs are oriented strong-first by pooled solve rate; ties keep
the judging config's order.

Usage:
  uv run python scripts/judge/swe_intercode_viewer_export.py
"""

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import value_to_float

from configs.paths import is_successful_eval
from judge.util import extract_transcript
from swe_intercode_figures import BENCHES, ROOT, judging_logs, load_solve_rates

DATA_ROOT = ROOT / "interactive_dashboards" / "transcript_viewer" / "data"
HF_ROUND_SCORES = ROOT / "data" / "hf-judging-2026-08-17" / "benchmark-scores"
ECI_TABLE = ROOT / "external_data" / "epoch-eci" / "eci_scores.csv"

DATASETS = {  # bench -> (dataset id, dropdown label)
    "swe_bench_mini": ("swe-mini-pilot", "SWE-bench Mini pilot"),
    "intercode": ("intercode-pilot", "Intercode CTF pilot"),
}

COLORS = {
    "deepseek-r1-0528": "#2a78d6", "deepseek-v4-flash-0731": "#eb6834",
    "gemma-4-31b-it": "#1baf7a", "gpt-oss-120b": "#eda100",
    "ling-2.6-flash": "#e87ba4", "o4-mini": "#4a3aa7",
    "Llama-3.1-8B-Instruct": "#008300",
}

# Epoch Capabilities Index, keyed by the published table's display names.
# Members absent from it get eci: null; the viewer then falls back to the
# accuracy gap for their pairs on the ECI axis.
ECI_NAME = {
    "claude-opus-4.7": "Claude Opus 4.7",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek-r1-0528": "DeepSeek-R1 (May 2025)",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash 0731",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemma-4-31b-it": "Gemma 4 31B IT",
    "glm-4.6": "GLM-4.6",
    "glm-5.2": "GLM-5.2",
    "gpt-4.1": "GPT-4.1",
    "gpt-5": "GPT-5",
    "gpt-oss-120b": "gpt-oss-120b",
    "kimi-k2.6": "Kimi K2.6",
    "kimi-k3": "Kimi K3",
    "Llama-3.1-8B-Instruct": "Llama 3.1-8B",
    "mistral-medium-3": "Mistral Medium 3",
    "nvidia-nemotron-3-ultra-550b-a55b-bf16": "Nemotron 3 Ultra",
    "o3": "o3",
    "o4-mini": "o4-mini",
}
# Not in the published table; values from configs/comparisons/model_set.json
# and the data-expansion export.
ECI_EXTRA = {"ling-2.6-flash": 131.8, "qwen3-32b": 130.0}

_score_to_float = value_to_float()


def solved(sample) -> bool:
    """CORRECT/"C" scorers store the string, SWE-bench a float, HLE a dict."""
    for score in (sample.scores or {}).values():
        value = score.value
        if isinstance(value, dict):
            value = value.get("score", 0)
        if _score_to_float(value) >= 1.0:
            return True
    return False


def load_eci() -> dict:
    table = {r["Display name"]: float(r["eci"])
             for r in csv.DictReader(ECI_TABLE.open()) if r.get("eci")}
    return ECI_EXTRA | {short: table[name] for short, name in ECI_NAME.items() if name in table}


def load_hf_round_scores(root=HF_ROUND_SCORES) -> dict:
    """(BENCH, model) -> accuracy, from the 2026-08-17 HF round's score bundles."""
    totals = defaultdict(lambda: [0.0, 0])
    for path in root.glob("*/*/*/*/logs.json"):
        bench, _, model, _ = path.relative_to(root).parts[:4]
        for log in json.loads(path.read_text()).values():
            if log.get("status") != "success" or log.get("invalidated") or "results" not in log:
                continue
            score = next(s for s in log["results"]["scores"] if "accuracy" in s["metrics"])
            n = score["scored_samples"]
            totals[(bench.upper(), model)][0] += score["metrics"]["accuracy"]["value"] * n
            totals[(bench.upper(), model)][1] += n
    return {key: correct / n for key, (correct, n) in totals.items()}


def wilson(c, n, z=1.96):
    if not n:
        return None, None
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def export_bench(bench, solve_by_bench, hf_scores, eci):
    dataset_id, label = DATASETS[bench]
    rollout_root, _ = BENCHES[bench]
    solve = solve_by_bench[bench]
    out = DATA_ROOT / dataset_id
    if out.exists():
        shutil.rmtree(out)  # supersedes any earlier export of this dataset

    logs = judging_logs(bench)
    # rollout transcripts + questions from the exact eval files the judging
    # referenced, solved samples only. Some of those logs never had their header
    # finalised (status "started"/"error" with all samples present), so a
    # status filter here would blank out epochs the judge did see.
    sources = set()
    for _path, header in logs.values():
        meta = header.eval.metadata
        for rel in (meta["major_path"], meta["minor_path"]):
            model, epoch, name = Path(rel).parts[-3:]
            sources.add((model, epoch, name))
    members = {model for model, _, _ in sources}

    questions, transcripts = {}, defaultdict(dict)
    for m, epoch, name in sorted(sources):
        path = rollout_root / m / epoch / name
        if not path.exists():
            # judging ran on a pod against a file the local copy doesn't have;
            # fall back to whatever this model/epoch has locally
            hits = sorted((rollout_root / m / epoch).glob("*.eval"))
            good = [p for p in hits if is_successful_eval(p)]
            if not (good or hits):
                continue
            path = (good or hits)[0]
        log = read_eval_log(str(path))
        for s in log.samples or []:
            questions.setdefault(f"{s.id}__ep{epoch}", str(s.input))
            if solved(s):
                transcripts[m][f"{s.id}__ep{epoch}"] = extract_transcript(
                    s, include_reasoning=False)

    pairs = defaultdict(dict)   # oriented pair -> item -> judge -> {"orders": [o1, o2]}
    cells = defaultdict(dict)   # (judge, oriented pair) -> item -> [order1, order2]
    for (judge, _pairname, epoch), (path, header) in sorted(logs.items()):
        meta = header.eval.metadata
        major = Path(meta["major_path"]).parts[-3]
        minor = Path(meta["minor_path"]).parts[-3]
        flip = solve[minor] > solve[major]
        strong, weak = (minor, major) if flip else (major, minor)
        oriented = f"{strong}-vs-{weak}"
        for s in read_eval_log(path).samples or []:
            base_item, _ = str(s.id).rsplit("_", 1)
            item = f"{base_item}__ep{epoch}"
            slot_strong = {"1": "2", "2": "1"}[s.target] if flip else s.target
            sc = next(iter((s.scores or {}).values()), None)
            pick = getattr(sc, "answer", None)
            entry = {
                "slot_of_major": slot_strong, "pick": pick,
                "picked_major": int(pick == slot_strong) if pick in ("1", "2") else None,
                "confidence": (sc.metadata or {}).get("confidence") if sc and sc.metadata else None,
                "explanation": getattr(sc, "explanation", None) if sc else None,
            }
            orders = pairs[oriented].setdefault(item, {}).setdefault(
                judge, {"orders": [None, None]})["orders"]
            orders[int(slot_strong) - 1] = entry
            cells[(judge, oriented)].setdefault(item, [None, None])[int(slot_strong) - 1] = entry

    runs = []
    for (judge, oriented), items in sorted(cells.items()):
        strong, weak = oriented.split("-vs-")
        flat = [o for orders in items.values() for o in orders if o is not None]
        valid = [o for o in flat if o["picked_major"] is not None]
        correct = sum(o["picked_major"] for o in valid)
        lo, hi = wilson(correct, len(valid))
        disagree = sum(o1 is not None and o2 is not None
                       and o1["picked_major"] != o2["picked_major"]
                       for o1, o2 in items.values())
        judge_solve = solve.get(judge)
        runs.append({
            "run_id": f"{judge}__{strong}__{weak}__full",
            "judge": judge, "major": strong, "minor": weak, "frac": "full",
            "d_acc": abs(solve[strong] - solve[weak]),
            "score": correct / len(valid) if valid else None,
            "score_lo": lo, "score_hi": hi, "n_items": len(items),
            # distinct benchmark questions; n_items counts (question x epoch) cells
            "n_questions": len({item.rsplit("__ep", 1)[0] for item in items}),
            "slot_a": (sum(o["pick"] == "1" for o in valid) / len(valid)) if valid else None,
            "disagree_rate": disagree / len(items) if items else None,
            "noanswer_rate": 1 - len(valid) / len(flat) if flat else None,
            "is_self": strong == weak,
            "weak_judge": bool(judge_solve is not None
                               and judge_solve < min(solve[strong], solve[weak])),
            "old_judge": False, "arm_a": True,
        })

    judges = sorted({r["judge"] for r in runs})
    models = {}
    for m in sorted(members | set(judges)):
        models[m] = {"label": m, "accuracy": solve.get(m),
                     "acc_swe_bench_mini": solve_by_bench["swe_bench_mini"].get(m),
                     "acc_intercode": solve_by_bench["intercode"].get(m),
                     "acc_gpqa": hf_scores.get(("GPQA", m)),
                     "acc_mbpp": hf_scores.get(("MBPP", m)),
                     "eci": eci.get(m), "release": "",
                     "arm": "judge" if m in judges and m not in members else "A"}

    for m, texts in transcripts.items():
        dump(texts, out / "transcripts" / f"{m}.json")
    for oriented, items in pairs.items():
        strong, weak = oriented.split("-vs-")
        dump(items, out / "pairs" / f"{strong}__{weak}__full.json")
    dump({"models": models, "judges": judges,
          "colors": {j: COLORS.get(j, "#888888") for j in judges},
          "questions": questions, "fits": {}, "runs": runs,
          "default_xmetric": bench,
          "available_xmetrics": ["eci", "swe_bench_mini", "intercode", "gpqa", "mbpp"],
          "metric_label": BENCHES[bench][1]}, out / "index.json")
    print(f"{dataset_id}: {len(runs)} runs, {len(pairs)} pairs, {len(models)} models, "
          f"{len(questions)} questions")
    return {"id": dataset_id, "label": label}


def main():
    solve_by_bench = {bench: load_solve_rates(root)
                      for bench, (root, _) in BENCHES.items()}
    hf_scores = load_hf_round_scores()
    eci = load_eci()
    entries = [export_bench(bench, solve_by_bench, hf_scores, eci) for bench in DATASETS]
    menu_path = DATA_ROOT / "datasets.json"
    menu = json.loads(menu_path.read_text()) if menu_path.exists() else []
    known = {e["id"] for e in entries}
    dump(entries + [d for d in menu if d["id"] not in known], menu_path)
    print("datasets.json updated")


if __name__ == "__main__":
    main()
