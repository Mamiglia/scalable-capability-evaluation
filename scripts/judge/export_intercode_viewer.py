"""Export the intercode-CTF pilot judging to the transcript viewer.

Reads rollouts from data/intercode/463eb861 and judge logs from
logs/pilot-intercode/epoch-<n>/<judge>/<pair>/phase-*/, writes one viewer
dataset (index.json, transcripts/, pairs/) and registers it in datasets.json.
Items carry an __ep<n> suffix so both rollout epochs share one dataset.

Usage:
  uv run python scripts/judge/export_intercode_viewer.py
"""

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))

from inspect_ai.log import read_eval_log

from hf_judging_round import load_benchmark_scores

from configs.paths import is_successful_eval
from judge.util import extract_score, extract_transcript

DATA_ROOT = Path("interactive_dashboards/transcript_viewer/data")
ROLLOUTS = Path("data/intercode/463eb861")
EPOCH_ROOTS = {0: Path("logs/pilot-intercode/epoch-0"), 1: Path("logs/pilot-intercode/epoch-1")}
DATASET_ID = "intercode-pilot"
LABEL = "Intercode CTF pilot"

# Pair dirs use the pilot's model spellings; rollout dirs differ for the gemmas.
ROLLOUT_NAME = {"gemma3-12b-it": "gemma-3-12b-it", "gemma4-e2b-it": "gemma-4-e2b-it"}

# Epoch Capabilities Index, from the published table (external_data/epoch-eci/).
# Keyed by that table's display names; members absent from it get eci: null and
# the viewer falls back to the accuracy gap for their pairs on the ECI axis.
ECI_TABLE = Path("external_data/epoch-eci/eci_scores.csv")
ECI_NAME = {
    "claude-opus-4.7": "Claude Opus 4.7",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek-r1-0528": "DeepSeek-R1 (May 2025)",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "glm-4.6": "GLM-4.6",
    "glm-5.2": "GLM-5.2",
    "gpt-4.1": "GPT-4.1",
    "gpt-5": "GPT-5",
    "kimi-k2.6": "Kimi K2.6",
    "kimi-k3": "Kimi K3",
    "mistral-medium-3": "Mistral Medium 3",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "ling-2.6-flash": "Ling-2.6-flash",
    "gemma-4-31b-it": "Gemma 4 31B IT",
}
# Not in Epoch's table; values from configs/comparisons/model_set.json and the
# data-expansion export (qwen3-32b).
ECI_EXTRA = {"ling-2.6-flash": 131.8, "qwen3-32b": 130.0}


def load_eci() -> dict:
    import csv
    table = {r["Display name"]: float(r["eci"])
             for r in csv.DictReader(ECI_TABLE.open()) if r.get("eci")}
    eci = {short: table[name] for short, name in ECI_NAME.items() if name in table}
    return ECI_EXTRA | eci

COLORS = {
    "deepseek-r1-0528": "#2a78d6", "deepseek-v4-flash-0731": "#eb6834",
    "gemma-4-31b-it": "#1baf7a", "gpt-oss-120b": "#eda100",
    "ling-2.6-flash": "#e87ba4", "o4-mini": "#4a3aa7",
    "Llama-3.1-8B-Instruct": "#008300",
}


def rollout_log(model: str, epoch: int) -> str | None:
    hits = [p for p in sorted(glob.glob(f"{ROLLOUTS}/{model}/{epoch}/*.eval"))
            if is_successful_eval(p)]
    return hits[0] if hits else None


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


def main():
    out = DATA_ROOT / DATASET_ID
    members = sorted(d.name for d in ROLLOUTS.iterdir() if d.is_dir())

    accuracy, questions = {}, {}
    transcripts = defaultdict(dict)  # model -> {item__ep: text}
    for m in members:
        for epoch in EPOCH_ROOTS:
            path = rollout_log(m, epoch)
            if path is None:
                continue
            log = read_eval_log(path)
            if epoch == 0:
                accuracy[m] = sum(bool(extract_score(s)) for s in log.samples) / len(log.samples)
            for s in log.samples:
                questions.setdefault(str(s.id), str(s.input))
                if extract_score(s):
                    transcripts[m][f"{s.id}__ep{epoch}"] = extract_transcript(
                        s, include_reasoning=False)

    # (judge, oriented pair) -> aggregation across epochs
    pairs = defaultdict(dict)   # pairname -> item -> judge -> {"orders": [o1, o2]}
    cells = defaultdict(dict)   # (judge, pairname) -> item -> [order1, order2]
    for epoch, root in EPOCH_ROOTS.items():
        for p in glob.glob(f"{root}/*/*/phase-*/*.eval"):
            judge, pairname = Path(p).relative_to(root).parts[:2]
            log = read_eval_log(p)
            if log.status != "success":
                continue
            major, minor = (ROLLOUT_NAME.get(m, m) for m in pairname.split("-vs-"))
            flip = accuracy.get(minor, 0) > accuracy.get(major, 0)
            strong, weak = (minor, major) if flip else (major, minor)
            oriented = f"{strong}-vs-{weak}"
            for s in log.samples or []:
                base_item, _ = str(s.id).rsplit("_", 1)
                item = f"{base_item}__ep{epoch}"
                slot_strong = ({"1": "2", "2": "1"}[s.target] if flip else s.target)
                sc = next(iter((s.scores or {}).values()), None)
                pick = getattr(sc, "answer", None)
                picked_strong = int(pick == slot_strong) if pick in ("1", "2") else None
                entry = {
                    "slot_of_major": slot_strong, "pick": pick,
                    "picked_major": picked_strong,
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
        judge_acc = accuracy.get(judge)
        runs.append({
            "run_id": f"{judge}__{strong}__{weak}__full",
            "judge": judge, "major": strong, "minor": weak, "frac": "full",
            "d_acc": abs(accuracy[strong] - accuracy[weak]),
            "score": correct / len(valid) if valid else None,
            "score_lo": lo, "score_hi": hi, "n_items": len(items),
            "slot_a": (sum(o["pick"] == "1" for o in valid) / len(valid)) if valid else None,
            "disagree_rate": disagree / len(items) if items else None,
            "noanswer_rate": 1 - len(valid) / len(flat) if flat else None,
            "is_self": strong == weak,
            "weak_judge": bool(judge_acc is not None
                               and judge_acc < min(accuracy[strong], accuracy[weak])),
            "old_judge": False, "arm_a": True,
        })

    judges = sorted({r["judge"] for r in runs})
    bench_scores = load_benchmark_scores(Path("data/hf-judging-2026-08-17/benchmark-scores"))
    eci = load_eci()
    models = {}
    for m in sorted(set(members) | set(judges)):
        models[m] = {"label": m, "accuracy": accuracy.get(m),
                     "acc_intercode": accuracy.get(m),
                     "acc_gpqa": bench_scores.get(("GPQA", m)),
                     "acc_mbpp": bench_scores.get(("MBPP", m)),
                     "eci": eci.get(m), "release": "",
                     "arm": "judge" if m in judges and m not in members else "A"}

    for m, texts in transcripts.items():
        dump(texts, out / "transcripts" / f"{m}.json")
    for pairname, items in pairs.items():
        strong, weak = pairname.split("-vs-")
        dump(items, out / "pairs" / f"{strong}__{weak}__full.json")
    dump({"models": models, "judges": judges,
          "colors": {j: COLORS.get(j, "#888888") for j in judges},
          "questions": questions, "fits": {}, "runs": runs,
          "default_xmetric": "intercode",
          "available_xmetrics": ["eci", "intercode", "gpqa", "mbpp"],
          "metric_label": "INTERCODE"}, out / "index.json")

    menu_path = DATA_ROOT / "datasets.json"
    menu = json.loads(menu_path.read_text()) if menu_path.exists() else []
    menu = [{"id": DATASET_ID, "label": LABEL}] + [d for d in menu if d["id"] != DATASET_ID]
    dump(menu, menu_path)
    print(f"{len(runs)} runs, {len(pairs)} pairs, {len(models)} models, "
          f"{len(questions)} questions")


if __name__ == "__main__":
    main()
