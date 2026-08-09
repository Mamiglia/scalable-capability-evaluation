"""Export the data-expansion judging into interactive_viewer/data/.

One dataset per bench (gpqa, aime2025, aime2026): index.json, transcripts/,
pairs/ in the shapes the viewer reads. The gpqa dataset also carries the
epoch-transcript big-gap probe (arm B). options/ answers/ fits are optional
in the viewer and skipped here.

Usage:
  uv run python scripts/judge/export_viewer.py
"""

import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import yaml
from inspect_ai.log import read_eval_log

from judge.util import extract_score
from judge.dataset import extract_transcript

DATA_ROOT = Path("interactive_viewer/data")
MODEL_SET = "configs/comparisons/model_set.json"
COMPARISONS = "configs/comparisons/comparisons.csv"
BIGGAP_CFG = "configs/judge_biggap_gpqa.yaml"

GRID_MEMBERS = ["qwen3-8b", "qwen3-vl-30b-a3b-instruct", "glm-4.6",
                "ling-2.6-flash", "deepseek-v4-flash-0731", "gemini-3.1-flash-lite"]

# bench -> (dataset label, collection hash dir, judge log roots, gemini subdir)
BENCHES = {
    "gpqa": ("Data expansion GPQA", "63793eb6",
             ["logs/data-expansion/gpqa", "logs/data-expansion-biggap/gpqa"],
             "gemini-3.1-flash-lite-rescored"),  # stock scorer misses gemini's $C$ answers
    "aime2025": ("Data expansion AIME 2025", "44136fa3",
                 ["logs/data-expansion/aime2025"], "gemini-3.1-flash-lite"),
    "aime2026": ("Data expansion AIME 2026", "44136fa3",
                 ["logs/data-expansion/aime2026"], "gemini-3.1-flash-lite"),
}

SHORT2DISPLAY = {
    "qwen3-8b": "Qwen3 8B", "qwen3-32b": "Qwen3 32B",
    "qwen3-vl-30b-a3b-instruct": "Qwen3 VL 30B A3B Instruct", "glm-4.6": "GLM-4.6",
    "ling-2.6-flash": "Ling-2.6-flash", "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
}
EPOCH_ECI = {"gpt-5.5-low": 159.21, "gemini-3.5-flash": 154.56, "qwen3.6-plus": 149.01,
             "claude-3-5-sonnet-0620": 130.0, "llama-3.1-70b": 125.2, "gemma-2-27b": 122.09}
JUDGE_COLORS = {"qwen3-8b": "#2a78d6", "qwen3-32b": "#eb6834",
                "qwen3-vl-30b-a3b-instruct": "#1baf7a", "ling-2.6-flash": "#eda100",
                "glm-4.6": "#e87ba4", "deepseek-v4-flash-0731": "#008300"}


def wilson(c, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = c / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def correct_epochs(log):
    """id -> correct samples, earliest epoch first (mirrors build_dataset)."""
    grouped = defaultdict(list)
    for s in log.samples:
        if extract_score(s):
            grouped[str(s.id)].append(s)
    for g in grouped.values():
        g.sort(key=lambda s: getattr(s, "epoch", 0))
    return grouped


def export_bench(bench, label, hashdir, judge_roots, gemini_dir, eci, flags):
    out = DATA_ROOT / f"data-expansion-{bench}"
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    (out / "pairs").mkdir(exist_ok=True)

    grid_logs = {}
    for m in GRID_MEMBERS:
        sub = gemini_dir if m == "gemini-3.1-flash-lite" else m
        grid_logs[m] = sorted(glob.glob(f"data/{bench}/{hashdir}/{sub}/0/*.eval"))[0]
    logs = dict(grid_logs)
    if bench == "gpqa":
        logs |= yaml.safe_load(open(BIGGAP_CFG))["transcripts"]

    models, questions = {}, {}
    for m, path in logs.items():
        log = read_eval_log(path)
        # from samples, not the header metric: the gemini rescore updates scores only
        acc = sum(bool(extract_score(s)) for s in log.samples) / len(log.samples)
        info = eci.get(SHORT2DISPLAY.get(m, ""), {})
        models[m] = {"label": SHORT2DISPLAY.get(m, m), "accuracy": acc,
                     "eci": info.get("eci") or EPOCH_ECI.get(m),
                     "release": (info.get("release") or "")[:7],
                     "arm": "A" if m in grid_logs else "B"}
        groups = correct_epochs(log)
        json.dump({i: extract_transcript(g[0], include_reasoning=False) for i, g in groups.items()},
                  open(out / f"transcripts/{m}.json", "w"))
        ep1 = {i: extract_transcript(g[1], include_reasoning=False) for i, g in groups.items() if len(g) > 1}
        if ep1:
            json.dump(ep1, open(out / f"transcripts/{m}__ep1.json", "w"))
        if m in grid_logs:
            for s in log.samples:
                questions.setdefault(str(s.id), str(s.input))

    runs = []
    pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"orders": [None, None]})))
    bypair = defaultdict(list)
    for root in judge_roots:
        for p in glob.glob(f"{root}/*/*/phase-*/*.eval"):
            if read_eval_log(p, header_only=True).status == "success":
                bypair[(p.split("/")[3], p.split("/")[4], root)].append(p)

    for (judge, pairname, root), paths in bypair.items():
        major, minor = pairname.split("-vs-")
        samples = [s for p in paths for s in read_eval_log(p).samples]
        n_c = n_ok = n_bad = slot_a = 0
        perorder = {}
        for s in samples:
            sc = list(s.scores.values())[0] if s.scores else None
            item, order = str(s.id).rsplit("_", 1)
            ans = sc.answer if sc else None
            if ans in ("1", "2"):
                n_ok += 1
                slot_a += ans == "1"
                if sc.value == "C":
                    n_c += 1
            else:
                n_bad += 1
            picked_major = None if ans not in ("1", "2") else int(ans == s.target)
            entry = {"slot_of_major": s.target, "pick": ans, "picked_major": picked_major,
                     "confidence": (sc.metadata or {}).get("confidence") if sc else None,
                     "explanation": sc.explanation if sc else None}
            pairs[pairname][item][judge]["orders"][int(order) - 1] = entry
            perorder.setdefault(item, {})[order] = picked_major
        lo, hi = wilson(n_c, n_ok)
        disagree = [1 for v in perorder.values()
                    if v.get("1") is not None and v.get("2") is not None and v["1"] != v["2"]]
        f = flags.get((SHORT2DISPLAY.get(judge, judge),
                       frozenset([SHORT2DISPLAY.get(major, major), SHORT2DISPLAY.get(minor, minor)])),
                      ("biggap" in root, False))
        runs.append({
            "run_id": f"{judge}__{major}__{minor}__full", "judge": judge,
            "major": major, "minor": minor, "frac": "full",
            "d_acc": abs(models[major]["accuracy"] - models[minor]["accuracy"]),
            "score": (n_c / n_ok) if n_ok else None, "score_lo": lo, "score_hi": hi,
            "n_items": len(perorder), "slot_a": (slot_a / n_ok) if n_ok else None,
            "disagree_rate": len(disagree) / max(len(perorder), 1),
            "noanswer_rate": n_bad / max(len(samples), 1),
            "is_self": major == minor, "weak_judge": f[0], "old_judge": f[1],
            "arm_a": major in grid_logs and minor in grid_logs,
        })

    for pairname, items in pairs.items():
        major, minor = pairname.split("-vs-")
        json.dump(items, open(out / f"pairs/{major}__{minor}__full.json", "w"))

    json.dump({"models": models, "judges": sorted({r["judge"] for r in runs}),
               "colors": JUDGE_COLORS, "questions": questions, "fits": {}, "runs": runs},
              open(out / "index.json", "w"))
    print(f"{bench}: {len(runs)} runs, {len(pairs)} pairs, {len(models)} models")
    return {"id": f"data-expansion-{bench}", "label": label}


def main():
    eci = {m["name"]: m for m in json.load(open(MODEL_SET))["models"]}
    flags = {}
    for r in csv.DictReader(open(COMPARISONS)):
        key = (r["judge"], frozenset([r["model_a"], r["model_b"]]))
        flags[key] = (r["weak_judge_case"] == "true", r["one_model_newer_case"] == "true")

    entries = [export_bench(b, *spec, eci, flags) for b, spec in BENCHES.items()]
    per_bench = {b: json.load(open(DATA_ROOT / f"data-expansion-{b}/index.json")) for b in BENCHES}
    for b, idx in per_bench.items():
        for m, meta in idx["models"].items():
            for ob, oidx in per_bench.items():
                if m in oidx["models"]:
                    meta[f"acc_{ob}"] = oidx["models"][m]["accuracy"]
        json.dump(idx, open(DATA_ROOT / f"data-expansion-{b}/index.json", "w"))
    listing = json.load(open(DATA_ROOT / "datasets.json"))
    listing = entries + [d for d in listing if not d["id"].startswith("data-expansion-")]
    json.dump(listing, open(DATA_ROOT / "datasets.json", "w"))


if __name__ == "__main__":
    main()
