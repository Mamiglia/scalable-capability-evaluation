"""Export the data-expansion judging into interactive_dashboards/transcript_viewer/data/.

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

DATA_ROOT = Path("interactive_dashboards/transcript_viewer/data")
MODEL_SET = "configs/comparisons/model_set.json"
COMPARISONS = "configs/comparisons/comparisons.csv"
BIGGAP_CFG = "configs/judge_biggap_gpqa.yaml"
GPQA_ROOT_CONFIGS = {
    "logs/data-expansion-bridge/gpqa": "configs/judge_bridge_gpqa.yaml",
    "logs/data-expansion-models/gpqa": "configs/judge_expand_gpqa_models.yaml",
    "logs/data-expansion-more-pairs/gpqa": "configs/judge_more_gpqa_pairs.yaml",
}

# collected members are whatever sits in the bench's collection dir

# bench -> (dataset label, collection hash dir, judge log roots, gemini subdir,
#            include_reasoning in exported transcripts)
BENCHES = {
    "gpqa": ("Data expansion GPQA", "63793eb6",
             ["logs/data-expansion/gpqa", "logs/data-expansion-biggap/gpqa",
              "logs/data-expansion-bridge/gpqa", "logs/data-expansion-models/gpqa",
              "logs/data-expansion-more-pairs/gpqa"],
             "gemini-3.1-flash-lite-rescored", False),  # stock scorer misses gemini's $C$ answers
    "aime2025": ("Data expansion AIME 2025", "44136fa3",
                 ["logs/data-expansion/aime2025"], "gemini-3.1-flash-lite", False),
    "aime2026": ("Data expansion AIME 2026", "44136fa3",
                 ["logs/data-expansion/aime2026"], "gemini-3.1-flash-lite", False),
    "aime2025-ablation": ("AIME 2025 reasoning ablation", "44136fa3",
                          ["logs/data-expansion-reasoning-ablation/aime2025"],
                          "gemini-3.1-flash-lite", True),
    "aime2026-ablation": ("AIME 2026 reasoning ablation", "44136fa3",
                          ["logs/data-expansion-reasoning-ablation/aime2026"],
                          "gemini-3.1-flash-lite", True),
}

SHORT2DISPLAY = {
    "qwen3-8b": "Qwen3 8B", "qwen3-32b": "Qwen3 32B",
    "qwen3-vl-30b-a3b-instruct": "Qwen3 VL 30B A3B Instruct", "glm-4.6": "GLM-4.6",
    "ling-2.6-flash": "Ling-2.6-flash", "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gpt-4.1": "GPT-4.1", "o4-mini": "o4-mini", "claude-opus-4.7": "Claude Opus 4.7",
    "deepseek-v4-flash-nothink": "DeepSeek V4 Flash (no think)",
}
# ECIs not in model_set.json: the epoch-import members, plus qwen3-32b
EPOCH_ECI = {"gpt-5.5-low": 159.21, "gemini-3.5-flash": 154.56, "qwen3.6-plus": 149.01,
             "claude-3-5-sonnet-0620": 130.0, "llama-3.1-70b": 125.2, "gemma-2-27b": 122.09,
             "qwen3-32b": 130.0, "deepseek-v4-flash-nothink": 150.22}
JUDGE_COLORS = {"qwen3-8b": "#2a78d6", "qwen3-32b": "#eb6834",
                "qwen3-vl-30b-a3b-instruct": "#1baf7a", "ling-2.6-flash": "#eda100",
                "glm-4.6": "#e87ba4", "deepseek-v4-flash-0731": "#008300"}


def configured_pair_names(config_path):
    """Return the pair directories authorized by the current run config."""
    config = yaml.safe_load(open(config_path))
    return {"-vs-".join(pair) for pair in config.get("pairs", [])}


def add_missing_transcripts(logs, config_paths):
    """Add expanded candidates without replacing canonical response logs."""
    logs = dict(logs)
    for config_path in config_paths:
        config = yaml.safe_load(open(config_path))
        for model, path in config.get("transcripts", {}).items():
            logs.setdefault(model, path)
    return logs


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


def export_bench(bench, label, hashdir, judge_roots, gemini_dir, incl_reasoning, eci, flags):
    out = DATA_ROOT / f"data-expansion-{bench}"
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    (out / "pairs").mkdir(exist_ok=True)

    members = sorted(d.name for d in Path(f"data/{bench}/{hashdir}").iterdir()
                     if d.is_dir() and not d.name.endswith("-rescored"))
    grid_logs = {}
    for m in members:
        sub = gemini_dir if m == "gemini-3.1-flash-lite" else m
        found = sorted(glob.glob(f"data/{bench}/{hashdir}/{sub}/0/*.eval"))
        if found:
            grid_logs[m] = found[0]
    logs = dict(grid_logs)
    if bench == "gpqa":
        logs = add_missing_transcripts(
            logs,
            [BIGGAP_CFG, *GPQA_ROOT_CONFIGS.values()],
        )

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
        json.dump({i: extract_transcript(g[0], include_reasoning=incl_reasoning) for i, g in groups.items()},
                  open(out / f"transcripts/{m}.json", "w"))
        ep1 = {i: extract_transcript(g[1], include_reasoning=incl_reasoning) for i, g in groups.items() if len(g) > 1}
        if ep1:
            json.dump(ep1, open(out / f"transcripts/{m}__ep1.json", "w"))
        if m in grid_logs:
            for s in log.samples:
                questions.setdefault(str(s.id), str(s.input))

    runs = []
    pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"orders": [None, None]})))
    bypair = defaultdict(list)
    for root in judge_roots:
        allowed_pairs = (
            configured_pair_names(GPQA_ROOT_CONFIGS[root])
            if root in GPQA_ROOT_CONFIGS
            else None
        )
        for p in glob.glob(f"{root}/*/*/phase-*/*.eval"):
            judge, pairname = Path(p).relative_to(root).parts[:2]
            if allowed_pairs is not None and pairname not in allowed_pairs:
                continue
            if read_eval_log(p, header_only=True).status == "success":
                bypair[(judge, pairname, root)].append(p)

    for (judge, pairname, root), paths in bypair.items():
        major, minor = pairname.split("-vs-")
        # log targets track the config's first-named model, and comparisons.csv
        # naming is mixed; orient every pair so major = higher-ECI member
        ea, eb = models[major]["eci"], models[minor]["eci"]
        flip = major != minor and ea and eb and ea < eb
        if flip:
            major, minor = minor, major
        oriented = f"{major}-vs-{minor}"
        samples = [s for p in paths for s in read_eval_log(p).samples]
        n_c = n_ok = n_bad = slot_a = 0
        perorder = {}
        for s in samples:
            sc = list(s.scores.values())[0] if s.scores else None
            item, order = str(s.id).rsplit("_", 1)
            tgt = {"1": "2", "2": "1"}[s.target] if flip else s.target
            ans = sc.answer if sc else None
            if ans in ("1", "2"):
                n_ok += 1
                slot_a += ans == "1"
                if ans == tgt:
                    n_c += 1
            else:
                n_bad += 1
            picked_major = None if ans not in ("1", "2") else int(ans == tgt)
            entry = {"slot_of_major": tgt, "pick": ans, "picked_major": picked_major,
                     "confidence": (sc.metadata or {}).get("confidence") if sc else None,
                     "explanation": sc.explanation if sc else None}
            pairs[oriented][item][judge]["orders"][int(order) - 1] = entry
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

    # Orient stronger-first (by ECI): the raw logs/dirs may name the weaker member
    # first; the viewer must read score as P(picked the stronger member).
    def eci_of(m):
        return models[m].get("eci") or 0
    for r in runs:
        if eci_of(r["minor"]) > eci_of(r["major"]):
            r["major"], r["minor"] = r["minor"], r["major"]
            if r["score"] is not None:
                r["score"], r["score_lo"], r["score_hi"] = 1 - r["score"], 1 - r["score_hi"], 1 - r["score_lo"]
            pairname = f"{r['minor']}-vs-{r['major']}"
            if pairname in pairs and not pairs[pairname].get("_flipped"):
                for item, judges in pairs[pairname].items():
                    for j, cell in judges.items():
                        for o in cell["orders"]:
                            if o:
                                o["slot_of_major"] = "2" if o["slot_of_major"] == "1" else "1"
                                if o["picked_major"] is not None:
                                    o["picked_major"] = 1 - o["picked_major"]
                pairs[f"{r['major']}-vs-{r['minor']}"] = pairs.pop(pairname)
                pairs[f"{r['major']}-vs-{r['minor']}"]["_flipped"] = True

    for pairname, items in pairs.items():
        items.pop("_flipped", None)
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
