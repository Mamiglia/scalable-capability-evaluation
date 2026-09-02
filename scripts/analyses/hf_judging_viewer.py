"""Export the isolated 2026-08-17 Hugging Face judging round to the transcript viewer."""

import json
import gc
import math
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from hf_judging_round import load_benchmark_scores, keep_judge


ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / "data" / "hf-judging-2026-08-17"
RAW = ROUND / "raw"
BENCHMARK_SCORES = ROUND / "benchmark-scores"
VIEWER_DATA = ROOT / "interactive_dashboards" / "transcript_viewer" / "data"
DATASET_PREFIX = "hf-judging-2026-08-17"
COLORS = {
    "deepseek-v4-flash": "#2a78d6",
    "gemma4-31b": "#eb6834",
    "gpt-oss-120b": "#1baf7a",
    "ling": "#eda100",
    "llama3.1-8b": "#4a3aa7",
}
LABELS = {
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "gemma4-31b": "Gemma 4 31B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "ling": "Ling 2.6 Flash",
    "llama3.1-8b": "Llama 3.1 8B",
}
JUDGE_SCORE_MODEL = {"ling": "ling-2.6-flash"}

PROMPT = re.compile(
    r"\AInput question:\s*\n---\n(?P<question>.*?)\n---\n\n"
    r"(?:Multiple choice options as shown to Model 1:\n(?P<options_1>.*?)\n\n)?"
    r"Transcript 1:\n---\n(?P<transcript_1>.*?)\n---\n\n"
    r"(?:Multiple choice options as shown to Model 2:\n(?P<options_2>.*?)\n\n)?"
    r"Transcript 2:\n---\n(?P<transcript_2>.*?)\n---\n\n"
    r"Which of the two transcripts is from the more capable model\?",
    re.DOTALL,
)


def parse_prompt(text):
    match = PROMPT.search(text)
    if not match:
        raise ValueError("unrecognised judge prompt")

    def options(block):
        return re.findall(r"^[A-Z]\)\s*(.*)$", block or "", re.MULTILINE)

    return {
        "question": match["question"].strip(),
        "transcript_1": match["transcript_1"].strip(),
        "transcript_2": match["transcript_2"].strip(),
        "options_1": options(match["options_1"]),
        "options_2": options(match["options_2"]),
    }


def answer_letter(transcript):
    matches = re.findall(r"(?:FINAL\s+)?ANSWER:\s*[*_$]*([A-Z])", transcript, re.IGNORECASE)
    return matches[-1].upper() if matches else None


def model_meta(model, bench, scores):
    score_model = JUDGE_SCORE_MODEL.get(model, model)
    return {
        "label": LABELS.get(model, model),
        "accuracy": scores.get((bench.upper(), score_model)),
        "acc_gpqa": scores.get(("GPQA", score_model)),
        "acc_mbpp": scores.get(("MBPP", score_model)),
        "release": "",
        "arm": "judge" if model in COLORS else "A",
    }


def wilson(correct, n, z=1.96):
    if not n:
        return None, None
    p, d = correct / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def export_bench(bench, scores):
    eval_name = bench.upper()
    out = VIEWER_DATA / f"{DATASET_PREFIX}-{bench}"
    grouped = defaultdict(list)
    for path in RAW.glob(f"pilot-*-{bench}-*/*.eval"):
        dirname = path.parent.name
        judge = dirname.removeprefix("pilot-").split(f"-{bench}-", 1)[0]
        if keep_judge(judge):
            grouped[dirname.split(f"-{bench}-", 1)[1]].append(path)

    used_models, runs, parse_failures = set(), [], 0
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"hf-viewer-{bench}-", dir=ROOT / "tmp") as tmp:
        connection = sqlite3.connect(Path(tmp) / "viewer.sqlite")
        connection.executescript("""
            CREATE TABLE transcripts (model TEXT, item TEXT, text TEXT, options TEXT, answer TEXT,
                                      PRIMARY KEY (model, item));
            CREATE TABLE questions (item TEXT PRIMARY KEY, question TEXT);
        """)

        for pair_name, paths in sorted(grouped.items()):
            items, strong = {}, None
            for path in sorted(paths):
                dirname = path.parent.name
                judge = dirname.removeprefix("pilot-").split(f"-{bench}-", 1)[0]
                log = read_eval_log(str(path))
                if log.status != "success":
                    continue
                meta = log.eval.metadata or {}
                major_path, minor_path = Path(meta["major_path"]), Path(meta["minor_path"])
                major, minor = major_path.parts[-3], minor_path.parts[-3]
                if (eval_name, major) not in scores or (eval_name, minor) not in scores:
                    continue
                major_is_strong = scores[(eval_name, major)] > scores[(eval_name, minor)]
                pair = (major, minor) if major_is_strong else (minor, major)
                if strong is None:
                    strong = pair
                assert strong == pair, f"orientation changed within {pair_name}"
                epoch = f"{major_path.parts[-2]}-{minor_path.parts[-2]}"
                used_models.update((*pair, judge))

                for sample in log.samples or []:
                    try:
                        parsed = parse_prompt(str(sample.input))
                    except ValueError:
                        # ponytail: Inspect stored 11 prompts truncated at 5,143 chars;
                        # fetch their source evals if transcript-complete coverage matters.
                        parse_failures += 1
                        continue
                    base_item, _ = str(sample.id).rsplit("_", 1)
                    item = f"{base_item}__ep{epoch}"
                    slot_major = str(sample.target)
                    slot_strong = slot_major if major_is_strong else ("2" if slot_major == "1" else "1")
                    slot_weak = "2" if slot_strong == "1" else "1"
                    strong_text = parsed[f"transcript_{slot_strong}"]
                    weak_text = parsed[f"transcript_{slot_weak}"]
                    for model, slot, text in ((pair[0], slot_strong, strong_text),
                                              (pair[1], slot_weak, weak_text)):
                        connection.execute(
                            "INSERT OR REPLACE INTO transcripts VALUES (?, ?, ?, ?, ?)",
                            (model, item, text, json.dumps(parsed[f"options_{slot}"]),
                             answer_letter(text)),
                        )
                    connection.execute("INSERT OR REPLACE INTO questions VALUES (?, ?)",
                                       (item, parsed["question"]))

                    score = next(iter((sample.scores or {}).values()), None)
                    pick = getattr(score, "answer", None)
                    picked_strong = int(pick == slot_strong) if pick in ("1", "2") else None
                    orders = items.setdefault(item, {}).setdefault(
                        judge, {"orders": [None, None]})["orders"]
                    orders[int(slot_strong) - 1] = {
                        "slot_of_major": slot_strong,
                        "pick": pick,
                        "picked_major": picked_strong,
                        "confidence": ((score.metadata or {}).get("confidence")
                                       if score and score.metadata else None),
                        "explanation": getattr(score, "explanation", None) if score else None,
                    }
                del log

            if strong is None:
                continue
            connection.commit()
            dump(items, out / "pairs" / f"{strong[0]}__{strong[1]}__full.json")
            for judge in sorted({j for judges in items.values() for j in judges}):
                cells = [judges[judge]["orders"] for judges in items.values() if judge in judges]
                flat = [order for orders_ in cells for order in orders_ if order is not None]
                valid = [order for order in flat if order["picked_major"] is not None]
                correct = sum(order["picked_major"] for order in valid)
                lo, hi = wilson(correct, len(valid))
                disagree = sum(orders_[0] is not None and orders_[1] is not None
                               and orders_[0]["picked_major"] != orders_[1]["picked_major"]
                               for orders_ in cells)
                runs.append({
                    "run_id": f"{judge}__{strong[0]}__{strong[1]}__full",
                    "judge": judge, "major": strong[0], "minor": strong[1], "frac": "full",
                    "d_acc": abs(scores[(eval_name, strong[0])] - scores[(eval_name, strong[1])]),
                    "score": correct / len(valid) if valid else None,
                    "score_lo": lo, "score_hi": hi, "n_items": len(cells),
                    "slot_a": (sum(order["pick"] == "1" for order in valid) / len(valid)
                               if valid else None),
                    "disagree_rate": disagree / len(cells) if cells else None,
                    "noanswer_rate": 1 - len(valid) / len(flat) if flat else None,
                    "is_self": False, "weak_judge": False, "old_judge": False, "arm_a": True,
                })
            del items
            gc.collect()

        if parse_failures:
            print(f"{bench}: skipped {parse_failures} stored prompts truncated before closing delimiters")

        candidate_models = sorted(model for model in used_models if model not in COLORS)
        for model in candidate_models:
            rows = connection.execute(
                "SELECT item, text, options, answer FROM transcripts WHERE model=?", (model,))
            text, opts, ans = {}, {}, {}
            for item, transcript, option_json, answer in rows:
                text[item] = transcript
                choices = json.loads(option_json)
                if choices:
                    opts[item] = choices
                ans[item] = answer
            dump(text, out / "transcripts" / f"{model}.json")
            dump(opts, out / "options" / f"{model}.json")
            dump(ans, out / "answers" / f"{model}.json")
        questions = dict(connection.execute("SELECT item, question FROM questions"))
        connection.close()

    models = {model: model_meta(model, bench, scores) for model in sorted(used_models)}
    for run in runs:
        judge_acc = models[run["judge"]]["accuracy"]
        run["weak_judge"] = bool(judge_acc is not None and judge_acc < min(
            scores[(eval_name, run["major"])], scores[(eval_name, run["minor"])]))

    judges = sorted(model for model in used_models if model in COLORS)
    assert judges and all(keep_judge(judge) for judge in judges)
    assert runs and candidate_models and questions
    available_xmetrics = [metric for metric in ("gpqa", "mbpp")
                          if all(scores.get((metric.upper(), model)) is not None
                                 for model in candidate_models)]
    dump({"models": models, "judges": judges,
          "colors": {judge: COLORS[judge] for judge in judges},
          "questions": questions, "fits": {}, "runs": runs,
          "default_xmetric": bench, "available_xmetrics": available_xmetrics,
          "metric_label": eval_name}, out / "index.json")
    print(f"{bench}: {len(runs)} runs, {len(grouped)} source pairs, {len(questions)} transcript items")
    return {"id": f"{DATASET_PREFIX}-{bench}", "label": f"HF judging 2026-08-17 — {eval_name}"}


def main():
    scores = load_benchmark_scores(BENCHMARK_SCORES)
    entries = [export_bench(bench, scores) for bench in ("gpqa", "mbpp")]
    menu_path = VIEWER_DATA / "datasets.json"
    menu = json.loads(menu_path.read_text()) if menu_path.exists() else []
    dump(entries + [item for item in menu if not item["id"].startswith(DATASET_PREFIX)], menu_path)
    print(f"datasets.json: prepended {len(entries)} HF datasets")


if __name__ == "__main__":
    main()
