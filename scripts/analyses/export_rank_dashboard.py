"""Export a rank/ judge log dir (the domain pipeline) to interactive_viewer/data/.

One viewer dataset per <judge, condition> log dir: the capability-gap curve (judge accuracy
vs the ground-truth q_dst accuracy gap), click-through to the two transcripts the judge saw
in both counterbalanced orders, with its pick, confidence and explanation.

Everything comes from the judge logs alone -- the pair metadata carries the model shortnames
and the q_dst accuracies, and the stored prompt carries the transcripts, so the source
data/ logs are never re-read.

Usage:
  uv run python scripts/analyses/export_rank_dashboard.py logs/rank/math/gemma-4-31b-it/*/
  uv run python scripts/analyses/export_rank_dashboard.py <dir> --id math-q1 --label "MATH Q1"
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning

from scipy.stats import kendalltau

from configs.domain import DomainConfig
from stats import sigmoid_fit

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DATA = ROOT / "interactive_viewer" / "data"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948", "#8c6d31", "#0b7285"]

# judge.dataset.USER_PROMPT, read back. The delimiter must be followed by the next section's
# header, so a transcript containing "---" cannot be mistaken for the end of a section.
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
    options = lambda block: re.findall(r"^[A-Z]\)\s*(.*)$", block or "", re.MULTILINE)
    return {"question": match["question"].strip(),
            "transcript_1": match["transcript_1"].strip(),
            "transcript_2": match["transcript_2"].strip(),
            "options_1": options(match["options_1"]),
            "options_2": options(match["options_2"])}


def answer_letter(transcript):
    """MCQ benches only: the letter the model answered, in its own option order."""
    matches = re.findall(r"(?:FINAL\s+)?ANSWER:\s*[*_$]*([A-Z])\b", transcript, re.IGNORECASE)
    return matches[-1].upper() if matches else None


def judge_reasoning(sample):
    """The judge's own CoT, if the provider returned one (the transcripts' CoT is a
    separate thing, controlled by include_reasoning)."""
    choices = getattr(sample.output, "choices", None) or []
    content = choices[0].message.content if choices else ""
    if not isinstance(content, list):
        return ""
    return "\n\n".join(c.reasoning for c in content if isinstance(c, ContentReasoning) and c.reasoning)


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


def frac_str(truncate_frac):
    return "full" if truncate_frac in (None, 0) else str(int(round(truncate_frac * 100)))


def load_csv(path, floats=(), ints=()):
    """A rank(1) output CSV as a list of dicts, absent (not yet ranked) -> []."""
    if not path.exists():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in floats:
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
        for key in ints:
            row[key] = int(float(row[key])) if row.get(key) not in (None, "") else None
    return rows


def collect(log_dir):
    """Read one condition dir into (transcripts, options, answers, questions, pairs, models, meta)."""
    transcripts, options, answers, questions = {}, {}, {}, {}
    pairs = defaultdict(lambda: defaultdict(dict))  # (major, minor, frac) -> item -> judge -> cell
    accuracy, condition, skipped = {}, {}, 0

    for path in sorted(Path(log_dir).glob("*.eval")):
        log = read_eval_log(str(path))
        if log.status != "success":
            skipped += 1
            continue
        md = log.eval.metadata or {}
        judge = log.eval.model.split("/")[-1].lower()
        major, minor, bench = md["major"], md["minor"], md["bench"]
        frac = frac_str(md.get("truncate_frac"))
        accuracy[major], accuracy[minor] = md["acc_major"], md["acc_minor"]
        condition = {k: md[k] for k in ("domain", "q_src", "q_dst", "include_reasoning")}
        self_pair = major == minor
        minor_stem = f"{minor}__ep1" if self_pair else minor

        for sample in log.samples or []:
            parsed = parse_prompt(str(sample.input))
            item = f"{bench}:{str(sample.id).rsplit('_', 1)[0]}"
            questions[item] = parsed["question"]
            slot_major = str(sample.target)
            slot_minor = "2" if slot_major == "1" else "1"
            for model, slot in ((major, slot_major), (minor_stem, slot_minor)):
                text = parsed[f"transcript_{slot}"]
                transcripts.setdefault(model, {})[item] = text
                if parsed[f"options_{slot}"]:
                    options.setdefault(model, {})[item] = parsed[f"options_{slot}"]
                    answers.setdefault(model, {})[item] = answer_letter(text)

            score = next(iter((sample.scores or {}).values()), None)
            pick = getattr(score, "answer", None)
            cell = pairs[(major, minor, frac)][item].setdefault(judge, {"orders": [None, None]})
            md_sample = sample.metadata or {}
            cell["orders"][int(slot_major) - 1] = {
                "slot_of_major": slot_major,
                "pick": pick if pick in ("1", "2") else None,
                "picked_major": int(pick == slot_major) if pick in ("1", "2") else None,
                "confidence": (score.metadata or {}).get("confidence") if score else None,
                "explanation": getattr(score, "explanation", None) if score else None,
                "reasoning": judge_reasoning(sample),
                # shown lengths, slot order: what length_corr and the length column read
                "chars": [md_sample.get("chars_1"), md_sample.get("chars_2")],
            }
        del log

    return transcripts, options, answers, questions, pairs, accuracy, condition, skipped


def pearson(xs, ys):
    """Pearson r, or None when a side is constant (statistics.correlation's error case)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / den if den else None


def metrics(cells):
    """judge.scorer's metrics over a list of per-item [orderA, orderB] pairs.

    Same definitions as the scorer (epochs = 1 here, so its epoch-reduction caveats do not
    apply): accuracy counts a missing verdict as wrong, position_bias is -1 for pure primacy
    and +1 for pure recency, consistent_accuracy needs both orders right, length_corr is phi
    between "the longer transcript is Transcript 1" and "the judge picked Transcript 1".
    """
    flat = [order for orders in cells for order in orders if order is not None]
    answered = [order for order in flat if order["picked_major"] is not None]
    picked_1 = [1.0 if order["pick"] == "1" else 0.0 for order in answered]
    both = [orders for orders in cells if all(o is not None for o in orders)]
    lengths = [(order, order["chars"]) for order in answered
               if order["chars"] and None not in order["chars"]]
    confidences = [order["confidence"] for order in flat if order["confidence"] is not None]
    return {
        "n": len(flat),
        "accuracy": sum(order["picked_major"] for order in answered) / len(flat) if flat else None,
        "score": (sum(order["picked_major"] for order in answered) / len(answered)
                  if answered else None),
        "answer_rate": len(answered) / len(flat) if flat else None,
        "position_bias": (sum(1.0 - 2.0 * p for p in picked_1) / len(picked_1)) if picked_1 else None,
        "consistent_accuracy": (sum(a["picked_major"] == 1 and b["picked_major"] == 1
                                    for a, b in both) / len(both)) if both else None,
        "length_corr": pearson([float(chars[0] > chars[1]) for _, chars in lengths],
                               [1.0 if order["pick"] == "1" else 0.0 for order, _ in lengths]),
        "confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "disagree_rate": (sum(a["picked_major"] != b["picked_major"] for a, b in both) / len(both)
                          if both else None),
    }


def judge_metrics(pairs):
    """The same metrics pooled over every cross-model run of each judge, plus its self-pairs
    (where accuracy is meaningless but position bias is the control)."""
    pooled, selves = defaultdict(list), defaultdict(list)
    for (major, minor, frac), items in pairs.items():
        for cells in items.values():
            for judge, cell in cells.items():
                (selves if major == minor else pooled)[judge].append(cell["orders"])
    out = {}
    for judge in sorted(set(pooled) | set(selves)):
        entry = metrics(pooled.get(judge, []))
        entry["n_runs"] = len({(major, minor, frac) for (major, minor, frac), items in pairs.items()
                               if major != minor
                               and any(judge in cells for cells in items.values())})
        if selves.get(judge):
            self_entry = metrics(selves[judge])
            entry["self_position_bias"] = self_entry["position_bias"]
            entry["self_n"] = self_entry["n"]
        out[judge] = entry
    return out


def runs_from_pairs(pairs, accuracy):
    """One point per (judge, major, minor, frac): the plotted score plus the judge metrics."""
    runs = []
    for (major, minor, frac), items in sorted(pairs.items()):
        judges = {judge for cells in items.values() for judge in cells}
        for judge in sorted(judges):
            cells = [cells[judge]["orders"] for cells in items.values() if judge in cells]
            m = metrics(cells)
            answered = round(m["answer_rate"] * m["n"]) if m["answer_rate"] is not None else 0
            lo, hi = wilson(round((m["score"] or 0) * answered), answered)
            runs.append({
                "run_id": f"{judge}__{major}__{minor}__{frac}",
                "judge": judge, "major": major, "minor": minor, "frac": frac,
                "d_acc": abs(accuracy[major] - accuracy[minor]),
                # the plotted score is over answered samples; metrics.accuracy counts a
                # missing verdict as wrong, the way inspect's accuracy metric does
                "score": m["score"], "score_lo": lo, "score_hi": hi,
                "n_items": len(cells),
                "slot_a": (1 - m["position_bias"]) / 2 if m["position_bias"] is not None else None,
                "disagree_rate": m["disagree_rate"],
                "noanswer_rate": 1 - m["answer_rate"] if m["answer_rate"] is not None else None,
                "is_self": major == minor,
                "metrics": m,
                # The judge is not one of the ranked models here, so neither subset filter applies.
                "weak_judge": False, "old_judge": False, "arm_a": True,
            })
    return runs


def fits_from_pairs(pairs, accuracy):
    """Per judge: score = sigmoid(a + d * gap), fitted on counterbalanced item scores.

    The item score averages the two orders, which cancels position bias; self-pairs are the
    control the fit is read against, so they stay out of it (stats.sigmoid_fit).
    """
    points = defaultdict(lambda: ([], []))
    for (major, minor, frac), items in pairs.items():
        if major == minor or frac != "full":
            continue
        gap = abs(accuracy[major] - accuracy[minor])
        for cells in items.values():
            for judge, cell in cells.items():
                picks = [order["picked_major"] for order in cell["orders"]
                         if order is not None and order["picked_major"] is not None]
                if picks:
                    points[judge][0].append(gap)
                    points[judge][1].append(sum(picks) / len(picks))
    fits = {}
    for judge, (gaps, scores) in points.items():
        fit = sigmoid_fit(gaps, scores)
        if fit is not None:
            a, d, se_a, se_d = fit
            fits[judge] = {"a": a, "d": d, "se_a": se_a, "se_d": se_d, "n_items": len(gaps)}
    return fits


# Contiguous quartile subsets of the domain, offered as x-axis choices. Q4 is the ground
# truth the pairs were oriented by; the others say whether the gap has to be measured there.
SUBSETS = [((1,), "Q1"), ((2,), "Q2"), ((3,), "Q3"), ((4,), "Q4"),
           ((1, 2), "Q1–Q2"), ((2, 3), "Q2–Q3"), ((3, 4), "Q3–Q4"),
           ((1, 2, 3), "Q1–Q3"), ((2, 3, 4), "Q2–Q4"), ((1, 2, 3, 4), "all")]


def subset_key(quartiles):
    if len(quartiles) == 4:
        return "all"
    return f"q{quartiles[0]}" if len(quartiles) == 1 else f"q{quartiles[0]}-{quartiles[-1]}"


def domain_accuracies(domain_name):
    """Per-model accuracy on each quartile subset: {subset key: {model: accuracy}}, [] if the
    domain artefacts are missing (run from the repo root, after `uv run domain`).

    Same definition as ExperimentConfig.accuracy: repeats averaged per item, items equally
    weighted, so the Q_dst column reproduces the acc_major/acc_minor in the log metadata.
    """
    config = ROOT / "configs" / "domain" / f"{domain_name}.yml"
    if not config.exists():
        return {}, []
    domain = DomainConfig.from_cli(str(config))
    try:
        responses = domain.load_responses().merge(domain.load_items()[["bench", "item", "quartile"]])
    except FileNotFoundError as err:
        print(f"  x-axis limited to the logged Q_dst accuracy: {err}")
        return {}, []
    columns, xmetrics = {}, []
    for quartiles, label in SUBSETS:
        subset = responses[responses.quartile.isin(quartiles)]
        if subset.empty:
            continue
        by_model = subset.groupby(["model", "bench", "item"]).correct.mean().groupby("model").mean()
        key = subset_key(quartiles)
        columns[key] = {model: float(value) for model, value in by_model.items()}
        xmetrics.append({"key": key, "label": label,
                         "title": f"capability gap ({label} accuracy difference)"})
    return columns, xmetrics


def export(log_dir, dataset_id=None, label=None, out_root=VIEWER_DATA):
    log_dir = Path(log_dir)
    transcripts, options, answers, questions, pairs, accuracy, condition, skipped = collect(log_dir)
    if not pairs:
        raise ValueError(f"no successful judge logs in {log_dir}")

    runs = runs_from_pairs(pairs, accuracy)
    judges = sorted({run["judge"] for run in runs})
    # include_reasoning is a separate condition, judged into its own hash dir: keep it in the id
    cot = "-cot" if condition["include_reasoning"] else ""
    dataset_id = dataset_id or f"{condition['domain']}-q{condition['q_src']}{cot}-{judges[0]}"
    label = label or (f"{condition['domain'].upper()} Q{condition['q_src']} "
                      f"(truth Q{condition['q_dst']}) — {judges[0]}"
                      + (", CoT" if condition["include_reasoning"] else ", no CoT"))
    out = out_root / dataset_id

    models = {model: {"label": model, "accuracy": acc, "acc_domain": acc, "release": "", "arm": "A"}
              for model, acc in sorted(accuracy.items())}
    for judge in judges:  # a judge that is not itself ranked has no ground-truth accuracy
        models.setdefault(judge, {"label": judge, "accuracy": None, "release": "", "arm": "judge"})

    columns, xmetrics = domain_accuracies(condition["domain"])
    truth = subset_key((condition["q_dst"],))
    for key, by_model in columns.items():
        for model, meta in models.items():
            if model in by_model:
                meta[f"acc_{key}"] = by_model[model]
    for model, acc in accuracy.items():  # the logged ground truth must be the Q_dst column
        logged = models[model].get(f"acc_{truth}")
        assert logged is None or abs(logged - acc) < 1e-9, f"{model}: {logged} != {acc} on {truth}"
    # Kendall tau-b of the fitted ability against each x-axis subset, so the ranking view can
    # state the correlation for the axis it is actually showing. ranking.scoring reports the
    # same statistic on Q1..Q4 and all, but means accuracy over response rows instead of
    # averaging repeats per item first, so its numbers sit a few points away.
    ranking = load_csv(log_dir / "ranking.csv", floats=("ability", "n_wins"),
                       ints=("rank", "n_comparisons", "n_ties"))
    taus = {}
    for metric in xmetrics:
        points = [(row["ability"], models[row["model"]][f"acc_{metric['key']}"])
                  for row in ranking if f"acc_{metric['key']}" in models.get(row["model"], {})]
        if len(points) > 2:
            taus[metric["key"]] = float(kendalltau([p[0] for p in points], [p[1] for p in points]).statistic)

    if not xmetrics:  # no domain artefacts: the logged Q_dst accuracy is all we have
        xmetrics = [{"key": "domain", "label": f"Q{condition['q_dst']}",
                     "title": f"capability gap (Q{condition['q_dst']} accuracy difference)"}]
        truth = "domain"

    for sub in ("transcripts", "options", "answers", "pairs"):
        if (out / sub).exists():
            for stale in (out / sub).glob("*.json"):
                stale.unlink()
    for model, items in transcripts.items():
        dump(items, out / "transcripts" / f"{model}.json")
    for model, items in options.items():
        dump(items, out / "options" / f"{model}.json")
    for model, items in answers.items():
        dump(items, out / "answers" / f"{model}.json")
    for (major, minor, frac), items in pairs.items():
        dump(items, out / "pairs" / f"{major}__{minor}__{frac}.json")
    dump({"models": models, "judges": judges,
          "colors": {judge: PALETTE[i % len(PALETTE)] for i, judge in enumerate(judges)},
          "questions": questions, "fits": fits_from_pairs(pairs, accuracy), "runs": runs,
          "default_xmetric": truth, "xmetrics": xmetrics,
          "available_xmetrics": [m["key"] for m in xmetrics],
          "judge_metrics": judge_metrics(pairs), "taus": taus,
          "metric_label": f"Q{condition['q_dst']} accuracy", "condition": condition,
          "ranking": ranking,
          "scores": load_csv(log_dir / "scores.csv", floats=("correlation",),
                             ints=("n_models", "n_items")),
          "source": str(log_dir)}, out / "index.json")

    ranked = len(json.loads((out / "index.json").read_text())["ranking"])
    print(f"{dataset_id}: {len(runs)} runs, {len(questions)} items, {len(transcripts)} transcript sets"
          + (f", {ranked} ranked models" if ranked else ", no ranking.csv (run `uv run rank`)")
          + (f", {skipped} unsuccessful logs skipped" if skipped else ""))
    return {"id": dataset_id, "label": label}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dirs", nargs="+", type=Path, help="judge log dirs, one per condition")
    parser.add_argument("--id", help="dataset id (single dir only); default <domain>-q<src>-<judge>")
    parser.add_argument("--label", help="dropdown label (single dir only)")
    parser.add_argument("--out", type=Path, default=VIEWER_DATA)
    args = parser.parse_args()
    if (args.id or args.label) and len(args.log_dirs) > 1:
        parser.error("--id/--label only make sense with a single log dir")

    entries = [export(d, args.id, args.label, args.out) for d in args.log_dirs]
    menu_path = args.out / "datasets.json"
    menu = json.loads(menu_path.read_text()) if menu_path.exists() else []
    new = {entry["id"] for entry in entries}
    dump(entries + [item for item in menu if item["id"] not in new], menu_path)
    print(f"datasets.json: {len(entries)} dataset(s) at the top of the menu")


if __name__ == "__main__":
    main()
