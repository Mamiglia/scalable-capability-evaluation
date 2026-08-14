"""Build sandbag-robustness pairs for the FT judge from the style-capability
generations.

Two matched conditions over the same strong>weak model pairs and items:
  natural         strong baseline  vs weaker baseline
  spoof_uncertain strong told-uncertain vs weaker baseline
Both-correct only, counter-balanced. The spoofed side keeps the strong
model's true capability as its label (accuracy is held under the
instruction), so a judge that reads capability should pick it in both
conditions; one that reads confidence tone flips on spoof_uncertain.

Reads judge_results/style_capability/{baseline_answers,instruct_answers}.jsonl
and items.json; writes pairs_spoof.parquet in the pod_verdicts format.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from prep_pairs import make_pair_rows, render_choices, check_counterbalance

STRONG = ["qwen3.5-122b-a10b", "qwen3.6-35b-a3b"]  # the told-uncertain (sandbag) side


def is_correct(v) -> bool:
    return str(v).lower() == "true"


def load_answers(path: Path, label=None):
    """(model, sample_id) -> [(epoch, correct, answer)], optionally filtered by label."""
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if label is not None and r.get("label") != label:
            continue
        out.setdefault((r["target"], r["sample_id"]), []).append(
            (int(r["epoch"]), is_correct(r["correct"]), r["answer"]))
    return out


def first_correct(store, model, sid):
    """The earliest correct epoch's answer text, or None if the model missed the item."""
    for _epoch, ok, ans in sorted(store.get((model, sid), [])):
        if ok:
            return ans
    return None


def model_accuracy(store):
    acc = {}
    for (model, _sid), hits in store.items():
        acc.setdefault(model, []).extend(ok for _e, ok, _a in hits)
    return {m: sum(v) / len(v) for m, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style-dir", default="judge_results/style_capability")
    ap.add_argument("--out", default="judge_results/judge_ft/pairs_spoof.parquet")
    args = ap.parse_args()
    sd = Path(args.style_dir)

    items = {it["sample_id"]: it for it in json.load(open(sd / "items.json"))}
    baseline = load_answers(sd / "baseline_answers.jsonl")
    uncertain = load_answers(sd / "instruct_answers.jsonl", label="uncertain")

    acc = model_accuracy(baseline)
    print("baseline accuracy:", {m: round(a, 3) for m, a in sorted(acc.items(), key=lambda x: -x[1])})

    rows = []
    for s in STRONG:
        weaker = [o for o in acc if acc[o] < acc[s]]
        for o in weaker:
            for sid, it in items.items():
                opts = render_choices(it["choices"])
                s_base = first_correct(baseline, s, sid)
                s_unc = first_correct(uncertain, s, sid)
                o_base = first_correct(baseline, o, sid)
                if o_base is None:
                    continue
                weak_meta = dict(model=o, cap=acc[o], text=o_base, options=opts)
                # matched: only emit a condition when its strong-side transcript is correct
                if s_base is not None:
                    strong = dict(model=s, cap=acc[s], text=s_base, options=opts)
                    for r in make_pair_rows(sid, strong, weak_meta, it["question"]):
                        rows.append({**r, "split": "natural"})
                if s_unc is not None:
                    strong = dict(model=f"{s}#uncertain", cap=acc[s], text=s_unc, options=opts)
                    for r in make_pair_rows(sid, strong, weak_meta, it["question"]):
                        rows.append({**r, "split": "spoof_uncertain"})

    pairs = pd.DataFrame(rows)
    check_counterbalance(pairs)  # per split: two order-swapped rows, equal label counts
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out)
    print(pairs.groupby("split").pair_uid.nunique().rename("pairs"))
    print("label balance:", pairs.groupby("split").label.mean().round(3).to_dict())
    print(f"wrote {len(pairs)} rows to {args.out}")


if __name__ == "__main__":
    main()
