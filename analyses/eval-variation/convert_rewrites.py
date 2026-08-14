"""Turn rewrites jsonl into judge-readable eval logs, one per model.

Copies the baseline log, keeps only first-correct-epoch samples whose rewrite
kept the original answer letter, and swaps the assistant message text for the
rewritten version. Scores/input/choices stay original: correctness is the
original attempt's, the rewrite is presentation only.

Usage: convert_rewrites.py --mode self|standard
"""

import argparse
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

sys.path.insert(0, str(Path(__file__).parent))
from gen_configs import CONDITIONS, SUBJECTS_BY_ECI, resolve_log  # noqa: E402
from run import first_correct_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("self", "standard"), required=True)
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()
    jsonl = args.jsonl or Path(f"judge_results/eval_prompt_variation/rewrites_{args.mode}.jsonl")
    out_root = args.out_root or Path(f"data/eval-variation/gpqa-rewrite-{args.mode}")

    rew = {}
    for line in open(jsonl):
        r = json.loads(line)
        if "rewritten" in r:
            rew[(r["model"], r["item"])] = r

    print(f"{'model':28} {'kept':>5} {'no-rewrite':>10} {'answer-changed':>14}")
    for m, _ in SUBJECTS_BY_ECI:
        path = resolve_log(CONDITIONS["baseline"], m)
        keep_ids = first_correct_samples(path)
        log = read_eval_log(path)
        kept, missing, infidel = [], 0, 0
        wanted = {(i, getattr(s, "epoch", 0)) for i, s in keep_ids.items()}
        for s in log.samples:
            if (s.id, getattr(s, "epoch", 0)) not in wanted:
                continue
            r = rew.get((m, str(s.id)))
            if r is None:
                missing += 1
                continue
            orig_letter = (s.scores["choice"].answer or "").strip().upper() if s.scores else ""
            if r["new_answer"] is None or (orig_letter and r["new_answer"] != orig_letter):
                infidel += 1
                continue
            for msg in s.messages:
                if getattr(msg, "role", None) == "assistant":
                    msg.content = r["rewritten"]
            try:
                s.output.choices[0].message.content = r["rewritten"]
            except Exception:
                pass
            kept.append(s)
        log.samples = kept
        sub = f"{m}-rescored" if m == "gemini-3.1-flash-lite" else m
        out = out_root / sub / "0" / "rewritten.eval"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(log, str(out))
        print(f"{m:28} {len(kept):5d} {missing:10d} {infidel:14d}")


if __name__ == "__main__":
    main()
