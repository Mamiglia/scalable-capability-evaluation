"""Export protocol-compatible verdict rows from the pinned HF judging mirror."""

import argparse
import json
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

PROMPT = {"gpqa": "assets/system_prompt/mcq.txt",
          "swe_bench_mini": "assets/system_prompt/default.txt"}


def compatible(meta, bench):
    return (meta.get("require_both_correct") is True
            and meta.get("include_reverse") is True
            and meta.get("include_reasoning") is False
            and meta.get("truncate_frac") is None
            and meta.get("system_prompt") == PROMPT[bench])


def rollout(path):
    parts = Path(path).parts
    if len(parts) < 5 or parts[0] != "data":
        raise ValueError(path)
    return parts[3], int(parts[4])


def ingest(root):
    rows, skipped = {}, {"unreadable": 0, "incompatible": 0, "bad_path": 0}
    for bench in PROMPT:
        for path in sorted((root / "pilot" / bench).rglob("*.eval")):
            if path.name.startswith("._"):
                continue
            try:
                log = read_eval_log(str(path))
            except Exception:
                skipped["unreadable"] += 1
                continue
            meta = log.eval.metadata or {}
            if log.status != "success" or not compatible(meta, bench):
                skipped["incompatible"] += 1
                continue
            try:
                major, epoch_major = rollout(meta["major_path"])
                minor, epoch_minor = rollout(meta["minor_path"])
            except (KeyError, TypeError, ValueError):
                skipped["bad_path"] += 1
                continue
            judge = log.eval.model.split("/")[-1]
            for sample in log.samples or []:
                try:
                    item, slot = str(sample.id).rsplit("_", 1)
                except ValueError:
                    continue
                score = next(iter((sample.scores or {}).values()), None)
                picked = getattr(score, "answer", None)
                if slot not in ("1", "2") or picked not in ("1", "2"):
                    continue
                major_first = slot == "1"
                left, right = (major, minor) if major_first else (minor, major)
                epochs = (epoch_major, epoch_minor) if major_first else (epoch_minor, epoch_major)
                row = {"bench": bench, "judge": judge, "item": item,
                       "model_left": left, "model_right": right,
                       "epoch_left": epochs[0], "epoch_right": epochs[1],
                       "judge_epoch": int(sample.epoch or 1),
                       "pick_left": int(picked == "1")}
                key = tuple(row[column] for column in row if column != "pick_left")
                if key not in rows or log.eval.created > rows[key][0]:
                    rows[key] = (log.eval.created, row)
    return pd.DataFrame(row for _, row in rows.values()), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    verdicts, skipped = ingest(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    verdicts.to_parquet(args.out / "verdicts.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "source": str(args.root), "protocol_prompts": PROMPT,
        "rows": len(verdicts), "skipped": skipped,
        "judges": sorted(verdicts.judge.unique()),
    }, indent=2) + "\n")
    print(verdicts.groupby(["bench", "judge"]).size().unstack(fill_value=0).to_string())
    print(f"{len(verdicts)} verdicts; skipped {skipped}")


if __name__ == "__main__":
    main()
