"""Flatten gpqa-pilot-rd2 eval logs into verdicts.csv and models.csv.

verdicts.csv has one row per judged sample (item x order x run), including the
transcript lengths as the judge saw them. models.csv has one row per transcript
model: accuracy, logit, theta, release. Prints a coverage summary.

Usage:
  uv run python scripts/analyses/digest_verdicts.py
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

from judge.dataset import USER_PROMPT, format_options, strip_answer, truncate_pair
from judge.util import extract_score, extract_transcript

from comparisons import JUDGE_ALIASES, MODELS

MAX_LEN = 65536  # judge_task default; run_judge_pilot does not override it
CAPABILITIES = Path("irt_results/epoch_gpqa/gpqa_capabilities.csv")

_logs: dict[Path, object] = {}
_pairs: dict[tuple, dict] = {}


def load_log(path):
    p = Path(path).resolve()
    if p not in _logs:
        _logs[p] = read_eval_log(str(p))
    return _logs[p]


class PairItem(NamedTuple):
    """One item's judged content for a pair, before the two orders are assigned."""
    question: str
    t_major: str
    t_minor: str
    s_major: Any        # the sample; the prompt rebuild needs its option order
    s_minor: Any
    reasoning_len_major: int
    reasoning_len_minor: int


def pair_items(major_path, minor_path, include_reasoning, require_both_correct,
               truncate_frac=None):
    """item -> PairItem.

    Mirrors build_dataset's epoch selection (first eligible epoch per item;
    self-pairs take the first two) and its truncation. Digest verifies the
    rebuild per item against the logged judge input, so drift from the
    builder is caught.
    """
    key = (str(major_path), str(minor_path), include_reasoning,
           require_both_correct, truncate_frac)
    if key in _pairs:
        return _pairs[key]

    def by_id(log):
        grouped = {}
        for s in log.samples:
            if not require_both_correct or extract_score(s):
                grouped.setdefault(s.id, []).append(s)
        for g in grouped.values():
            g.sort(key=lambda s: getattr(s, "epoch", 0))
        return grouped

    self_pair = Path(major_path).resolve() == Path(minor_path).resolve()
    major = by_id(load_log(major_path))
    minor = major if self_pair else by_id(load_log(minor_path))

    items = {}
    for sid in sorted(minor.keys() & major.keys(), key=str):
        if self_pair:
            if len(major[sid]) < 2:
                continue
            a, b = major[sid][0], major[sid][1]
        else:
            a, b = major[sid][0], minor[sid][0]
        t_major = extract_transcript(a, include_reasoning=include_reasoning)[:MAX_LEN]
        t_minor = extract_transcript(b, include_reasoning=include_reasoning)[:MAX_LEN]
        # Measured before truncation so the degenerate-transcript filter keeps
        # the same items in every truncation condition.
        rlen_major, rlen_minor = len(strip_answer(t_major)), len(strip_answer(t_minor))
        if truncate_frac is not None:
            t_major, t_minor = truncate_pair(t_major, t_minor, truncate_frac)
        items[str(sid)] = PairItem(a.input, t_major, t_minor, a, b, rlen_major, rlen_minor)
    _pairs[key] = items
    return items


def digest(log_root: Path, config: Path):
    cfg = yaml.safe_load(config.read_text())
    by_path = {Path(p).resolve(): name for name, p in cfg["transcripts"].items()}
    # Only runs specified in the config are digested; stray logs from earlier
    # pilot iterations are skipped.
    def judge_short(j):
        s = j.split("/")[-1]
        return JUDGE_ALIASES.get(s, s)
    configured = {(judge_short(j), maj, mnr) for j, maj, mnr in cfg["comparisons"]}

    rows: dict[tuple, dict] = {}
    n_files = n_not_success = n_unlisted = n_replaced = n_unmatched = n_len_mismatch = 0

    # Filename sort puts timestamps in order, so later files win on duplicates.
    for f in sorted(log_root.rglob("*.eval")):
        n_files += 1
        log = read_eval_log(str(f))
        if log.status != "success":
            n_not_success += 1
            continue
        meta = log.eval.metadata or {}
        judge_raw = log.eval.model.split("/")[-1]
        judge = JUDGE_ALIASES.get(judge_raw, judge_raw)
        major = by_path.get(Path(meta["major_path"]).resolve())
        minor = by_path.get(Path(meta["minor_path"]).resolve())
        if major is None or minor is None:
            print(f"SKIP {f}: transcript paths not in {config}")
            continue
        if (judge, major, minor) not in configured:
            n_unlisted += 1
            continue
        tf = meta.get("truncate_frac")
        items = pair_items(meta["major_path"], meta["minor_path"],
                           meta["include_reasoning"], meta["require_both_correct"],
                           truncate_frac=tf)
        # Phase lives in the parent dir (phase-N) or, after log normalization,
        # in a _pN filename token.
        phase = f.parent.name
        if phase.startswith("phase-"):
            phase = int(phase.split("-")[1])
        else:
            m = re.search(r"_p(\d)_", f.name)
            phase = int(m.group(1)) if m else None

        n_correct = 0
        for s in log.samples or []:
            item, slot = str(s.id).rsplit("_", 1)
            score = next(iter((s.scores or {}).values()), None)
            picked = getattr(score, "answer", None)
            picked_major = (1.0 if picked == slot else 0.0) if picked in ("1", "2") else np.nan
            conf = (score.metadata or {}).get("confidence", np.nan) if score and score.metadata else np.nan

            it = items.get(item)
            len_major = len_minor = rlen_major = rlen_minor = np.nan
            if it is None:
                n_unmatched += 1
            else:
                rlen_major, rlen_minor = it.reasoning_len_major, it.reasoning_len_minor
                major_first = slot == "1"
                t_1, t_2 = (it.t_major, it.t_minor) if major_first else (it.t_minor, it.t_major)
                s_1, s_2 = (it.s_major, it.s_minor) if major_first else (it.s_minor, it.s_major)
                expected = USER_PROMPT.format(
                    question=it.question,
                    options_1=format_options(s_1, 1),
                    options_2=format_options(s_2, 2),
                    transcript_1=t_1,
                    transcript_2=t_2,
                )
                if isinstance(s.input, str) and len(s.input) == len(expected):
                    len_major, len_minor = len(it.t_major), len(it.t_minor)
                else:
                    n_len_mismatch += 1

            n_correct += picked == slot
            key = (judge, major, minor, item, slot, tf)
            if key in rows:
                n_replaced += 1
            rows[key] = dict(
                judge=judge, major=major, minor=minor, item=item,
                slot_of_major=slot, picked=picked, picked_major=picked_major,
                confidence=conf, len_major=len_major, len_minor=len_minor,
                reasoning_len_major=rlen_major, reasoning_len_minor=rlen_minor,
                truncate_frac=tf, phase=phase, log_file=str(f),
            )

        # The flattened rows must reproduce the log's own accuracy metric.
        header_acc = log.results.scores[0].metrics["accuracy"].value
        if log.samples and abs(n_correct / len(log.samples) - header_acc) > 1e-9:
            print(f"WARN {f}: rows give {n_correct / len(log.samples):.4f}, "
                  f"header says {header_acc:.4f}")

    df = pd.DataFrame(rows.values())
    print(f"{n_files} eval files: {n_not_success} not-success, "
          f"{n_unlisted} runs not in config skipped, "
          f"{n_replaced} rows replaced by newer logs, "
          f"{n_unmatched} samples without a rebuilt item, "
          f"{n_len_mismatch} length-check mismatches")
    present = {(r["judge"], r["major"], r["minor"]) for r in rows.values()}
    for j, maj, mnr in sorted(configured - present):
        print(f"NO LOG YET: {j} on {maj}-vs-{mnr}")
    return df


def model_table(config: Path):
    transcripts = yaml.safe_load(config.read_text())["transcripts"]
    caps = (pd.read_csv(CAPABILITIES).set_index("model")["theta_1pl"]
            if CAPABILITIES.exists() else pd.Series(dtype=float))

    rows = []
    for name, p in transcripts.items():
        log = load_log(p)
        scores = [extract_score(s) for s in log.samples]
        acc = float(np.mean(scores))
        n_items = len({s.id for s in log.samples})
        meta = MODELS.get(name, {})
        rows.append(dict(
            model=name,
            arm="B" if "external_data" in Path(p).parts else "A",
            n_samples=len(scores),
            n_items=n_items,
            n_epochs=len(scores) // n_items,
            accuracy=acc,
            logit=math.log(acc / (1 - acc)),
            theta_1pl=caps.get(Path(p).stem, np.nan),
            label=meta.get("label", name),
            release=meta.get("release"),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", type=Path, default=Path("logs/gpqa-pilot-rd2"))
    ap.add_argument("--config", type=Path, default=Path("configs/judge_pilot.yaml"))
    ap.add_argument("--out", type=Path, default=Path("judge_results/gpqa-pilot-rd2"))
    args = ap.parse_args()

    df = digest(args.log_root, args.config)
    models = model_table(args.config)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "verdicts.csv", index=False)
    models.to_csv(args.out / "models.csv", index=False)

    runs = (df.groupby(["judge", "major", "minor"])
              .agg(n_rows=("item", "size"), n_items=("item", "nunique"))
              .reset_index())
    print(f"\n{len(runs)} runs, {len(df)} verdict rows -> {args.out}/verdicts.csv")
    print(runs.to_string(index=False))
    n_theta = models["theta_1pl"].notna().sum()
    print(f"\n{len(models)} models -> {args.out}/models.csv ({n_theta} with theta)")
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
