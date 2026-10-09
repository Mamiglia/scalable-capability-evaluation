"""Refused samples are "not scored", not failures.

A safety-classifier refusal says nothing about capability, so a refused item must drop out of the
theta* fit and of judging instead of counting as wrong. Two shapes, both from the model events:

- Anthropic (claude-fable-5): a normal completion with ``stop_reason == "content_filter"``
  (OpenRouter's rendering of Anthropic's ``refusal``); the harbor scorer then scores the sample 0.
- OpenAI (gpt-5.6-sol): HTTP 403 "flagged for possible cybersecurity risk" -> a model event error;
  the sample errors out (already unscored) unless it was cancelled and scored on partial work.

For every such sample that carries a score: drop the score, drop its reduction entry, recompute the
header accuracy over the samples that still have one, and stamp the edit (eval.metadata["edits"],
``sample.metadata["refusal"] = True`` on each sample). Dry run by default.

    PYTHONPATH=src .venv/bin/python src/challenges/frontier/unscore_refusals.py [--write] [--also ID ...] [log.eval ...]

``--also`` names samples whose refusal is not in the log (gpt-5.6-sol `vulnerable-secret`: 403 on the
first attempt, then a cancelled retry scored on partial work; the first attempt lives only in the
inspect ctl history).

Note for the integration step: ``evals.responses`` maps a sample with no score to ``correct=False``;
filter on ``sample.metadata["refusal"]`` (or on an empty ``scores``) before fitting theta*.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

sys.path.insert(0, str(Path("scripts/audit/logs")))
from eval_edits import stamp_edit  # noqa: E402

TOOL = "src/challenges/frontier/unscore_refusals.py"
DEFAULT = ["data/terminal_bench_2_1/0af2ba2c/*/0/*.eval", "data/swe_bench_mini/2d1e487f/*/0/*.eval"]
OPENAI_403 = "flagged for possible cybersecurity"


def refused(sample) -> str | None:
    """Why the sample counts as refused, or None."""
    for ev in sample.events or []:
        if ev.event != "model":
            continue
        if ev.output and ev.output.stop_reason == "content_filter":
            return "content_filter"
        if ev.error and OPENAI_403 in str(ev.error):
            return "openai_403"
    if sample.error and OPENAI_403 in str(sample.error):
        return "openai_403"
    return None


def is_correct(score) -> bool:
    return bool(score and score.value in ("C", 1, 1.0, True))


def unscore(path: Path, write: bool, also: set[str] = frozenset()) -> list[str]:
    log = read_eval_log(str(path), resolve_attachments=True)
    hits = {str(s.id): why for s in log.samples
            if (why := refused(s) or (str(s.id) in also and "manual")) and s.scores}
    if not hits:
        return []
    names = sorted({n for s in log.samples for n in (s.scores or {})})
    before = {n: sum(is_correct((s.scores or {}).get(n)) for s in log.samples) / len(log.samples) for n in names}
    for s in log.samples:
        if str(s.id) in hits:
            s.scores = {}
            s.metadata = {**(s.metadata or {}), "refusal": hits[str(s.id)]}
    for reduction in log.reductions or []:
        reduction.samples = [e for e in reduction.samples if str(e.sample_id) not in hits]
    scored = {n: [s for s in log.samples if (s.scores or {}).get(n)] for n in names}
    after = {n: sum(is_correct(s.scores[n]) for s in scored[n]) / max(1, len(scored[n])) for n in names}
    for sc in log.results.scores if log.results else []:
        if "accuracy" in sc.metrics and sc.name in after:
            sc.metrics["accuracy"].value = after[sc.name]
    if write:
        stamp_edit(log, tool=TOOL, reason="refused samples are not scored (decision 2026-09-17): "
                   + ", ".join(f"{i} ({w})" for i, w in sorted(hits.items())),
                   source=path, samples_changed=len(hits),
                   before={"accuracy": before[names[0]], "scored": len(log.samples)},
                   after={"accuracy": after[names[0]], "scored": len(scored[names[0]])})
        tmp = path.with_name(f".unscore-{path.name}")
        try:
            write_eval_log(log, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"{'wrote' if write else 'would unscore'} {len(hits)} in {path.name[:40]}: "
          f"acc {before[names[0]]:.3f} -> {after[names[0]]:.3f} over {len(scored[names[0]])} scored; "
          + ", ".join(f"{i.split('/')[-1]} ({w})" for i, w in sorted(hits.items())))
    return sorted(hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--also", action="append", default=[], help="sample id to unscore regardless of its events")
    args = ap.parse_args()
    logs = args.logs or [Path(p) for pat in DEFAULT for p in sorted(glob.glob(pat))]
    total = sum(len(unscore(p, args.write, set(args.also))) for p in logs)
    print(f"{len(logs)} logs, {total} refused samples {'unscored' if args.write else 'would be unscored (dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
