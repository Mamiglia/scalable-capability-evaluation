"""Record edits that were applied to a log but never written into it.

``aime_scorer`` scores by reading ``output.completion``. A sample scored correct whose
``score.answer`` does not appear in that field cannot have been scored by the scorer: it was
rescored afterwards. That is routine and fine -- but the log has to say so, or the next
person cannot tell a corrected score from an as-collected one, and re-running the fixer
looks like a no-op for reasons nobody can reconstruct.

This finds that signature and adds the missing record (``eval.metadata["edits"]`` plus the
``edited`` tags), describing what was detected rather than claiming to know who did it.
Logs that already carry a record are left alone.

    uv run python scripts/audit/logs/stamp_unrecorded_edits.py            # dry run
    uv run python scripts/audit/logs/stamp_unrecorded_edits.py --write
"""

import argparse
import os
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

from configs.paths import collection_logs, is_successful_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_edits import stamp_edit  # noqa: E402

# Scorers that read output.completion, so the check below is meaningful for them.
COMPLETION_SCORERS = {"aime_scorer"}


def unrecorded(path: Path) -> tuple[list[str], bool]:
    """Sample ids whose score the scorer could not have produced, and whether it is recorded."""
    log = read_eval_log(str(path), header_only=True)
    recorded = bool((log.eval.metadata or {}).get("edits"))
    full = read_eval_log(str(path), resolve_attachments=True)
    ghosts = []
    for sample in full.samples or []:
        for name, score in (sample.scores or {}).items():
            if name not in COMPLETION_SCORERS:
                continue
            if score.value not in ("C", 1, 1.0, True) or not score.answer:
                continue
            if str(score.answer) not in (sample.output.completion or ""):
                ghosts.append(str(sample.id))
    return ghosts, recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    paths = [p for root in args.roots for p in collection_logs(root) if is_successful_eval(p)]
    stamped = 0
    for path in paths:
        try:
            ghosts, recorded = unrecorded(path)
        except Exception as exc:
            print(f"SKIP {path}: {str(exc).splitlines()[0][:70]}", file=sys.stderr)
            continue
        if not ghosts or recorded:
            continue
        print(f"{len(ghosts):3d} rescored sample(s), unrecorded: {path}")
        stamped += 1
        if not args.write:
            continue
        log = read_eval_log(str(path), resolve_attachments=True)
        stamp_edit(
            log,
            tool="scripts/audit/logs/stamp_unrecorded_edits.py",
            reason=(
                "record added retrospectively: "
                f"{len(ghosts)} sample(s) are scored correct with an answer that does not appear "
                "in output.completion, which the scorer reads, so their scores were edited after "
                "collection by a tool that left no record. Sample ids: "
                f"{', '.join(sorted(ghosts))}"
            ),
            source=path,
            samples_changed=0,  # this pass records the edit, it does not change any score
            before={},
            after={},
            transcripts_modified=False,
        )
        tmp = path.with_name(f".stamp-{path.name}")
        try:
            write_eval_log(log, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    print(f"\n{stamped} log(s) with unrecorded rescoring{'' if args.write else ' (dry run)'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
