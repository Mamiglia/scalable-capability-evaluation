"""Provenance for .eval logs that an audit edits (rescoring, re-grading, patching).

A `.eval` is a research artifact: once it has been changed, nothing outside the file
records that. `data/collection_runs.md` and git history live in a different repo and get
detached from the log the moment it is copied. So the record goes *inside* the log:

  eval.metadata["edits"]  -- append-only list of edit records
  eval.tags               -- "edited" plus "edited:<tool>", so a header-only read sees it

Use from any script that writes a modified log:

    from eval_edits import stamp_edit, edit_history
    stamp_edit(log, tool="rescore_mcq.py", reason="tolerant ANSWER regex",
               source=input_path, samples_changed=185,
               before={"accuracy": 0.061}, after={"accuracy": 0.778})
    write_eval_log(log, out)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stamp_edit(
    log: Any,
    *,
    tool: str,
    reason: str,
    source: Path | str | None = None,
    samples_changed: int | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    **extra: Any,
) -> dict:
    """Append an edit record to the log's own metadata and tag it. Returns the record."""
    record = {
        "tool": tool,
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **({"source": str(source)} if source else {}),
        **({"samples_changed": samples_changed} if samples_changed is not None else {}),
        **({"before": before} if before else {}),
        **({"after": after} if after else {}),
        **extra,
    }
    metadata = dict(log.eval.metadata or {})
    metadata["edits"] = [*metadata.get("edits", []), record]
    log.eval.metadata = metadata
    log.eval.tags = sorted({*(log.eval.tags or []), "edited", f"edited:{Path(tool).stem}"})
    return record


def edit_history(log: Any) -> list[dict]:
    """Edit records of a log (header-only read is enough)."""
    return list((log.eval.metadata or {}).get("edits", []))
