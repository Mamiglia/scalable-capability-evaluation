"""Strip the '[Assistant]: ' role label so every condition set presents the same way.

The label is a transcript-extraction artefact, not model output. Haiku removed it from all
its rewrites, codex kept it on all of its, and the originals all carry it - which lets a judge
tell the conditions apart on formatting alone. Stripping is the direction that works: adding
it back would mean inserting text no rewriter produced.

Idempotent, so it is safe to re-run and safe on files already clean.

Usage:
  uv run python scripts/analyses/rewrites/normalise_prefix.py judge_results/rewrites/*.jsonl
"""
import json
import re
import sys
from pathlib import Path

ROLE_PREFIX_RE = re.compile(r"^\s*\[Assistant\]:\s*")
TEXT_FIELDS = ("transcript", "rewritten")


def strip_role_prefix(text: str) -> str:
    return ROLE_PREFIX_RE.sub("", text)


def normalise_file(path: Path) -> tuple[int, int]:
    """Rewrite the file in place. Returns (rows, rows changed)."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    changed = 0
    for row in rows:
        for field in TEXT_FIELDS:
            if field in row:
                new = strip_role_prefix(row[field])
                if new != row[field]:
                    row[field] = new
                    changed += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows), changed


def _self_check() -> None:
    assert strip_role_prefix("[Assistant]: hello") == "hello"
    assert strip_role_prefix("[Assistant]:   hello") == "hello"
    assert strip_role_prefix("hello") == "hello"  # already clean
    assert strip_role_prefix(strip_role_prefix("[Assistant]: x")) == "x"  # idempotent
    assert strip_role_prefix("text [Assistant]: mid") == "text [Assistant]: mid"  # only leading
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        for arg in sys.argv[1:]:
            n, c = normalise_file(Path(arg))
            print(f"{arg}: {n} rows, {c} stripped")
