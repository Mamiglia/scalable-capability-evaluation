#!/usr/bin/env bash
# Gate for data/ (see data/CONTRACT.md). Run before committing data/ and after pulling it.
# Usage: bash scripts/audit/logs/check.sh [data/<bench> ...]    # exit 1 = fix first
set -u
cd "$(dirname "$0")/../../.."
fail=0
uv run python scripts/audit/logs/validate_data.py "$@" || fail=1
leftover=$(find data -path data/legacy -prune -o -type d -name reruns -print)
[ -n "$leftover" ] && { printf 'unspliced reruns:\n%s\n' "$leftover"; fail=1; }
uv run python scripts/audit/logs/stamp_unrecorded_edits.py 2>&1 | tail -1  # dry run, informational
exit $fail
