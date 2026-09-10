#!/usr/bin/env bash
# Gate for data/ (see data/CONTRACT.md). Run before committing data/ and after pulling it.
# Usage: bash scripts/audit/logs/check.sh [data/<bench> ...]    # exit 1 = fix first
# With no arguments it sweeps all of data/, which takes many minutes; after a collection
# pass the bench you touched (e.g. data/arc_challenge) and it finishes in seconds.
set -u
cd "$(dirname "$0")/../../.."
roots=("$@")
[ ${#roots[@]} -eq 0 ] && roots=(data)
fail=0
uv run python scripts/audit/logs/validate_data.py "${roots[@]}" || fail=1
leftover=$(find "${roots[@]}" -path data/legacy -prune -o -type d -name reruns -print)
[ -n "$leftover" ] && { printf 'unspliced reruns:\n%s\n' "$leftover"; fail=1; }
uv run python scripts/audit/logs/stamp_unrecorded_edits.py "${roots[@]}" 2>&1 | tail -1  # dry run, informational
exit $fail
