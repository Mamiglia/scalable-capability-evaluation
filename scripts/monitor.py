#!/usr/bin/env python3
"""
Filtered, colorized view of `inspect ctl task list`.

Run once per refresh via `watch`, e.g.:
    watch -n 2 -c ./inspect_watch.py

Parses the table as fixed-width columns (using the dashed separator line
to find column boundaries), so fields containing spaces (like "3 running")
don't get mis-split the way whitespace-splitting would.
"""
import subprocess
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"

# Columns to keep, in order. Must match the header names printed by
# `inspect ctl task list` exactly (case-insensitive).
WANTED = ["task", "model", "samples", "errors", "http_retries", "started"]


def get_column_spans(sep_line):
    """Given a line like '------  ----------  ---', return [(start, end), ...]."""
    spans = []
    i = 0
    n = len(sep_line)
    while i < n:
        if sep_line[i] == "-":
            start = i
            while i < n and sep_line[i] == "-":
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def shorten(col, value):
    v = value.strip()
    if col in ("task", "model"):
        v = v.rsplit("/", 1)[-1]
    if col == "model" and len(v) > 15:
        v = v[:14] + "\u2026"
    return v


def colorize(col, value):
    v = value.strip()
    if col == "errors":
        return (GREEN if v == "0" else RED) + v + RESET
    if col == "http_retries":
        if v == "0":
            return GREEN + v + RESET
        try:
            n = int(v)
        except ValueError:
            return v
        return (YELLOW if n < 50 else RED) + v + RESET
    if col == "samples":
        return (CYAN + v + RESET) if "running" in v else v
    return v


def main():
    try:
        proc = subprocess.run(
            ["inspect", "ctl", "task", "list"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        sys.exit("inspect command not found on PATH")
    except subprocess.CalledProcessError as e:
        sys.stdout.write(e.stdout or "")
        sys.stderr.write(e.stderr or "")
        sys.exit(e.returncode)

    lines = proc.stdout.splitlines()

    # Find header + separator line (separator is the first line made only
    # of '-' and spaces).
    sep_idx = None
    for i, line in enumerate(lines):
        if line and set(line.strip()) <= {"-", " "} and "-" in line:
            sep_idx = i
            break

    if sep_idx is None or sep_idx == 0:
        # Fallback: nothing recognizable, just print raw output.
        print(proc.stdout)
        return

    header_line = lines[sep_idx - 1]
    spans = get_column_spans(lines[sep_idx])

    all_cols = [header_line[s:e].strip() for s, e in spans]
    col_index = {name.lower(): idx for idx, name in enumerate(all_cols)}

    missing = [c for c in WANTED if c not in col_index]
    if missing:
        print(f"{YELLOW}Warning: columns not found in output: {missing}{RESET}")
        print(f"{DIM}Available columns: {all_cols}{RESET}\n")
    keep = [c for c in WANTED if c in col_index]

    # Data rows: everything after separator, until a blank line.
    data_rows = []
    end_idx = sep_idx + 1
    for j in range(sep_idx + 1, len(lines)):
        if lines[j].strip() == "":
            end_idx = j
            break
        end_idx = j + 1
        row = []
        for c in keep:
            idx = col_index[c]
            s, e = spans[idx]
            # last column may run past its dash width; take to end of line
            if idx == len(spans) - 1:
                val = lines[j][s:]
            else:
                val = lines[j][s:e]
            row.append(shorten(c, val))
        data_rows.append(row)

    footer_lines = [l for l in lines[end_idx:] if l.strip() != ""]

    # Compute display widths from the *filtered* columns only.
    widths = [len(c) for c in keep]
    for row in data_rows:
        for k, val in enumerate(row):
            widths[k] = max(widths[k], len(val))

    header_out = "  ".join(
        BOLD + keep[k].upper().ljust(widths[k]) + RESET for k in range(len(keep))
    )
    sep_out = "  ".join("-" * widths[k] for k in range(len(keep)))
    print(header_out)
    print(sep_out)
    for row in data_rows:
        cells = []
        for k, val in enumerate(row):
            colored = colorize(keep[k], val)
            pad = widths[k] - len(val)
            cells.append(colored + " " * pad)
        print("  ".join(cells))

    if footer_lines:
        print()
        for l in footer_lines:
            if "error" in l.lower() and not l.strip().startswith("0 "):
                print(RED + l + RESET)
            else:
                print(DIM + l + RESET)


if __name__ == "__main__":
    main()
