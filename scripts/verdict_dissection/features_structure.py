"""Structural transcript features: agentic trajectory shape and answer presentation.

    uv run python scripts/verdict_dissection/features_structure.py

Reads the canonical transcript store and the surface feature table (for the row set and keys) and
writes <features_root>/raw/structure.parquet, one row per canonical-correct transcript, keyed like
the other feature tables. Agentic features come from transcript_parts_json (tool calls, tool
outputs, prose); they are zero for single-message transcripts. Presentation features come from the
assistant prose: bare answers, display equations, option references and elimination language,
whether the final answer is restated at the end.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]

TOOL_ERROR = re.compile(r"Traceback|Error|error:|not found|No such file|Permission denied|command not found|"
                        r"syntax error|Exception|failed|Failed|cannot |Cannot |non-zero exit|exit code [1-9]", re.I)
ANSWER_LINE = re.compile(r"^\s*ANSWER:.*$", re.M)
DISPLAY_MATH = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$|\\begin\{(?:align|equation)\*?\}", re.S)
INLINE_MATH = re.compile(r"\\\(.*?\\\)|(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)")
OPTION_REFERENCE = re.compile(r"\b(?:option|choice|answer)s?\s*\(?[A-D]\b|\(\s*[A-D]\s*\)|\b[A-D]\)", re.I)
ELIMINATION = re.compile(r"\b(?:incorrect|eliminat\w*|rule[sd]? out|wrong|cannot be|can't be|not (?:the )?(?:correct|right)|"
                         r"inconsistent|distractor\w*|does not (?:match|satisfy|hold)|doesn't (?:match|satisfy|hold))\b", re.I)
FINAL_RESTATED = re.compile(r"(?:final answer|the answer is|therefore|thus|hence|so the answer|answer:)", re.I)
TEST_COMMAND = re.compile(r"pytest|unittest|\btests?\b|tox\b|npm test|make test|cargo test|go test", re.I)
BARE_ANSWER_CHARS = 40


def structure_of(parts_json):
    """Agentic-shape counts from the transcript's parts."""
    parts = json.loads(parts_json)
    prose = tool_call = tool_output = 0
    n_assistant = n_calls = n_outputs = n_errors = n_empty = n_submit = n_tests = 0
    payloads, tools = Counter(), set()
    for part in parts:
        if part["role"] == "assistant":
            n_assistant += 1
        for sub in part["subparts"]:
            kind, text = sub["kind"], sub["content"] or ""
            if kind == "content":
                prose += len(text)
            elif kind == "tool_call":
                n_calls += 1
                tool_call += len(text)
                tools.add(sub["function"] or "")
                if sub["function"] == "submit":
                    n_submit += 1
                if TEST_COMMAND.search(json.dumps(sub["raw_payload"])):
                    n_tests += 1
                payloads[json.dumps(sub["raw_payload"], sort_keys=True)] += 1
            elif kind == "tool_output":
                n_outputs += 1
                tool_output += len(text)
                body = sub["raw_payload"] if isinstance(sub["raw_payload"], str) else text
                if not (body or "").strip():
                    n_empty += 1
                elif TOOL_ERROR.search(body):
                    n_errors += 1
    total = prose + tool_call + tool_output
    return {"assistant_turns": n_assistant, "tool_calls": n_calls, "tool_errors": n_errors,
            "tool_outputs_empty": n_empty, "submit_calls": n_submit, "test_commands": n_tests,
            "repeated_tool_calls": sum(c - 1 for c in payloads.values() if c > 1),
            "distinct_tools": len(tools), "prose_chars": prose, "tool_call_chars": tool_call,
            "tool_output_chars": tool_output, "prose_share": prose / total if total else 1.0}


def presentation_of(parts_json):
    """Answer-presentation counts from the assistant prose (tool calls and outputs excluded)."""
    parts = json.loads(parts_json)
    prose = "\n".join(sub["content"] or "" for part in parts for sub in part["subparts"] if sub["kind"] == "content")
    body = ANSWER_LINE.sub("", prose)
    tail = prose[-300:]
    return {"bare_answer": int(len(body.strip()) < BARE_ANSWER_CHARS),
            "display_equations": len(DISPLAY_MATH.findall(body)),
            "inline_math": len(INLINE_MATH.findall(body)),
            "option_references": len(OPTION_REFERENCE.findall(body)),
            "elimination_terms": len(ELIMINATION.findall(body)),
            "final_answer_restated": int(bool(FINAL_RESTATED.search(tail))),
            "bold_spans": body.count("**") // 2,
            "headers": len(re.findall(r"^#{1,4}\s", body, re.M))}


def main():
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    rows = pd.read_parquet(features_root / "raw/surface-textual.parquet", columns=KEYS)
    store = pd.read_parquet(ROOT / config["freeze"] / "transcript_store.parquet", columns=KEYS + ["transcript_parts_json"])
    table = rows.merge(store, on=KEYS, how="left", validate="one_to_one")
    if table.transcript_parts_json.isna().any():
        raise ValueError(f"{int(table.transcript_parts_json.isna().sum())} transcripts missing from the store")
    features = pd.DataFrame([structure_of(p) | presentation_of(p) for p in table.transcript_parts_json])
    out = pd.concat([table[KEYS], features], axis=1)
    out.to_parquet(features_root / "raw/structure.parquet", index=False)
    print(out.drop(columns=KEYS).groupby(out.bench).mean().round(2).T.to_string())


if __name__ == "__main__":
    main()
