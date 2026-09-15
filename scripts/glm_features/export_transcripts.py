"""Export the transcripts the structure tier parses: correct first attempts on every task's items.

    uv run python scripts/glm_features/export_transcripts.py <bench> [--out <dir>]

For every domain and semester, every train/test model's correct epoch-1 samples on the
task's items (as recorded in the glm_features CSVs) are read from the bench's logs and
written to <out>/<bench>.parquet: log (path under data/), item, epoch, parts. `parts` is a
JSON list in transcript order of {"kind": text|reasoning|tool_call|tool_output|feedback,
"text": ..., "function": ..., "call_id": ...}; the question and system prompt are omitted.
The RST labeller (rst_labels.py) consumes this file; it needs no logs.
"""

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from inspect_ai.log import EvalSample

from capability.evaluate import semester_names
from capability.task import Task
from configs.domain import DomainConfig
from evals.sample import format_reasoning, format_text

DOMAINS = ["science", "math", "coding", "cyber"]
OUTPUT_CHARS = 4000  # tool outputs are nodes, not parsed prose; keep enough to classify them


def selected(bench: str) -> dict[str, set[str]]:
    """{log path: item ids} the tasks need from this bench."""
    need: dict[str, set[str]] = {}
    for domain in DOMAINS:
        config = DomainConfig.from_cli(f"configs/domain/{domain}.yml")
        for semester in semester_names(config):
            task = Task.from_semester(semester, config)
            for cohort in (task.train, task.test):
                items = set(cohort.items[cohort.items.bench == bench].item.astype(str))
                if not items or f"path_{bench}" not in cohort.models:
                    continue
                for path in cohort.models[f"path_{bench}"].dropna():
                    rows = pd.read_csv(Path("glm_features") / Path(path).with_suffix(".csv"), dtype={"item": str})
                    ok = rows[(rows.epoch == 1) & rows.correct & (rows.reasoning == "visible") & rows.item.isin(items)]
                    need.setdefault(path, set()).update(ok.item)
    return need


def parts(sample: EvalSample) -> list[dict]:
    messages = sample.messages
    if not messages:
        choices = getattr(sample.output, "choices", None) or []
        messages = [choices[0].message] if choices else []
    out, seen_user = [], False
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            if seen_user:  # later user turns are feedback; the first is the question
                out.append({"kind": "feedback", "text": _text(message.content)})
            seen_user = True
        elif message.role == "tool":
            out.append({"kind": "tool_output", "text": _text(message.content)[:OUTPUT_CHARS],
                        "call_id": message.tool_call_id, "function": getattr(message, "function", None)})
        elif message.role == "assistant":
            if isinstance(message.content, str):
                out.append({"kind": "text", "text": format_text(message.content)})
            else:
                for block in message.content:
                    if block.type == "text":
                        out.append({"kind": "text", "text": format_text(block.text)})
                    elif block.type == "reasoning" and format_reasoning(block):
                        out.append({"kind": "reasoning", "text": block.reasoning})
            for call in message.tool_calls or []:
                out.append({"kind": "tool_call", "text": json.dumps(call.arguments, ensure_ascii=False),
                            "function": call.function, "call_id": call.id})
    return [p for p in out if p["text"] and p["text"].strip()]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", "") == "text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench")
    parser.add_argument("--out", type=Path, default=Path("glm_features/rst/transcripts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, items in sorted(selected(args.bench).items()):
        with zipfile.ZipFile(Path("data") / path) as archive:
            for item in sorted(items):
                sample = EvalSample.model_validate_json(archive.read(f"samples/{item}_epoch_1.json"))
                rows.append({"log": path, "item": item, "epoch": 1, "parts": json.dumps(parts(sample), ensure_ascii=False)})
    table = pd.DataFrame(rows, columns=["log", "item", "epoch", "parts"])
    table.to_parquet(args.out / f"{args.bench}.parquet", index=False)
    print(f"{args.bench}: {len(table)} transcripts from {table.log.nunique()} logs, {table.parts.str.len().sum() / 1e6:.1f}M chars")


if __name__ == "__main__":
    main()
