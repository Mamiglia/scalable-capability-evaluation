"""Deterministic transcript features of the reasoning channel alone, for the channel ablation of the
Deterministic features protocol, and of transcripts with one third removed or kept alone, for the thirds ablation.
Run through scripts/analysis/features.py.

logs/features/ holds two views per transcript, "visible" (messages + reasoning) and "hidden" (messages
only); the other channel views derive from them except the reasoning alone, extracted here: every
assistant message reduced to its reasoning blocks, no tool calls, then protocols.transcript_features
.features.extract_features. Rows: the correct first attempts of the references' and targets' train
items of the given semesters and domains; columns path (the data/ log), item, correct, FEATURES.

The thirds view cuts as the pairwise and Embedding thirds runs do (judge.dataset.cut, thirds-<kept>): the rendered
transcript (evals.sample.extract_transcript) is split into three equal character spans and the characters of the
removed span are dropped from every text and reasoning block they fall in, a tool call dropped when its rendered
call starts there; the features are then extracted from the cut messages. Rows gain a condition column: full
(no cut, the glm_features "visible" view), each third removed (thirds-23, thirds-13, thirds-12) and each kept
alone (thirds-1, thirds-2, thirds-3).
"""

import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from inspect_ai.log import EvalSample

from analysis import loading
from evals.sample import PLACEHOLDER_BLOCKS, extract_score, extract_transcript, format_reasoning, format_single_message, format_text
from protocols.transcript_features.features import FEATURES, extract_features
from protocols.transcript_features.reader import MEMBER
from reference import tables

THIRDS = ("full", "thirds-23", "thirds-13", "thirds-12", "thirds-1", "thirds-2", "thirds-3")  # removed, then kept alone
THOUGHT_START = "<THOUGHT_START>\n "  # evals.sample.format_reasoning's prefix before the reasoning text


def reasoning_only(sample: EvalSample) -> list:
    """The sample's messages with every assistant message cut to its reasoning blocks and no tool calls."""
    out = []
    for message in sample.messages:
        if message.role != "assistant":
            out.append(message)
            continue
        blocks = [] if isinstance(message.content, str) else [b for b in message.content if b.type == "reasoning"]
        out.append(message.model_copy(update={"content": blocks, "tool_calls": None}))
    return out


def thirds_cut(sample: EvalSample, condition: str) -> list:
    """The sample's messages with the removed third's characters dropped, located on the rendered transcript
    (extract_transcript: system prompt and first user turn left out, "[Role]: " before each turn, turns joined
    by a blank line, blocks by a newline, tool calls after the content)."""
    if condition == "full" or not any(m.role == "assistant" for m in sample.messages):
        return sample.messages  # no assistant turn: nothing to cut, every feature is NaN
    n = len(extract_transcript(sample))
    kept = []
    for i in (i for i in range(3) if str(i + 1) in condition[7:]):  # adjacent kept thirds merge into one span
        a, b = i * n // 3, (i + 1) * n // 3
        kept[-1:] = [(kept[-1][0], b)] if kept and kept[-1][1] == a else [*kept[-1:], (a, b)]

    def keep(text: str, start: int) -> str:
        return "\n".join(p for a, b in kept if (p := text[max(a - start, 0):max(b - start, 0)]))

    turns, seen_user, out, pos = [], False, [], 0
    for message in sample.messages:
        if message.role == "system" or (message.role == "user" and not seen_user):
            seen_user = seen_user or message.role == "user"
            out.append(message)
            continue
        pos += (2 if turns else 0) + len(f"[{message.role.capitalize()}]: ")
        turns.append(message)
        rendered = len(format_single_message(message))
        if message.role != "assistant":
            out.append(message)
            pos += rendered
            continue
        at, blocks, pieces = pos, [], []
        content = message.content
        for block in [content] if isinstance(content, str) else content:
            if isinstance(block, str) or block.type == "text":
                text = format_text(block if isinstance(block, str) else block.text or "")
                cut = keep(text, at)
                blocks.append(cut if isinstance(block, str) else block.model_copy(update={"text": cut}))
            elif block.type == "reasoning":
                if not (text := format_reasoning(block)):
                    blocks.append(block)
                    continue
                cut = keep(block.reasoning, at + len(THOUGHT_START))
                blocks.append(block.model_copy(update={"reasoning": cut}))
            else:
                text = PLACEHOLDER_BLOCKS[block.type]
                blocks.append(block)
            pieces.append(text)
            at += len(text) + 1
        body = len("\n".join(pieces))
        at = pos + body + (1 if body else 0)
        calls = []
        for call in message.tool_calls or []:
            if any(a <= at < b for a, b in kept):
                calls.append(call)
            at += len(f"[Tool Call: {call.function}({call.arguments})]") + 1
        content = blocks[0] if isinstance(content, str) else blocks
        out.append(message.model_copy(update={"content": content, "tool_calls": calls or None}))
        pos += rendered
    if pos != n:
        raise ValueError(f"rendered length {n}, walked {pos}: the cut is misaligned with evals.sample")
    return out


def extract_thirds(job: tuple[str, set[str]]) -> list[dict]:
    """extract for every THIRDS condition: one row per first attempt on `items` and condition."""
    path, items = job
    rows = []
    with zipfile.ZipFile(tables.DATA / path) as archive:
        for name in archive.namelist():
            if not (m := MEMBER.match(name)) or int(m[2]) != 1 or m[1] not in items:
                continue
            sample = EvalSample.model_validate_json(archive.read(name))
            if not sample.messages:
                continue
            correct = extract_score(sample)
            for condition in THIRDS:
                rows.append({"path": path, "item": m[1], "correct": correct, "condition": condition,
                             **extract_features(thirds_cut(sample, condition), include_reasoning=True)})
    return rows


def extract(job: tuple[str, set[str]]) -> list[dict]:
    """Feature rows of the first attempts on `items` in the log at `path` (relative to data/)."""
    path, items = job
    rows = []
    with zipfile.ZipFile(tables.DATA / path) as archive:
        for name in archive.namelist():
            if not (m := MEMBER.match(name)) or int(m[2]) != 1 or m[1] not in items:
                continue
            sample = EvalSample.model_validate_json(archive.read(name))
            if not sample.messages:
                continue
            rows.append({"path": path, "item": m[1], "correct": extract_score(sample),
                         **extract_features(reasoning_only(sample), include_reasoning=True)})
    return rows


def jobs(domains: list[str], semesters: list[str]) -> list[tuple[str, set[str]]]:
    """(log path, train item ids) per transcript log of every reference and target of the given tasks."""
    wanted: dict[str, set[str]] = {}
    for domain in domains:
        for semester in semesters:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            for cohort in (task.references, task.test):
                for bench, ids in cohort.items.groupby("bench").item:
                    for model in cohort.models.index:
                        if isinstance(path := cohort.models.loc[model, f"path_{bench}"], str):
                            wanted.setdefault(path, set()).update(map(str, ids))
    return sorted(wanted.items())


def feature_rows(view: str, domains: list[str], semesters: list[str], workers: int = 6) -> pd.DataFrame:
    """The features of every correct first attempt of the given tasks' references and targets in one view.

    view: "reasoning_only" (every assistant message cut to its reasoning) or "thirds" (one row per THIRDS condition).
    Returns columns path, item, correct, [condition,] FEATURES.
    """
    assert view in ("reasoning_only", "thirds"), view
    todo, rows = jobs(domains, semesters), []
    with ProcessPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(extract_thirds if view == "thirds" else extract, todo), start=1):
            rows.extend(part)
            if i % 25 == 0:
                print(f"{view}: {i}/{len(todo)} logs, {len(rows)} rows", flush=True)
    return pd.DataFrame(rows, columns=["path", "item", "correct", *(["condition"] if view == "thirds" else []), *FEATURES])
