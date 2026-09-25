"""Read one eval log into Transcripts: what a model wrote for each item, per epoch.

read_transcripts(path, items=...) is cached per process and reads only the samples asked for
(an HLE log is ~60 MB). read_header reads a log's header once.
"""

import functools
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, read_eval_log, read_eval_log_samples

from evals.sample import extract_score, extract_transcript

REASONING_MARK = "<THOUGHT_START>"  # how extract_transcript renders a reasoning block


@dataclass
class Transcript:
    """All the judge sees of one model's answer to one question."""

    id: str
    question: Any
    text: str
    options: tuple[str, ...] = ()
    epoch: int = 0
    tokens: int = 0  # output tokens billed for the sample (reasoning included)


def question_text(sample_input: Any) -> str:
    """The question as text. Chat inputs (e.g. HLE) are a list of messages whose ids differ
    per run, so compare and show their user text, not the message objects."""
    if isinstance(sample_input, str):
        return sample_input
    return "\n\n".join(m.text for m in sample_input if getattr(m, "role", "user") == "user")


def read_transcripts(
    path: str | Path,
    require_correct: bool = True,
    include_reasoning: bool = True,
    items: set[str] | None = None,
    channels: str = "",
) -> dict[str, list[Transcript]]:
    """Read one eval log into transcripts per question id, ordered by epoch.

    ``items`` restricts to those question ids, read by random access instead of streaming
    the log (an HLE log is ~60 MB; a quartile of it a few dozen samples), so pass ids the
    log is known to contain. Cached per log: however many pairs a model appears in, each
    sample is read once per process.
    """
    key = (str(path), require_correct, include_reasoning, channels)
    if items is None:
        if key not in _STREAMED:
            _STREAMED[key] = _group(read_eval_log_samples(str(path), all_samples_required=False),
                                    require_correct, include_reasoning, channels)
        return _STREAMED[key]
    # Random access, cached incrementally: an id maps to [] when read but not correct.
    store = _RANDOM.setdefault(key, {})
    missing = sorted(set(map(str, items)) - store.keys())
    if missing:
        read = _group(_read_samples(str(path), missing), require_correct, include_reasoning, channels)
        store.update({item: read.get(item, []) for item in missing})
    return {item: store[item] for item in map(str, items) if item in store and store[item]}


_STREAMED: dict[tuple, dict[str, list[Transcript]]] = {}
_RANDOM: dict[tuple, dict[str, list[Transcript]]] = {}


def _read_samples(path: str, ids: list[str], epoch: int = 1):
    """Samples of an .eval log by id, as zip member reads (``samples/<id>_epoch_<n>.json``).

    inspect's ``read_eval_log_sample`` re-reads the whole archive on every call — 60 MB per
    sample on HLE; this reads the central directory once and then only the members asked for.
    """
    with zipfile.ZipFile(path) as archive:
        for item in ids:
            yield EvalSample.model_validate_json(archive.read(f"samples/{item}_epoch_{epoch}.json"))


def _group(samples, require_correct: bool, include_reasoning: bool, channels: str = "") -> dict[str, list[Transcript]]:
    by_id: dict[str, list[Transcript]] = {}
    for sample in samples:
        if require_correct and not extract_score(sample):
            continue
        record = Transcript(
            id=str(sample.id),
            question=question_text(sample.input),
            text=extract_transcript(sample, include_reasoning=include_reasoning, channels=channels),
            options=tuple(getattr(sample, "choices", None) or ()),
            epoch=getattr(sample, "epoch", 0) or 0,
            tokens=sum(u.output_tokens or 0 for u in (getattr(sample, "model_usage", None) or {}).values()),
        )
        by_id.setdefault(record.id, []).append(record)
    for group in by_id.values():
        group.sort(key=lambda transcript: transcript.epoch)
    return by_id


@functools.cache
def read_header(path: str | Path):
    """Header-only read of a log, once per process (judging touches each log many times)."""
    return read_eval_log(str(path), header_only=True)
