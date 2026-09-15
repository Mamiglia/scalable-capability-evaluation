"""Turn two eval logs into a dataset of pairwise judging prompts.

One sample per question that both models answered correctly, showing their two
transcripts. With ``include_reverse`` each pair also appears in the opposite order, which
is what makes position bias visible in the results.
"""

import re
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

from configs.logs import logger
from evals.transcripts import Transcript, read_header, read_transcripts

# Characters kept per transcript; must match the JudgeConfig default.
MAX_LENGTH = 262144
# A pair is judged on an item only if both models wrote at least this much (role prefixes excluded):
# an answer-only "ANSWER: B" gives the judge nothing to compare. Always on, for every judging pair.
MIN_RESPONSE_CHARS = 20
ROLE_PREFIX = re.compile(r"^\[[^\]\n]+\]: ", re.M)  # "[Assistant]: ", "[User]: ", "[Tool Output (f)]: " (evals.sample)


def cut(text: str, max_len: int, truncate: str = "head") -> str:
    """The max_len characters a judge sees of a longer transcript: its prefix (head) or suffix (tail)."""
    if truncate not in ("head", "tail"):
        raise ValueError(f"truncate must be head or tail, got {truncate!r}")
    return text[:max_len] if truncate == "head" else text[-max_len:]


def response_chars(text: str) -> int:
    """Characters the model itself wrote in a shown transcript: the text minus the renderer's role prefixes."""
    return len(ROLE_PREFIX.sub("", text))

USER_PROMPT = """Input question:
---

{question}

---

{options_1}Transcript 1:
---

{transcript_1}

---

{options_2}Transcript 2:
---

{transcript_2}

---

Which of the two transcripts is from the more capable model? Answer in the instructed format.
"""

OPTIONS_BLOCK = """Multiple choice options as shown to Model {n}:
{options}

"""

DIRECT_USER_PROMPT = """Input question:
---

{question}

---

{options}Transcript 1:
---

{transcript}

---

How capable is the model that produced this transcript, on a scale from 1 to 10? Answer in the instructed format.
"""


def format_options(options: tuple[str, ...], n: int) -> str:
    """Render the options in the order that model saw them. Empty if it had none."""
    if not options:
        return ""
    rendered = "\n".join(
        f"{chr(ord('A') + i)}) {option.strip()}" for i, option in enumerate(options)
    )
    return OPTIONS_BLOCK.format(n=n, options=rendered)


def strip_answer(text: str) -> str:
    """Drop the trailing ANSWER line."""
    idx = text.rfind("ANSWER:")
    return text[:idx].rstrip() if idx != -1 else text


def has_reasoning(text: str, min_chars: int = 100) -> bool:
    """False for answer-only transcripts: nothing left after the ANSWER line."""
    return len(strip_answer(text)) >= min_chars


def responses_to_sample(
    first: Transcript,
    second: Transcript,
    target: str = '1',
    format_str: str = USER_PROMPT,
    max_len: int = MAX_LENGTH,
    truncate: str = "head",
) -> Sample:
    """Render two transcripts of the same question as one judging sample.

    Options are shuffled per model, so each transcript is shown with its own option order.

    Args:
        first: shown as Transcript 1.
        second: shown as Transcript 2.
        target: which position holds the more capable model ('1' or '2').
        format_str: prompt template to fill.
        max_len: maximum characters kept per transcript.
        truncate: which end of a longer transcript is kept, head or tail.
    """
    assert first.id == second.id, "Sample IDs must match for comparison."
    assert first.question == second.question, "Sample inputs must match for comparison."

    text_1, text_2 = cut(first.text, max_len, truncate), cut(second.text, max_len, truncate)

    return Sample(
        id=f"{first.id}_{target}",
        input=format_str.format(
            question=first.question,
            options_1=format_options(first.options, 1),
            options_2=format_options(second.options, 2),
            transcript_1=text_1,
            transcript_2=text_2,
        ),
        target=target,
        # Length is itself a signal: record what the judge actually saw (``chars_*`` are the
        # shown lengths) and whether max_len cut a transcript.
        metadata={
            "chars_1": len(text_1),
            "chars_2": len(text_2),
            "response_chars_1": response_chars(text_1),
            "response_chars_2": response_chars(text_2),
            "truncated_1": len(first.text) > max_len,
            "truncated_2": len(second.text) > max_len,
        },
    )


def build_dataset(
    major_path: Path,
    minor_path: Path,
    format_str: str = USER_PROMPT,
    max_len: int = MAX_LENGTH,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    include_reasoning: bool = True,
    items: set[str] | None = None,
    truncate: str = "head",
) -> MemoryDataset:
    """Build the judging dataset comparing a major and a minor model's eval log.

    Args:
        major_path: eval log of the model that is expected to be more capable.
        minor_path: eval log of the other model; the same path runs the self-pair control,
            which compares two epochs of one model.
        format_str: prompt template to fill.
        max_len: maximum characters kept per transcript.
        require_both_correct: keep only questions both models answered correctly.
        include_reverse: also add each pair with the transcripts swapped.
        include_reasoning: include CoT reasoning blocks in the transcripts.
        items: question ids to keep (e.g. one difficulty quartile); None = all.
    Questions where either model wrote fewer than MIN_RESPONSE_CHARS are skipped.
    """
    self_pair = Path(major_path) == Path(minor_path)
    major = read_transcripts(major_path, require_both_correct, include_reasoning, items)
    minor = major if self_pair else read_transcripts(minor_path, require_both_correct, include_reasoning, items)

    ids = major.keys() & minor.keys()
    # With include_reasoning a reasoning model's trace is shown and a non-reasoning model's
    # transcript is shown as is: mixed pairs are judged like any other.
    samples = []
    for sample_id in sorted(ids):
        if self_pair:
            # Self-pair control: two distinct correct epochs of the same model.
            if len(major[sample_id]) < 2:
                continue
            first, second = major[sample_id][0], major[sample_id][1]
        else:
            first, second = major[sample_id][0], minor[sample_id][0]
        if min(response_chars(cut(first.text, max_len, truncate)), response_chars(cut(second.text, max_len, truncate))) < MIN_RESPONSE_CHARS:
            continue  # e.g. an answer-only transcript: nothing for the judge to compare

        pair_kwargs = dict(format_str=format_str, max_len=max_len, truncate=truncate)
        samples.append(responses_to_sample(first, second, target='1', **pair_kwargs))
        if include_reverse:
            samples.append(responses_to_sample(second, first, target='2', **pair_kwargs))

    header_major = read_header(major_path)
    header_minor = read_header(minor_path)
    return MemoryDataset(
        samples=samples,
        name=(
            f"Comparison of {header_major.eval.model} and {header_minor.eval.model} "
            f"over the {header_major.eval.dataset.name} benchmark"
        ),
    )


def build_direct_dataset(
    path: str | Path,
    items: set[str],
    max_len: int = MAX_LENGTH,
    include_reasoning: bool = True,
    truncate: str = "head",
) -> MemoryDataset:
    """One sample per correct first-attempt transcript on the requested train items.

    Explicit item reads select the first epoch, as in Cohort.transcripts. Every correct
    transcript is eligible, including short answers; there is no second model to match.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    transcripts = read_transcripts(path, require_correct=True, include_reasoning=include_reasoning, items=items)
    samples = []
    for item in sorted(transcripts):
        transcript = transcripts[item][0]
        text = cut(transcript.text, max_len, truncate)
        samples.append(Sample(
            id=item,
            input=DIRECT_USER_PROMPT.format(question=transcript.question,
                                           options=format_options(transcript.options, 1), transcript=text),
            metadata={"chars": len(text), "response_chars": response_chars(text),
                      "truncated": len(transcript.text) > max_len},
        ))
    return MemoryDataset(samples=samples, name=f"Direct judging of {Path(path).stem}")
