"""Turn two eval logs into a dataset of pairwise judging prompts.

One sample per question that both models answered correctly, showing their two
transcripts. With ``include_reverse`` each pair also appears in the opposite order, which
is what makes position bias visible in the results.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log, read_eval_log_samples

from judge.util import extract_score, extract_transcript

# Characters kept per transcript; must match the ExperimentConfig default.
MAX_LENGTH = 262144

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


@dataclass
class Transcript:
    """All the judge sees of one model's answer to one question."""

    id: str
    question: Any
    text: str
    options: tuple[str, ...] = ()
    epoch: int = 0


def read_transcripts(
    path: str | Path,
    require_correct: bool = True,
    include_reasoning: bool = True,
) -> dict[str, list[Transcript]]:
    """Read one eval log into transcripts per question id, ordered by epoch.

    Samples are streamed and reduced to text as they arrive: single logs reach tens of MB
    compressed, and judging runs many of them in parallel processes.
    """
    by_id: dict[str, list[Transcript]] = {}
    for sample in read_eval_log_samples(str(path), all_samples_required=False):
        if require_correct and not extract_score(sample):
            continue
        record = Transcript(
            id=str(sample.id),
            question=sample.input,
            text=extract_transcript(sample, include_reasoning=include_reasoning),
            options=tuple(getattr(sample, "choices", None) or ()),
            epoch=getattr(sample, "epoch", 0) or 0,
        )
        by_id.setdefault(record.id, []).append(record)

    for group in by_id.values():
        group.sort(key=lambda transcript: transcript.epoch)
    return by_id


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


def truncate_pair(a: str, b: str, frac: float) -> tuple[str, str]:
    """Cut both transcripts to frac of the shorter one's length, answers removed."""
    a, b = strip_answer(a), strip_answer(b)
    cut = max(1, int(frac * min(len(a), len(b))))
    return a[:cut], b[:cut]


def responses_to_sample(
    first: Transcript,
    second: Transcript,
    target: str = '1',
    format_str: str = USER_PROMPT,
    max_len: int = MAX_LENGTH,
    truncate_frac: float | None = None,
) -> Sample:
    """Render two transcripts of the same question as one judging sample.

    Options are shuffled per model, so each transcript is shown with its own option order.

    Args:
        first: shown as Transcript 1.
        second: shown as Transcript 2.
        target: which position holds the more capable model ('1' or '2').
        format_str: prompt template to fill.
        max_len: maximum characters kept per transcript.
        truncate_frac: if set, cut both transcripts to this fraction of the shorter one's
            length and drop their answers (length-control condition).
    """
    assert first.id == second.id, "Sample IDs must match for comparison."
    assert first.question == second.question, "Sample inputs must match for comparison."

    text_1, text_2 = first.text[:max_len], second.text[:max_len]
    if truncate_frac is not None:
        text_1, text_2 = truncate_pair(text_1, text_2, truncate_frac)

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
        # Length is itself a signal, and a cut transcript loses its answer: record both so
        # the analysis can tell what the judge actually saw. ``chars_*`` are the *shown*
        # lengths -- under ``truncate_frac`` the raw lengths are exactly what the condition
        # removes, so correlating a verdict against them would measure the wrong thing.
        metadata={
            "chars_1": len(text_1),
            "chars_2": len(text_2),
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
    truncate_frac: float | None = None,
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
        truncate_frac: length-control fraction, see ``responses_to_sample``.
    """
    self_pair = Path(major_path) == Path(minor_path)
    major = read_transcripts(major_path, require_both_correct, include_reasoning)
    minor = major if self_pair else read_transcripts(minor_path, require_both_correct, include_reasoning)

    samples = []
    for sample_id in sorted(major.keys() & minor.keys()):
        if self_pair:
            # Self-pair control: two distinct correct epochs of the same model.
            if len(major[sample_id]) < 2:
                continue
            first, second = major[sample_id][0], major[sample_id][1]
        else:
            first, second = major[sample_id][0], minor[sample_id][0]

        pair_kwargs = dict(format_str=format_str, max_len=max_len, truncate_frac=truncate_frac)
        samples.append(responses_to_sample(first, second, target='1', **pair_kwargs))
        if include_reverse:
            samples.append(responses_to_sample(second, first, target='2', **pair_kwargs))

    header_major = read_eval_log(str(major_path), header_only=True)
    header_minor = read_eval_log(str(minor_path), header_only=True)
    return MemoryDataset(
        samples=samples,
        name=(
            f"Comparison of {header_major.eval.model} and {header_minor.eval.model} "
            f"over the {header_major.eval.dataset.name} benchmark"
        ),
    )
