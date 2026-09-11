"""Turn two eval logs into a dataset of pairwise judging prompts.

One sample per question that both models answered correctly, showing their two
transcripts. With ``include_reverse`` each pair also appears in the opposite order, which
is what makes position bias visible in the results.
"""

from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

from configs.logs import logger
from evals.transcripts import REASONING_MARK, Transcript, read_header, read_transcripts

# Characters kept per transcript; must match the JudgeConfig default.
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
) -> Sample:
    """Render two transcripts of the same question as one judging sample.

    Options are shuffled per model, so each transcript is shown with its own option order.

    Args:
        first: shown as Transcript 1.
        second: shown as Transcript 2.
        target: which position holds the more capable model ('1' or '2').
        format_str: prompt template to fill.
        max_len: maximum characters kept per transcript.
    """
    assert first.id == second.id, "Sample IDs must match for comparison."
    assert first.question == second.question, "Sample inputs must match for comparison."

    text_1, text_2 = first.text[:max_len], second.text[:max_len]

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
    """
    self_pair = Path(major_path) == Path(minor_path)
    major = read_transcripts(major_path, require_both_correct, include_reasoning, items)
    minor = major if self_pair else read_transcripts(minor_path, require_both_correct, include_reasoning, items)

    ids = major.keys() & minor.keys()
    if include_reasoning and ids and not self_pair:
        # Reasoning shown on one side only is not a comparison of models: refuse the pair.
        shown = [any(REASONING_MARK in t.text for i in ids for t in side[i]) for side in (major, minor)]
        if shown[0] != shown[1]:
            logger.warning("%s vs %s: reasoning present on one side only, refusing the pair", major_path, minor_path)
            ids = set()
    samples = []
    for sample_id in sorted(ids):
        if self_pair:
            # Self-pair control: two distinct correct epochs of the same model.
            if len(major[sample_id]) < 2:
                continue
            first, second = major[sample_id][0], major[sample_id][1]
        else:
            first, second = major[sample_id][0], minor[sample_id][0]

        pair_kwargs = dict(format_str=format_str, max_len=max_len)
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
