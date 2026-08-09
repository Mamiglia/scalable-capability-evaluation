from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log

from judge.util import extract_transcript, extract_score

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


def format_options(sample: Sample, n: int) -> str:
    """Render a sample's options in the order that model saw them. Empty if it had none."""
    choices = getattr(sample, "choices", None)
    if not choices:
        return ""
    options = "\n".join(
        f"{chr(ord('A') + i)}) {choice.strip()}" for i, choice in enumerate(choices)
    )
    return OPTIONS_BLOCK.format(n=n, options=options)


def strip_answer(text: str) -> str:
    """Drop the trailing ANSWER line."""
    idx = text.rfind("ANSWER:")
    return text[:idx].rstrip() if idx != -1 else text


def truncate_pair(a: str, b: str, frac: float) -> tuple[str, str]:
    """Cut both transcripts to frac of the shorter one's length, answers removed."""
    a, b = strip_answer(a), strip_answer(b)
    cut = max(1, int(frac * min(len(a), len(b))))
    return a[:cut], b[:cut]


def responses_to_sample(
    sample_1: Sample,
    sample_2: Sample,
    target: str = '1',
    format_str: str = USER_PROMPT,
    max_len: int = 24000,
    include_reasoning: bool = True,
    truncate_frac: float | None = None,
) -> Sample:
    """
    Converts two model samples into a single formatted string representation.

    Options are shuffled per model, so each transcript is shown with its own
    option order.

    Args:
        sample_1 (Sample): The sample shown as Transcript 1.
        sample_2 (Sample): The sample shown as Transcript 2.
        format_str (str): The format string to use for the output.
        max_len (int): The maximum length of each model's response.
        include_reasoning (bool): Whether to include CoT reasoning blocks.
        truncate_frac (float | None): If set, cut both transcripts to this
            fraction of the shorter one's length (length-control condition).

    Returns:
        Sample: A formatted sample representing both model responses.
    """
    assert sample_1.id == sample_2.id, "Sample IDs must match for comparison."
    assert sample_1.input == sample_2.input, "Sample inputs must match for comparison."
    input_text = sample_1.input

    output_1 = extract_transcript(sample_1, include_reasoning=include_reasoning)[:max_len]
    output_2 = extract_transcript(sample_2, include_reasoning=include_reasoning)[:max_len]
    if truncate_frac is not None:
        output_1, output_2 = truncate_pair(output_1, output_2, truncate_frac)

    text = format_str.format(
        question=input_text,
        options_1=format_options(sample_1, 1),
        options_2=format_options(sample_2, 2),
        transcript_1=output_1,
        transcript_2=output_2,
    )
    return Sample(
        id=f"{sample_1.id}_{target}",
        input=text,
        target=target
    )
    
def build_dataset(
    major_path: Path,
    minor_path: Path,
    format_str: str = USER_PROMPT,
    max_len: int = 24000,
    require_both_correct: bool = True,
include_reverse: bool = True,
    include_reasoning: bool = True,
    truncate_frac: float | None = None,
):
    """
    Builds a dataset from two eval logs, comparing the responses of two models.

    Args:
        major_path (Path): Path to the eval log for the major model.
        minor_path (Path): Path to the eval log for the minor model.
        format_str (str): The format string to use for the output.
        max_len (int): The maximum length of each model's response.
        include_reasoning (bool): Whether to include CoT reasoning blocks.
    Returns:
        Dataset: A dataset containing the formatted comparisons of model responses.
    """
    log_major = read_eval_log(major_path)
    log_minor = read_eval_log(minor_path)

    def eligible(sample: Sample) -> bool:
        return hasattr(sample, "id") and (
            not require_both_correct or extract_score(sample)
        )

    def samples_by_id(log) -> dict:
        grouped: dict = {}
        for s in log.samples:
            if eligible(s):
                grouped.setdefault(s.id, []).append(s)
        for group in grouped.values():
            group.sort(key=lambda s: getattr(s, "epoch", 0))
        return grouped

    self_pair = Path(major_path) == Path(minor_path)
    major_samples = samples_by_id(log_major)
    minor_samples = major_samples if self_pair else samples_by_id(log_minor)

    samples = []

    for sample_id in sorted(minor_samples.keys() & major_samples.keys(), key=str):
        if self_pair:
            # For self-pair control, make sure to select two distinct correct epochs of the same model.
            group = major_samples[sample_id]
            if len(group) < 2:
                continue
            sample_major, sample_minor = group[0], group[1]
        else:
            sample_major = major_samples[sample_id][0]
            sample_minor = minor_samples[sample_id][0]

        samples.append(responses_to_sample(sample_major, sample_minor, target='1', format_str=format_str, max_len=max_len, include_reasoning=include_reasoning, truncate_frac=truncate_frac))

        if include_reverse:
            samples.append(responses_to_sample(sample_minor, sample_major, target='2', format_str=format_str, max_len=max_len, include_reasoning=include_reasoning, truncate_frac=truncate_frac))
        
    name_major = log_major.eval.model
    name_minor = log_minor.eval.model
    benchmark_name = log_major.eval.dataset.name
    
    return MemoryDataset(samples=samples, name=f"Comparison of {name_major} and {name_minor} over the {benchmark_name} benchmark")
