from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.log import read_eval_log

from util import format_message, extract_score

USER_PROMPT = """Input question:
---
{question}
---

Transcript A:
---
{transcript_a}
---

Transcript B:
---
{transcript_b}
---

Which of the two transcripts is from the more capable model? Please answer in the instructed format below.
"""

def responses_to_sample(
    sample_a: Sample,
    sample_b: Sample,
    target: str = 'A',
    format_str: str = USER_PROMPT,
    max_len: int = 24000,
) -> Sample:
    """
    Converts two model samples into a single formatted string representation.

    Args:
        sample_a (Sample): The sample from Model A.
        sample_b (Sample): The sample from Model B.
        format_str (str): The format string to use for the output.
        max_len (int): The maximum length of each model's response.

    Returns:
        Sample: A formatted sample representing both model responses.
    """
    assert sample_a.id == sample_b.id, "Sample IDs must match for comparison."
    assert sample_a.input == sample_b.input, "Sample inputs must match for comparison."
    input_text = sample_a.input
    
    output_a = format_message(sample_a.output.choices[0].message)[:max_len]
    output_b = format_message(sample_b.output.choices[0].message)[:max_len]
    
    text = format_str.format(
        question=input_text,
        transcript_a=output_a,
        transcript_b=output_b,
    )
    return Sample(
        id=f"{sample_a.id}_{target}",
        input=text,
        target=target
    )
    
def build_dataset(
    log_major_path: Path,
    log_minor_path: Path,
    format_str: str = USER_PROMPT,
    max_len: int = 24000,
):
    """
    Builds a dataset from two eval logs, comparing the responses of two models.

    Args:
        log_major_path (Path): Path to the eval log for the major model.
        log_minor_path (Path): Path to the eval log for the minor model.
        format_str (str): The format string to use for the output.
        max_len (int): The maximum length of each model's response.
    Returns:
        Dataset: A dataset containing the formatted comparisons of model responses.
    """
    log_major = read_eval_log(log_major_path)
    log_minor = read_eval_log(log_minor_path)
    
    # Only include samples where both models provided correct responses
    minor_samples = {s.id: s for s in log_minor.samples if hasattr(s, "id") and extract_score(s) == True}
    major_samples = {s.id: s for s in log_major.samples if hasattr(s, "id") and extract_score(s) == True}

    samples = []
    
    for sample_id in minor_samples.keys() & major_samples.keys():
        sample_a = major_samples[sample_id]
        sample_b = minor_samples[sample_id]
        
        samples.append(responses_to_sample(sample_a, sample_b, target='A', format_str=format_str, max_len=max_len))
        samples.append(responses_to_sample(sample_b, sample_a, target='B', format_str=format_str, max_len=max_len))
        
    name_major = log_major.eval.model
    name_minor = log_minor.eval.model
    benchmark_name = log_major.eval.dataset.name
    
    return MemoryDataset(samples=samples, name=f"Comparison of {name_major} and {name_minor} over the {benchmark_name} benchmark")


if __name__ == "__main__":
    # Example usage
    log_major_path = Path("data/gemma_2b/gpqa_diamond/100000000001.eval")
    log_minor_path = Path("data/gemma_26b/gpqa_diamond/100000000001.eval")

    
    dataset = build_dataset(log_major_path, log_minor_path)
    print(f"Built dataset with {len(dataset.samples)} samples.")
    
    