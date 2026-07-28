from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.log import read_eval_log

from src.util import format_message, extract_score

USER_PROMPT = """Question (shared by both transcripts):
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

Reply with the reply format only."""

def responses_to_sample(
    sample_a: Sample,
    sample_b: Sample,
    target: str = 'A',
    format_str: str = USER_PROMPT,
    max_len: int = 16000,
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
    max_len: int = 16000,
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
    
    assert len(log_major.samples) == len(log_minor.samples), "Eval logs must have the same number of samples."
    
    samples = []
    
    for sample_a, sample_b in zip(log_major.samples, log_minor.samples):
        if not extract_score(sample_a) or not extract_score(sample_b):
            # Only include samples where both models provided correct responses
            continue
        samples.append(responses_to_sample(sample_a, sample_b, target='A', format_str=format_str, max_len=max_len))
        samples.append(responses_to_sample(sample_b, sample_a, target='B', format_str=format_str, max_len=max_len))
        
    name_major = log_major.eval.model
    name_minor = log_minor.eval.model
    benchmark_name = log_major.eval.dataset.name
    
    return MemoryDataset(samples=samples, name=f"Comparison of {name_major} and {name_minor} over the {benchmark_name} benchmark")


if __name__ == "__main__":
    # Example usage
    log_major_path = Path("data/gemma_2b/gpqa_diamond/k0/seed_100000000001/2026-06-28T11-22-17-00-00_gpqa-diamond_B8pfKA6zF3zZMGrumTGWaU.eval")
    log_minor_path = Path("data/gemma_26b/gpqa_diamond/k0/seed_100000000001/2026-07-22T13-13-13-00-00_gpqa-diamond_ktJitVYtpBtDPz4bNpbZjp.eval")

    
    dataset = build_dataset(log_major_path, log_minor_path)
    print(f"Built dataset with {len(dataset.samples)} samples.")
    
    