"""MathArena Apex: hard final-answer olympiad problems.

Datasets: MathArena/apex-shortlist (47 problems), MathArena/apex_2025 (12).
Prompt, solver and scorer are reused from inspect_evals' MATH task. One scorer only:
the judge pipeline (`judge.util.extract_score`) refuses samples with several scores.
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset

from inspect_evals.math.math import expression_exact_match_sympy, math_solver


@task
def apex(dataset: str = "MathArena/apex-shortlist") -> Task:
    """MathArena Apex (shortlist by default)."""
    return Task(
        dataset=hf_dataset(path=dataset, split="train", sample_fields=record_to_sample),
        solver=math_solver(fewshot=0, fewshot_seed=0),
        scorer=expression_exact_match_sympy(),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=str(record["problem_idx"]),
        input=record["problem"],
        target=str(record["answer"]),
        metadata={"source": record["source"]},
    )
