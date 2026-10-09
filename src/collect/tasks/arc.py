"""ARC-Challenge with chain-of-thought prompting.

``inspect_evals/arc_challenge`` uses the bare ``multiple_choice()`` template, so models
answer with a lone ``ANSWER: X`` and the transcript carries nothing for a judge to compare.
``cot`` switches to inspect's CoT template ("Think step by step before answering."), the
same knob gpqa uses. It is a real task argument, so CoT runs hash to their own data dir.

``subset``/``seed`` take a fixed random draw of the 1172 test items, identical across models
(both are real task args, so the draw is part of the data-dir hash).
"""

from inspect_ai import Task, task
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_evals.arc.arc import arc_task


@task
def arc_challenge_cot(cot: bool = True, subset: int | None = None, seed: int = 0) -> Task:
    base = arc_task("ARC-Challenge")
    dataset = base.dataset
    if subset is not None:
        dataset.shuffle(seed=seed)
        dataset = dataset[:subset]
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=cot),
        scorer=choice(),
        version=base.version,
        metadata=base.metadata,
    )
