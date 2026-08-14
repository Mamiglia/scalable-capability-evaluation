"""GPQA with a style-standardizing system message prepended."""

from inspect_ai import Task, task
from inspect_ai.solver import chain, system_message
from inspect_evals.gpqa import gpqa_diamond


@task
def gpqa_instructed(
    instruction: str,
    cot: bool = True,
    epochs: int = 4,
    high_level_domain: str | list[str] | None = None,
    subdomain: str | list[str] | None = None,
) -> Task:
    """gpqa_diamond, with `instruction` prepended as a system message.

    Params mirror gpqa_diamond explicitly: inspect's task loader drops
    task_args not named in the signature (**kwargs does not receive them).
    """
    t = gpqa_diamond(
        cot=cot,
        epochs=epochs,
        high_level_domain=high_level_domain,
        subdomain=subdomain,
    )
    t.solver = chain(system_message(instruction), t.solver)
    return t
