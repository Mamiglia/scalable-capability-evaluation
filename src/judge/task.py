"""One inspect Task: a judge compares two models' correct transcripts of the same items."""

from pathlib import Path

from inspect_ai import Task
from inspect_ai.solver import generate, system_message

from configs.logs import logger
from evals.transcripts import read_header
from judge.dataset import MAX_LENGTH, build_dataset
from judge.scorer import judge_scorer

DEFAULT_SYSTEM_PROMPT = "assets/system_prompt/default.txt"


def judge_task(
    major_path: str | Path,
    minor_path: str | Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_length: int = MAX_LENGTH,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    include_reasoning: bool = True,
    task_name: str | None = None,
    items: set[str] | None = None,
    metadata: dict | None = None,
) -> Task | None:
    """Build the judging task that compares one major eval log against one minor log.

    Returns None when no question survives the filters (e.g. both models never agree on a
    correct answer), since inspect refuses an empty dataset."""
    major_path = Path(major_path)
    minor_path = Path(minor_path)

    log_major = read_header(major_path)
    log_minor = read_header(minor_path)

    if log_major.eval.task.rsplit("/", 1)[-1] != log_minor.eval.task.rsplit("/", 1)[-1]:
        raise ValueError(
            f"The two logs must be from the same task, got {log_major.eval.task!r} "
            f"({major_path}) and {log_minor.eval.task!r} ({minor_path})"
        )

    benchmark_name = log_major.eval.task
    name = task_name or f"{benchmark_name}={major_path.stem}|{minor_path.stem}"

    dataset = build_dataset(
        major_path=major_path,
        minor_path=minor_path,
        max_len=max_length,
        require_both_correct=require_both_correct,
        include_reverse=include_reverse,
        include_reasoning=include_reasoning,
        items=items,
    )
    if len(dataset) == 0:
        logger.warning("%s: no samples after filtering (%s vs %s), skipping", name, major_path, minor_path)
        return None

    return Task(
        name=name,
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=dataset,
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "major_path": str(major_path),
            "minor_path": str(minor_path),
            "system_prompt": system_prompt,
            "name_major": log_major.eval.model,
            "name_minor": log_minor.eval.model,
            "benchmark_name": benchmark_name,
            "require_both_correct": require_both_correct,
            "include_reverse": include_reverse,
            "include_reasoning": include_reasoning,
            "max_length": max_length,
            **(metadata or {}),
        },
    )
