from pathlib import Path

from inspect_ai import Task
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate, system_message

from configs.experiment import ExperimentConfig
from judge.dataset import build_dataset
from judge.scorer import judge_scorer

DEFAULT_SYSTEM_PROMPT = "assets/system_prompt/default.txt"
# Same prompt plus a note that both transcripts stop mid-reasoning on purpose.
TRUNCATED_SYSTEM_PROMPT = "assets/system_prompt/truncated.txt"


def judge_task(
    major_path: str | Path,
    minor_path: str | Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_length: int = 65536,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    include_reasoning: bool = True,
    truncate_frac: float | None = None,
    experiment_name: str | None = None,
    task_name: str | None = None,
) -> Task:
    """Build the judging task that compares one major eval log against one minor log."""
    major_path = Path(major_path)
    minor_path = Path(minor_path)

    log_major = read_eval_log(major_path, header_only=True)
    log_minor = read_eval_log(minor_path, header_only=True)

    if log_major.eval.task.rsplit("/", 1)[-1] != log_minor.eval.task.rsplit("/", 1)[-1]:
        raise ValueError(
            f"The two logs must be from the same task, got {log_major.eval.task!r} "
            f"({major_path}) and {log_minor.eval.task!r} ({minor_path})"
        )

    benchmark_name = log_major.eval.task
    name = task_name or f"{benchmark_name}={major_path.stem}|{minor_path.stem}"

    if truncate_frac is not None and system_prompt == DEFAULT_SYSTEM_PROMPT:
        # Truncated transcripts stop mid-reasoning, so the judge has to be told not to
        # read that as a failure.
        system_prompt = TRUNCATED_SYSTEM_PROMPT

    return Task(
        name=name,
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            major_path=major_path,
            minor_path=minor_path,
            max_len=max_length,
            require_both_correct=require_both_correct,
            include_reverse=include_reverse,
            include_reasoning=include_reasoning,
            truncate_frac=truncate_frac,
        ),
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
            "truncate_frac": truncate_frac,
            "experiment_name": experiment_name,
        },
    )


def build_judge_tasks(experiment: ExperimentConfig) -> list[Task]:
    """One judging task per <major log, minor log> pair the experiment resolves to."""
    pairs = experiment.resolve_pairs()
    tasks = []
    for index, (major, minor) in enumerate(pairs):
        # Several pairs in one eval_set need distinct names, or their logs collide.
        # Inspect builds file:// URLs from log paths, so "#" in the filename truncates the path
        # at urlparse() time and breaks header reads.
        task_name = experiment.task_name if len(pairs) == 1 else f"{experiment.task_name}+{index}"
        tasks.append(
            judge_task(
                major_path=major,
                minor_path=minor,
                system_prompt=experiment.system_prompt,
                max_length=experiment.max_length,
                require_both_correct=experiment.require_both_correct,
                include_reverse=experiment.include_reverse,
                include_reasoning=experiment.include_reasoning,
                truncate_frac=experiment.truncate_frac or None,
                experiment_name=experiment.name,
                task_name=task_name,
            )
        )
    return tasks
