from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate, system_message

from configs.experiment import ExperimentConfig
from judge.dataset import build_dataset
from judge.scorer import judge_scorer

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript A and Transcript B, both coming to the correct solution.
One of the two models is more capable than the other, and your task is to identify which is the smarter model (A or B). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response is better and why. 2-4 sentences.>
ANSWER: <A or B, indicating which response is better.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>
"""


@task
def judge_task(
    major_path: str | Path,
    minor_path: str | Path,
    system_prompt: str = SYSTEM_PROMPT,
    max_length: int = 65536,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    experiment_name: str | None = None,
    task_name: str | None = None,
) -> Task:
    major_path = Path(major_path)
    minor_path = Path(minor_path)

    log_major = read_eval_log(major_path, header_only=True)
    log_minor = read_eval_log(minor_path, header_only=True)
    
    assert log_major.eval.task == log_minor.eval.task, "The two logs must be from the same task."

    major_name = log_major.eval.model
    minor_name = log_minor.eval.model
    benchmark_name = log_major.eval.task

    name = task_name or f"{benchmark_name}={major_path.stem}|{minor_path.stem}"

    return Task(
        name=name,
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            major_path=major_path,
            minor_path=minor_path,
            max_len=max_length,
            require_both_correct=require_both_correct,
            include_reverse=include_reverse,
        ),
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "major_path": str(major_path),
            "minor_path": str(minor_path),
            "system_prompt": system_prompt,
            "name_major": major_name,
            "name_minor": minor_name,
            "benchmark_name": benchmark_name,
            "require_both_correct": require_both_correct,
            "include_reverse": include_reverse,
            "experiment_name": experiment_name,
        },
    )


def build_judge_tasks(
    experiment: ExperimentConfig,
) -> list[Task]:
    tasks = []
    pairs = experiment.resolve_pairs()

    for major, minor in pairs:

        t = judge_task(
            major_path=major,
            minor_path=minor,
            system_prompt=experiment.system_prompt,
            max_length=experiment.max_length,
            require_both_correct=experiment.require_both_correct,
            include_reverse=experiment.include_reverse,
            experiment_name=experiment.name,
            task_name=experiment.task_name,
        )
        tasks.append(t)

    return tasks
