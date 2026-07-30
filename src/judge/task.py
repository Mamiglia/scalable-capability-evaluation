from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from judge.dataset import build_dataset
from judge.scorer import judge_scorer
from registry.query import resolve_eval_path

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript A and Transcript B, both coming to the correct solution.
One of the two models is more capable than the other, and your task is to identify which is the smarter model (A or B). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response is better and why. 2-4 sentences.>
ANSWER: <A or B, indicating which response is better.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>
"""

@task
def judge_task(
    major: str = "",
    minor: str = "",
    benchmark_name: str = "",
    task_args: dict | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_length: int = 65536,
    require_both_correct: bool = True,
    include_reverse: bool = True,
    log_major_path: str | None = None,
    log_minor_path: str | None = None,
    experiment_name: str | None = None,
    **kwargs,
):
    major = major or kwargs.get("major_model", "")
    minor = minor or kwargs.get("minor_model", "")
    log_major_path = log_major_path or str(
        resolve_eval_path(major, benchmark_name, task_args)
    )
    log_minor_path = log_minor_path or str(
        resolve_eval_path(minor, benchmark_name, task_args)
    )
        
    return Task(
        name=f'{benchmark_name}={major}|{minor}',
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            log_major_path=log_major_path,
            log_minor_path=log_minor_path,
            max_len=max_length,
            require_both_correct=require_both_correct,
            include_reverse=include_reverse,
        ),
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "log_major_path": log_major_path,
            "log_minor_path": log_minor_path,
            "system_prompt": system_prompt,
            "name_major": major,
            "name_minor": minor,
            "benchmark_name": benchmark_name,
            "task_args": task_args,
            "require_both_correct": require_both_correct,
            "include_reverse": include_reverse,
            "experiment_name": experiment_name,
        }
    )
