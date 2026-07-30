from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message

from dataset import build_dataset
from scorer import judge_scorer
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
    major_model: str,
    minor_model: str,
    benchmark_name: str,
    task_args: dict | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_length: int = 65536,
):
    log_major_path = resolve_eval_path(major_model, benchmark_name, task_args)
    log_minor_path = resolve_eval_path(minor_model, benchmark_name, task_args)
        
    return Task(
        name=f'{benchmark_name}={major_model}|{minor_model}',
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            log_major_path=log_major_path,
            log_minor_path=log_minor_path,
            max_len=max_length
        ),
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "log_major_path": log_major_path,
            "log_minor_path": log_minor_path,
            "system_prompt": system_prompt,
            "name_major": major_model,
            "name_minor": minor_model,
            "benchmark_name": benchmark_name,
        }
    )
