from inspect_ai import Task, task
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate, system_message

from dataset import build_dataset

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript A and Transcript B, both coming to the correct solution.
One of the two models is more capable than the other, and your task is to identify which is the smarter model (A or B). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response is better and why. 2-4 sentences.>
ANSWER: <A or B, indicating which response is better.>
CONFIDENCE: <A number between 0 and 1 indicating your confidence in your answer.>
"""

@task
def judge_task(
    log_major_path: str,
    log_minor_path: str,
    system_prompt: str = SYSTEM_PROMPT,
):
    log_major = read_eval_log(log_major_path)
    log_minor = read_eval_log(log_minor_path)
    
    name_major = log_major.eval.model
    name_minor = log_minor.eval.model
    benchmark_name = log_major.eval.dataset.name
        
    return Task(
        name=f'{benchmark_name}%{name_major}|{name_minor}',
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            log_major_path=log_major_path,
            log_minor_path=log_minor_path,
        ),
        solver=[system_message(system_prompt), generate()],
        scorer=judge_scorer(),
        metadata={
            "log_major_path": log_major_path,
            "log_minor_path": log_minor_path,
            "system_prompt": system_prompt,
            "name_major": name_major,
            "name_minor": name_minor,
            "benchmark_name": benchmark_name,
        }
    )