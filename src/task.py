import re
from inspect_ai import Task, task
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import (
    Score, Target, CORRECT, INCORRECT, NOANSWER,
    accuracy, stderr, scorer
)
from inspect_ai.solver import TaskState, generate, system_message

from dataset import build_dataset

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript A and Transcript B, both coming to the correct solution.
One of the two models is more capable than the other, and your task is to identify which is the smarter model (A or B). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response is better and why. 2-4 sentences.>
ANSWER: <A or B, indicating which response is better.>
CONFIDENCE: <A number between 0 and 1 indicating your confidence in your answer.>
"""

def parse_judge_output(output_text: str):
    """
    Parses ANSWER, CONFIDENCE, and EXPLANATION from the judge model's output text.
    """
    if not output_text:
        return None, None, None

    # Parse ANSWER (e.g., 'ANSWER: A', 'ANSWER: Model B', 'ANSWER: B')
    answer_match = re.search(r'ANSWER:\s*(?:Model\s+)?\*?([AB])\*?', output_text, re.IGNORECASE)
    answer = answer_match.group(1).upper() if answer_match else None

    # Parse CONFIDENCE (e.g., 'CONFIDENCE: 0.85', 'CONFIDENCE: 85%')
    conf_match = re.search(r'CONFIDENCE:\s*\*?([0-9]+(?:\.[0-9]+)?%?)\*?', output_text, re.IGNORECASE)
    confidence = None
    if conf_match:
        val_str = conf_match.group(1).rstrip('%')
        try:
            val = float(val_str)
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            confidence = val
        except ValueError:
            pass

    # Parse EXPLANATION (text after EXPLANATION: up to the next field header or end of string)
    expl_match = re.search(r'EXPLANATION:\s*(.*?)(?=\n\s*(?:ANSWER|CONFIDENCE):|\Z)', output_text, re.DOTALL | re.IGNORECASE)
    explanation = expl_match.group(1).strip() if expl_match else None

    return answer, confidence, explanation

@scorer(metrics=[accuracy(), stderr()])
def judge_scorer():
    async def score(state: TaskState, target: Target):
        output_text = state.output.completion if state.output else ""
        answer, confidence, explanation = parse_judge_output(output_text)

        target_str = target.text if isinstance(target.text, str) else target.text[0]
        if answer is None:
            value = NOANSWER
        elif answer == target_str:
            value = CORRECT
        else:
            value = INCORRECT

        metadata = {}
        if confidence is not None:
            metadata["confidence"] = confidence

        return Score(
            value=value,
            answer=answer,
            explanation=explanation,
            metadata=metadata if metadata else None
        )
    return score

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