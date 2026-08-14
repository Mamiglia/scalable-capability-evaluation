# scripts/analyses/direct-judge/score_task.py
"""Single-transcript capability scoring: prompt, dataset, task, scorer.

Pointwise control for the pairwise judge: same transcripts and judges, but the
judge sees one transcript at a time and rates the model 0-100.
"""
import re
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score, Target, NOANSWER, mean, stderr, scorer
from inspect_ai.solver import TaskState, generate, system_message

from judge.util import extract_transcript, extract_score

SYSTEM_PROMPT = """You are judging the response of an AI model to an evaluation benchmark question.
You will be given a question and the response from one model, labeled Transcript. The transcript comes to the correct solution.

For multiple choice questions, the transcript is preceded by the options as the model saw them.

Your task is to rate the capability of the model that produced the transcript, on a 0-100 scale anchored to the current model population: 0 = far below a typical current language model, 50 = a typical current language model, 80 = the best current frontier models, 100 = clearly beyond any current model. You MUST follow this reply format:

EXPLANATION: <What the transcript reveals about the model's capability. 2-4 sentences.>
SCORE: <An integer between 0 and 100, your capability rating for the model that produced the transcript.>
CONFIDENCE: <An integer between 1 and 5, how sure you are of the rating: 1 = the transcript is uninformative and you are guessing, 5 = certain. Rate the rating, not the explanation.>
"""

USER_PROMPT = """Input question:
---
{question}
---

{options}Transcript:
---
{transcript}
---

Rate the capability of the model that produced this transcript. Answer in the instructed format.
"""

OPTIONS_BLOCK = """Multiple choice options as shown to the model:
{options}

"""


def format_options(sample) -> str:
    """Render the options as the model saw them. Empty if it had none."""
    choices = getattr(sample, "choices", None)
    if not choices:
        return ""
    options = "\n".join(
        f"{chr(ord('A') + i)}) {choice.strip()}" for i, choice in enumerate(choices)
    )
    return OPTIONS_BLOCK.format(options=options)


def parse_score_output(output_text: str):
    """Parse SCORE, CONFIDENCE, and EXPLANATION from the judge's reply."""
    if not output_text:
        return None, None, None

    score = None
    m = re.search(r"SCORE:[*_]*\s*[*_]*(\d{1,3})\b", output_text, re.IGNORECASE)
    if m and 0 <= int(m.group(1)) <= 100:
        score = int(m.group(1))

    confidence = None
    m = re.search(r"CONFIDENCE:[*_]*\s*[*_]*([0-9]+(?:\.[0-9]+)?)", output_text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 1.0 <= val <= 5.0:
            confidence = int(val) if val.is_integer() else val

    m = re.search(r"EXPLANATION:[*_]*\s*(.*?)(?=\n\s*[*_]*(?:SCORE|CONFIDENCE):|\Z)",
                  output_text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else None

    return score, confidence, explanation


@scorer(metrics=[mean(), stderr()])
def score_scorer():
    async def score(state: TaskState, target: Target):
        output_text = state.output.completion if state.output else ""
        val, confidence, explanation = parse_score_output(output_text)
        metadata = {"confidence": confidence} if confidence is not None else None
        if val is None:
            return Score(value=NOANSWER, answer=None, explanation=explanation, metadata=metadata)
        return Score(value=val, answer=str(val), explanation=explanation, metadata=metadata)
    return score


def build_score_dataset(
    log_path: Path,
    item_ids: list,
    max_len: int = 65536,
    include_reasoning: bool = False,
) -> MemoryDataset:
    """One sample per item id: the model's first correct epoch, same epoch
    convention as the pairwise dataset."""
    log = read_eval_log(str(log_path))
    grouped: dict = {}
    for s in log.samples:
        if extract_score(s):
            grouped.setdefault(s.id, []).append(s)
    for group in grouped.values():
        group.sort(key=lambda s: getattr(s, "epoch", 0))

    samples = []
    for item_id in item_ids:
        group = grouped.get(item_id)
        if not group:
            continue  # id not correct in this log; ids come from both-correct pools so this is rare
        s = group[0]
        transcript = extract_transcript(s, include_reasoning=include_reasoning)[:max_len]
        samples.append(Sample(
            id=item_id,
            input=USER_PROMPT.format(
                question=s.input, options=format_options(s), transcript=transcript),
        ))
    return MemoryDataset(samples=samples, name=f"Capability scoring of {log.eval.model}")


def score_task(
    log_path: str | Path,
    item_ids: list,
    system_prompt: str = SYSTEM_PROMPT,
    max_length: int = 65536,
    include_reasoning: bool = False,
    task_name: str | None = None,
    experiment_name: str | None = None,
) -> Task:
    log_path = Path(log_path)
    header = read_eval_log(str(log_path), header_only=True)
    return Task(
        name=task_name or f"score={log_path.stem}",
        description="Rate the capability of a model from a single response transcript.",
        dataset=build_score_dataset(
            log_path, item_ids, max_len=max_length, include_reasoning=include_reasoning),
        solver=[system_message(system_prompt), generate()],
        scorer=score_scorer(),
        metadata={
            "log_path": str(log_path),
            "model_name": header.eval.model,
            "benchmark_name": header.eval.task,
            "system_prompt": system_prompt,
            "max_length": max_length,
            "include_reasoning": include_reasoning,
            "experiment_name": experiment_name,
        },
    )
