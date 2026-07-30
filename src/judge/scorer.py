import re

from inspect_ai.scorer import (
    Score, Target, CORRECT, INCORRECT, NOANSWER,
    accuracy, stderr, scorer
)
from inspect_ai.solver import TaskState

def parse_judge_output(output_text: str):
    """
    Parses ANSWER, CONFIDENCE, and EXPLANATION from the judge model's output text.
    """
    if not output_text:
        return None, None, None

    # Parse ANSWER (e.g., 'ANSWER: A', '**ANSWER:** A', 'ANSWER: Model B', 'ANSWER: B')
    answer_match = re.search(r'ANSWER:[*_]*\s*(?:Model\s+)?[*_]*([AB])', output_text, re.IGNORECASE)
    answer = answer_match.group(1).upper() if answer_match else None

    # Parse CONFIDENCE on a 1-5 scale (e.g., 'CONFIDENCE: 4', '**CONFIDENCE:** 5')
    conf_match = re.search(r'CONFIDENCE:[*_]*\s*[*_]*([0-9]+(?:\.[0-9]+)?)', output_text, re.IGNORECASE)
    confidence = None
    if conf_match:
        try:
            val = float(conf_match.group(1))
            if 1.0 <= val <= 5.0:
                confidence = int(val) if val.is_integer() else val
        except ValueError:
            pass

    # Parse EXPLANATION (text after EXPLANATION: up to the next field header or end of string)
    expl_match = re.search(r'EXPLANATION:[*_]*\s*(.*?)(?=\n\s*[*_]*(?:ANSWER|CONFIDENCE):|\Z)', output_text, re.DOTALL | re.IGNORECASE)
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
