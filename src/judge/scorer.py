import re

from inspect_ai.scorer import (
    Score, Target, CORRECT, INCORRECT, NOANSWER,
    Metric, SampleScore, accuracy, metric, scorer, stderr
)
from inspect_ai.solver import TaskState

def parse_judge_output(output_text: str):
    """
    Parses ANSWER, CONFIDENCE, and EXPLANATION from the judge model's output text.

    ``answer`` comes from the ``ANSWER:`` line; when it is missing, ``lenient_answer`` is
    the last "Transcript/Model/Response 1|2" the judge mentions, for judges that state a
    verdict but ignore the format (Phi-3 does this a lot).
    """
    if not output_text:
        return None, None, None

    # Parse ANSWER (e.g., 'ANSWER: 1', '**ANSWER:** 1', 'ANSWER: Model 2', 'ANSWER: $1$';
    # some models wrap the answer in $..$, imitating the prompt's '$LETTER' placeholder).
    # The last match wins and the digit must stand alone: a judge that quotes a transcript's
    # own 'ANSWER: 128' before its verdict must not be read as '1'.
    answers = re.findall(r'ANSWER:[*_]*\s*(?:(?:Model|Transcript)\s+)?[*_$]*([12])(?!\d)', output_text, re.IGNORECASE)
    answer = answers[-1] if answers else None
    if answer is None:
        mentions = re.findall(r'\b(?:transcript|model|response)\s*#?\s*([12])\b', output_text, re.IGNORECASE)
        answer = mentions[-1] if mentions else None

    # Parse CONFIDENCE on a 1-5 scale (e.g., 'CONFIDENCE: 4', '**CONFIDENCE:** 5')
    conf_match = re.search(r'CONFIDENCE:[*_]*\s*[*_$]*([0-9]+(?:\.[0-9]+)?)', output_text, re.IGNORECASE)
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

@metric
def answer_rate() -> Metric:
    """Fraction of judgements with a verdict, in the reply format or not.

    Reported separately because ``accuracy`` counts a missing verdict as wrong, which
    would hide a judge that keeps refusing or rambling.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        answered = sum(sample.score.value != NOANSWER for sample in scores)
        return answered / len(scores)

    return metric_fn


@metric
def format_rate() -> Metric:
    """Fraction of judgements that used the ``ANSWER:`` line (see ``parse_judge_output``)."""

    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        formatted = sum(bool((sample.score.metadata or {}).get("format")) for sample in scores)
        return formatted / len(scores)

    return metric_fn


@metric
def consistency() -> Metric:
    """Fraction of question pairs where the judge picked the same model in both orders.

    Sample ids are ``<question>_1`` and ``<question>_2`` (see ``responses_to_sample``); the
    two orders name the same model iff both verdicts are correct or both are wrong. Below
    0.5 means the judge is mostly reading position, not the transcripts.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        by_question: dict[str, list] = {}
        for sample in scores:
            by_question.setdefault(str(sample.sample_id).rsplit("_", 1)[0], []).append(sample.score.value)
        pairs = [values for values in by_question.values() if len(values) == 2]
        if not pairs:
            return 0.0
        agree = sum(a == b and a != NOANSWER for a, b in pairs)
        return agree / len(pairs)

    return metric_fn


@scorer(metrics=[accuracy(), stderr(), answer_rate(), format_rate(), consistency()])
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

        metadata = {"format": bool(re.search(r"ANSWER:", output_text, re.IGNORECASE))}
        if confidence is not None:
            metadata["confidence"] = confidence

        return Score(value=value, answer=answer, explanation=explanation, metadata=metadata)
    return score
