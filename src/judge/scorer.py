import re
from statistics import StatisticsError, correlation

from inspect_ai.scorer import (
    Score, Target, CORRECT, INCORRECT, NOANSWER,
    Metric, SampleScore, accuracy, metric, scorer, stderr, value_to_float
)
from inspect_ai.solver import TaskState

def parse_judge_output(output_text: str):
    """
    Parses ANSWER, CONFIDENCE, and EXPLANATION from the judge model's output text.

    ``answer`` comes from the ``ANSWER:`` line only; a verdict stated in prose is no verdict.
    """
    if not output_text:
        return None, None, None

    # Parse ANSWER (e.g., 'ANSWER: 1', '**ANSWER:** 1', 'ANSWER: Model 2', 'ANSWER: $1$';
    # some models wrap the answer in $..$, imitating the prompt's '$LETTER' placeholder).
    # The last match wins and the digit must stand alone: a judge that quotes a transcript's
    # own 'ANSWER: 128' before its verdict must not be read as '1'.
    answers = re.findall(r'ANSWER:[*_]*\s*(?:(?:Model|Transcript)\s+)?[*_$]*([12])(?!\d)', output_text, re.IGNORECASE)
    answer = answers[-1] if answers else None

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

# Metrics see epoch-reduced scores, where CORRECT/INCORRECT/NOANSWER have already become
# 1.0/0.0/0.0 -- so "answered" is read off ``Score.answer`` and "correct" via value_to_float.
_to_float = value_to_float()


def _answered(sample: SampleScore) -> bool:
    return sample.score.answer is not None


def _correct(sample: SampleScore) -> bool:
    return _to_float(sample.score.value) == 1.0


@metric
def answer_rate() -> Metric:
    """Fraction of judgements with an ``ANSWER:`` verdict.

    Reported separately because ``accuracy`` counts a missing verdict as wrong, which
    would hide a judge that keeps refusing, rambling, or ignoring the format.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        answered = sum(_answered(sample) for sample in scores)
        return answered / len(scores)

    return metric_fn


@metric
def position_bias() -> Metric:
    """Mean of +1 for a "2" verdict and -1 for a "1" verdict, over answered samples.

    Orders are counterbalanced, so a judge reading the transcripts sits near 0; -1 is pure
    primacy (always picks Transcript 1), +1 pure recency.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        answers = [s.score.answer for s in scores if _answered(s)]
        if not answers:
            return 0.0
        return sum(1 if a == "2" else -1 for a in answers) / len(answers)

    return metric_fn


@metric
def consistent_accuracy() -> Metric:
    """Fraction of questions where the judge picked the right model in *both* orders.

    Sample ids are ``<question>_1`` and ``<question>_2`` (see ``responses_to_sample``).
    ``accuracy`` gives half credit to a split pair, so a pure position-reader scores 0.5
    there and 0 here.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        by_question: dict[str, list] = {}
        for sample in scores:
            by_question.setdefault(str(sample.sample_id).rsplit("_", 1)[0], []).append(_correct(sample))
        pairs = [values for values in by_question.values() if len(values) == 2]
        if not pairs:
            return 0.0
        return sum(a and b for a, b in pairs) / len(pairs)

    return metric_fn


@metric
def length_corr() -> Metric:
    """Phi correlation between "Transcript 1 is longer" and "judge picked 1", over answered
    samples (``chars_1/2`` from the sample metadata). +1 = always picks the longer one."""

    def metric_fn(scores: list[SampleScore]) -> float:
        xs, ys = [], []
        for s in scores:
            md = s.sample_metadata or {}
            if not _answered(s) or "chars_1" not in md:
                continue
            xs.append(int(md["chars_1"] > md["chars_2"]))
            ys.append(int(s.score.answer == "1"))
        try:
            return correlation(xs, ys)
        except StatisticsError:  # fewer than two points, or a constant side
            return 0.0

    return metric_fn


@scorer(metrics=[accuracy(), stderr(), answer_rate(), position_bias(), consistent_accuracy(), length_corr()])
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

        metadata = {"confidence": confidence} if confidence is not None else {}

        return Score(value=value, answer=answer, explanation=explanation, metadata=metadata)
    return score
