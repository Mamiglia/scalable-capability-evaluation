import re
from math import nan
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
    return _to_float(sample.score.value) > 0.5


def _picked_1(sample: SampleScore) -> float | None:
    """P(judge picked Transcript 1), or None if it never gave a verdict.

    Read off the *value*, not ``Score.answer``: with epochs > 1 the mean reducer blanks
    ``answer`` unless every epoch agreed, so filtering on it keeps only the samples the
    judge was unanimous on -- exactly the ones a length heuristic drives (measured: 0.79
    vs 0.63 on a 5-epoch run). The value is the fraction of epochs scored correct, and the
    sample id suffix says which position was correct, so the pick rate follows.
    """
    value = _to_float(sample.score.value)
    if sample.score.answer is None and value == 0.0:  # no verdict in any epoch
        return None
    target = str(sample.sample_id).rsplit("_", 1)[-1]
    return value if target == "1" else 1.0 - value


@metric
def answer_rate() -> Metric:
    """Fraction of judgements with an ``ANSWER:`` verdict.

    Reported separately because ``accuracy`` counts a missing verdict as wrong, which
    would hide a judge that keeps refusing, rambling, or ignoring the format.

    Caveat: with epochs > 1 the mean reducer keeps ``answer`` only when every epoch agreed,
    so this reads as an epoch-unanimity rate rather than an answer rate.
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
        picks = [p for p in map(_picked_1, scores) if p is not None]
        if not picks:
            return nan
        return sum(1.0 - 2.0 * p for p in picks) / len(picks)

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
            return nan
        return sum(a and b for a, b in pairs) / len(pairs)

    return metric_fn


@metric
def length_corr() -> Metric:
    """Correlation between "Transcript 1 is longer" and P(judge picks 1), over samples that
    got a verdict (``chars_1/2`` from the sample metadata). +1 = always picks the longer one.

    NaN, not 0.0, when it cannot be computed -- logs collected before ``chars_1/2`` existed
    carry no lengths, and "no data" must not read as "no length bias".
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        xs, ys = [], []
        for s in scores:
            md = s.sample_metadata or {}
            picked_1 = _picked_1(s)
            if picked_1 is None or "chars_1" not in md:
                continue
            xs.append(float(md["chars_1"] > md["chars_2"]))
            ys.append(picked_1)
        try:
            return correlation(xs, ys)
        except StatisticsError:  # fewer than two points, or a constant side
            return nan

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


def parse_direct_judge_output(output_text: str):
    """Read the final ANSWER field as a 1–10 rating, preserving decimals."""
    text = output_text or ""
    answers = []
    for line in text.splitlines():
        if re.match(r"^[ \t]*[*_`]*(?:ANSWER|EXPLANATION)[*_`]*:", line, re.I):
            fields = list(re.finditer(r"[*_`]*ANSWER[*_`]*:[*_`]*[ \t]*", line, re.I))
            if fields:
                answers.append(line[fields[-1].end():])
    match = re.match(r"[*_`$]*[ \t]*([0-9]+(?:\.[0-9]+)?)(?![0-9]|\.[0-9])"
                     r"(?:[ \t]*/[ \t]*10)?[ \t]*[*_`$]*(?=$|[^0-9A-Za-z/])",
                     answers[-1].strip()) if answers else None
    answer = match.group(1) if match else None
    if answer is not None and not 1 <= float(answer) <= 10:
        answer = None
    _, _, explanation = parse_judge_output(text)
    return answer, explanation


@scorer(metrics=[answer_rate()])
def direct_judge_scorer():
    """Store the capability rating itself; missing ratings stay NaN, with no artificial target."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion if state.output else ""
        answer, explanation = parse_direct_judge_output(text)
        return Score(value=float(answer) if answer is not None else nan,
                     answer=answer, explanation=explanation)
    return score
