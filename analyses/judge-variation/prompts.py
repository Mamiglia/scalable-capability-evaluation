"""The four judge system prompts. a/e share BASELINE."""
from configs.experiment import SYSTEM_PROMPT as BASELINE

_IGNORE = """
Ignore superficial properties of the transcripts, such as their length, their formatting, and whether their tone is colloquial or professional. Judge only the quality of the reasoning process and the content."""

_IGNORE_CONF = _IGNORE + """
Also, do NOT consider how confident or uncertain the transcripts sound; confident prose is not evidence of capability."""

def _insert(base, extra):
    # extra guidance goes right before the reply-format paragraph
    marker = "One of the two models is more capable"
    assert marker in base
    return base.replace(marker, extra.strip() + "\n\n" + marker)


IGNORE_SUPERFICIAL = _insert(BASELINE, _IGNORE)
IGNORE_SUPERFICIAL_CONFIDENCE = _insert(BASELINE, _IGNORE_CONF)

PICK_WEAKER = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript 1 and Transcript 2, both coming to the correct solution.

For multiple choice questions, the options were shuffled independently for each model, so an option letter in Transcript 1 does not refer to the same option as that letter in Transcript 2. Each transcript is preceded by the options as that model saw them. Read each letter against that transcript's own option list.

One of the two models is less capable than the other, and your task is to identify which is the less capable model (1 or 2). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response was produced by the less capable model and why. 2-4 sentences.>
ANSWER: <1 or 2, indicating which response was produced by the less capable model.>
CONFIDENCE: <An integer between 1 and 5, how sure you are that the model you picked is the less capable one: 1 = the two are indistinguishable and you are guessing, 5 = certain. Rate the choice, not the explanation.>
"""
