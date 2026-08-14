"""Judge prompt constants, copied verbatim from src so pod scripts stay
self-contained. test_prompts_match_src guards against drift."""

SYSTEM_PROMPT_FULL = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript 1 and Transcript 2, both coming to the correct solution.

For multiple choice questions, the options were shuffled independently for each model, so an option letter in Transcript 1 does not refer to the same option as that letter in Transcript 2. Each transcript is preceded by the options as that model saw them. Read each letter against that transcript's own option list.

One of the two models is more capable than the other, and your task is to identify which is the more capable model (1 or 2). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response was produced by the more capable model and why. 2-4 sentences.>
ANSWER: <1 or 2, indicating which response was produced by the more capable model.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>
"""

SYSTEM_PROMPT_SHORT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript 1 and Transcript 2, both coming to the correct solution.

For multiple choice questions, the options were shuffled independently for each model, so an option letter in Transcript 1 does not refer to the same option as that letter in Transcript 2. Each transcript is preceded by the options as that model saw them. Read each letter against that transcript's own option list.

One of the two models is more capable than the other, and your task is to identify which is the more capable model (1 or 2). You MUST follow this reply format:

ANSWER: <1 or 2, indicating which response was produced by the more capable model.>
"""

USER_PROMPT = """Input question:
---
{question}
---

{options_1}Transcript 1:
---
{transcript_1}
---

{options_2}Transcript 2:
---
{transcript_2}
---

Which of the two transcripts is from the more capable model? Answer in the instructed format.
"""

OPTIONS_BLOCK = """Multiple choice options as shown to Model {n}:
{options}

"""


def format_options(choices, n):
    """Render an option list for slot n. Empty string if there are no choices."""
    if not choices:
        return ""
    options = "\n".join(f"{chr(ord('A') + i)}) {c.strip()}" for i, c in enumerate(choices))
    return OPTIONS_BLOCK.format(n=n, options=options)


def user_text(question, options_1, options_2, text_1, text_2):
    return USER_PROMPT.format(question=question, options_1=options_1,
                              options_2=options_2, transcript_1=text_1,
                              transcript_2=text_2)
