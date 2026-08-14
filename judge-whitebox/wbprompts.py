"""Verdict prompt constants and rendering, copied from fine-tuning pipeline and drift-guarded by tests."""

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

ANSWER_TOKENS = ("1", "2")

# Mirrors SYSTEM_PROMPT_SHORT with the ask inverted: identify the LESS capable
# model instead of the more capable one. Written out in full (not derived by
# .replace()) so it's explicit and independently readable.
SYSTEM_PROMPT_SHORT_WEAKER = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript 1 and Transcript 2, both coming to the correct solution.

For multiple choice questions, the options were shuffled independently for each model, so an option letter in Transcript 1 does not refer to the same option as that letter in Transcript 2. Each transcript is preceded by the options as that model saw them. Read each letter against that transcript's own option list.

One of the two models is more capable than the other, and your task is to identify which is the LESS capable model (1 or 2). You MUST follow this reply format:

ANSWER: <1 or 2, indicating which response was produced by the less capable model.>
"""

# Mirrors USER_PROMPT with the closing line inverted to match
# SYSTEM_PROMPT_SHORT_WEAKER (an inverted system prompt with the original
# closing line would be contradictory).
USER_PROMPT_WEAKER = """Input question:
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

Which of the two transcripts is from the less capable model? Answer in the instructed format.
"""


def format_options(choices, n):
    """Render an option list for slot n. Empty string if there are no choices."""
    if not choices:
        return ""
    options = "\n".join(f"{chr(ord('A') + i)}) {c.strip()}" for i, c in enumerate(choices))
    return OPTIONS_BLOCK.format(n=n, options=options)


def user_text(question, options_1, options_2, text_1, text_2, invert=False):
    template = USER_PROMPT_WEAKER if invert else USER_PROMPT
    return template.format(question=question, options_1=options_1,
                              options_2=options_2, transcript_1=text_1,
                              transcript_2=text_2)


def render_verdict_prompt(row, tokenizer, invert=False):
    """Render a verdict prompt as a chat-templated string ending with ANSWER: teacher-forced.

    invert=True swaps in SYSTEM_PROMPT_SHORT_WEAKER + USER_PROMPT_WEAKER, asking
    for the LESS capable model instead. Both surfaces invert together.
    """
    system = SYSTEM_PROMPT_SHORT_WEAKER if invert else SYSTEM_PROMPT_SHORT
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user_text(
                row["question"], row["options_1"], row["options_2"], row["text_1"], row["text_2"],
                invert=invert)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) + "ANSWER: "
