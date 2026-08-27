from typing import Any
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import Sample
from inspect_ai.scorer import value_to_float

from configs.logs import logger

# CORRECT/"C" -> 1, INCORRECT/"I" -> 0, numbers -> themselves, unparseable -> 0.
_score_to_float = value_to_float()

# Content the judge cannot read: shown as a marker so the transcript stays coherent.
PLACEHOLDER_BLOCKS = {
    "image": "[Image]",
    "audio": "[Audio]",
    "video": "[Video]",
    "document": "[Document]",
    "data": "[Data]",
    "tool_use": "[Tool Use]",
}


def strip_think(text: str) -> str:
    """Drop inline `<think>…</think>` chain-of-thought from a text block.

    A normalized log never has any: ``scripts/audit/logs/normalize_reasoning.py`` moves it
    into a real reasoning block, where ``format_reasoning`` decides whether it is shown.
    This is the fallback for a log that missed that migration -- the CoT is dropped rather
    than promoted, so it can never leak into the answer text of the no-reasoning condition.
    Dropping it also leaves the transcript with no reasoning mark, so ``build_dataset``
    refuses to pair the model against one that has CoT, rather than judging CoT vs nothing.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip()
    if "<think>" in text:
        return text.split("<think>", 1)[0].rstrip()
    return text


def format_reasoning(block: Any) -> str:
    """Render one reasoning block, or nothing when it holds no raw chain-of-thought.

    Only genuine CoT ever reaches the judge. ``redacted`` marks the blocks whose real CoT
    never left the provider: ``reasoning`` is then an encrypted blob (o3, gpt-5 arrive as
    ``gAAAAAB...``) and ``summary`` is a short paraphrase written by the provider, 3-5x
    shorter than the CoT it stands for and in a different register.

    Neither is shown. A blob is not evidence, and a summary is not comparable to a CoT --
    judging one against the other measures the provider's disclosure policy, not the model.
    Dropping it here is what makes ``build_dataset`` refuse such a pair: the transcript
    carries no reasoning mark, so the one-side-only guard catches it.

    Not read via ``ContentReasoning.text``: that property silently swaps in the summary on
    redacted blocks and wraps everything in ``<think>``.
    """
    if getattr(block, "redacted", False):
        return ""
    reasoning = getattr(block, "reasoning", "") or ""
    # ponytail: a provider that returns a summary and nothing else arrives indistinguishable
    # from raw CoT (OpenRouter collapses it into `reasoning`); nothing in the log to tell.
    return f"<THOUGHT_START>\n {reasoning}\n<THOUGHT_END>" if reasoning.strip() else ""


def format_text(text: str) -> str:
    """Render one text block. Inline `<think>` means the log was never normalized."""
    if "<think>" not in text:
        return text
    logger.warning(
        "inline <think> in a text block (%r...): log not normalized, dropping the CoT -- "
        "run scripts/audit/logs/normalize_reasoning.py", text[:60],
    )
    return strip_think(text)


def format_content(content: Any, include_reasoning: bool = True) -> str:
    """
    Format content blocks (text, reasoning, image, audio, video, etc.) into readable string.

    Non-text modalities become a placeholder: the judge only ever sees text.
    """
    if isinstance(content, str):
        return format_text(content)
    if not isinstance(content, list):
        return str(content)

    texts = []
    for block in content:
        b_type = getattr(block, "type", None)
        if b_type == "text":
            text_val = getattr(block, "text", "")
            texts.append(format_text(str(text_val) if text_val is not None else ""))
        elif b_type == "reasoning":
            if include_reasoning and (rendered := format_reasoning(block)):
                texts.append(rendered)
        elif b_type in PLACEHOLDER_BLOCKS:
            # No way to show media to the judge, so just record that it was there. TODO
            texts.append(PLACEHOLDER_BLOCKS[b_type])
        else:
            raise TypeError(f"Unsupported content block type: {b_type!r}")
    return "\n".join(texts)


def format_single_message(msg: ChatMessage | str | Any, include_reasoning: bool = True) -> str:
    """
    Format a single message (assistant, user, system, or tool).
    """
    if isinstance(msg, str):
        return msg

    parts = []
    content_str = format_content(getattr(msg, "content", ""), include_reasoning=include_reasoning)

    # Handle tool output role
    if getattr(msg, "role", None) == "tool":
        func = getattr(msg, "function", "tool")
        err = getattr(msg, "error", None)
        err_msg = getattr(err, "message", str(err)) if err else ""
        err_str = f" (Error: {err_msg})" if err else ""
        return f"[Tool Output ({func}){err_str}]: {content_str}"

    if content_str:
        parts.append(content_str)

    # Handle tool calls on assistant messages
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        tc_strs = []
        for call in tool_calls:
            func = getattr(call, "function", "tool")
            args = getattr(call, "arguments", {})
            tc_strs.append(f"[Tool Call: {func}({args})]")
        parts.append("\n".join(tc_strs))

    return "\n".join(parts)


def format_message(msg: ChatMessage | list[ChatMessage] | str | Any, include_reasoning: bool = True) -> str:
    """
    Format the message object or list of messages into a readable string.
    Supports single messages, message sequences (transcripts), tool calls, and tool outputs.
    """
    if not isinstance(msg, list):
        return format_single_message(msg, include_reasoning=include_reasoning)

    formatted_msgs = []
    for m in msg:
        role = getattr(m, "role", "message").capitalize()
        fmt = format_single_message(m, include_reasoning=include_reasoning)
        formatted_msgs.append(f"[{role}]: {fmt}")
    return "\n\n".join(formatted_msgs)


def extract_transcript(sample: Sample, include_reasoning: bool = True) -> str:
    """
    Extract the complete response transcript from a sample,
    supporting both single message outputs and multi-turn agent histories with tool calls.
    """
    messages = getattr(sample, "messages", None)
    if messages:
        # Filter out system messages and the initial user prompt; later user turns are
        # scaffold feedback (retry prompts etc.) and stay in the transcript.
        turns, seen_user = [], False
        for m in messages:
            role = getattr(m, "role", None)
            if role == "system":
                continue
            if role == "user" and not seen_user:
                seen_user = True
                continue
            turns.append(m)
        if turns:
            return format_message(turns, include_reasoning=include_reasoning)
        return format_message(messages, include_reasoning=include_reasoning)

    # Fallback to output.choices[0].message
    if hasattr(sample, "output") and sample.output and getattr(sample.output, "choices", None):
        return format_message(sample.output.choices[0].message, include_reasoning=include_reasoning)

    return ""


def _is_correct(value: Any) -> bool:
    """Correct if the scorer's value is 1.0 (full score)

    Covers every scorer we collect: choice/match store the string "C"; SWE-bench stores a
    float 1.0/0.0 from its in-container test run; HLE nests the label under "score" in a
    dict alongside a confidence.
    """
    if isinstance(value, dict):
        value = value.get("score", 0)
    return _score_to_float(value) >= 1.0


def extract_score(sample: Sample) -> bool:
    """
    Whether the sample was scored correct.

    Uses the first scorer, which is the only one for every benchmark we collect; more
    than one would need the judge configs to say which counts.
    """
    scores = getattr(sample, "scores", None) or {}
    if len(scores) > 1:
        raise ValueError(
            f"sample {getattr(sample, 'id', '?')} has several scores ({sorted(scores)}); "
            f"cannot tell which one decides correctness"
        )
    return any(_is_correct(score.value) for score in scores.values())
