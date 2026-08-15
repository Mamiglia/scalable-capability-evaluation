from typing import Any
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import Sample


def strip_think(text: str) -> str:
    """
    Drop DeepSeek-style inline chain-of-thought from a text block.

    DeepSeek emits reasoning as `<think> ... </think>` then the answer. When the
    provider returns it as plain text (not a separate reasoning block) it reaches
    the judge verbatim. Keep only what follows the last `</think>`; if `<think>`
    is never closed (the model spent its whole budget reasoning) there is no
    answer, so drop everything from the first `<think>` on.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip()
    if "<think>" in text:
        return text.split("<think>", 1)[0].rstrip()
    return text


def format_content(content: Any, include_reasoning: bool = True) -> str:
    """
    Format content blocks (text, reasoning, image, audio, video, etc.) into readable string.
    """
    if isinstance(content, str):
        return content if include_reasoning else strip_think(content)
    if not isinstance(content, list):
        return str(content)

    texts = []
    for block in content:
        b_type = getattr(block, "type", None)
        if b_type == "text":
            text_val = getattr(block, "text", "")
            text_val = str(text_val) if text_val is not None else ""
            texts.append(text_val if include_reasoning else strip_think(text_val))
        elif b_type == "reasoning":
            if include_reasoning:
                text_val = getattr(block, "text", "")
                texts.append(f"<THOUGHT_START>\n {text_val}\n<THOUGHT_END>")
        elif b_type == "image":
            texts.append("[Image]") # Right now we don't have a way to display media in the judge model, so we just indicate that an image was present. TODO
        elif b_type == "audio":
            texts.append("[Audio]")
        elif b_type == "video":
            texts.append("[Video]")
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
        # Filter out system and initial user prompt
        turns = [m for m in messages if getattr(m, "role", None) not in ("system", "user")]
        if turns:
            return format_message(turns, include_reasoning=include_reasoning)
        return format_message(messages, include_reasoning=include_reasoning)

    # Fallback to output.choices[0].message
    if hasattr(sample, "output") and sample.output and getattr(sample.output, "choices", None):
        return format_message(sample.output.choices[0].message, include_reasoning=include_reasoning)

    return ""


def extract_score(sample: Sample) -> bool:
    """
    Extract the score from the sample's output.
    """
    score = sample.score
    return score is not None and score.value == "C"


def first_correct_by_id(log) -> dict:
    """
    id -> that item's correct samples, earliest epoch first.

    The epoch order build_dataset judges: it takes [0] for each side of a pair,
    and [0] and [1] for the two sides of a self-pair.
    """
    grouped: dict = {}
    for s in log.samples:
        if extract_score(s):
            grouped.setdefault(s.id, []).append(s)
    for group in grouped.values():
        group.sort(key=lambda s: getattr(s, "epoch", 0))
    return grouped
