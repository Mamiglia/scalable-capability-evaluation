from typing import Any
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import Sample

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

    Non-text modalities become a placeholder: the judge only ever sees text.
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
    return any(score.value == "C" for score in scores.values())
