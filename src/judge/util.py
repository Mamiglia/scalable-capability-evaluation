from typing import Any
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import Sample


def format_content(content: Any) -> str:
    """
    Format content blocks (text, reasoning, image, audio, video, etc.) into readable string.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    texts = []
    for block in content:
        b_type = getattr(block, "type", None)
        if b_type == "text":
            text_val = getattr(block, "text", "")
            texts.append(str(text_val) if text_val is not None else "")
        elif b_type == "reasoning":
            text_val = getattr(block, "text", "")
            texts.append(f"<THOUGHT_START>\n {text_val}\n<THOUGHT_END>")
        elif b_type == "image":
            texts.append("[Image]") # Right now we don't have a way to display media in the judge model, so we just indicate that an image was present. TODO
        elif b_type == "audio":
            texts.append("[Audio]")
        elif b_type == "video":
            texts.append("[Video]")
        elif hasattr(block, "text") and isinstance(getattr(block, "text"), str):
            texts.append(block.text)
        else:
            texts.append(f"[{b_type}]" if b_type else str(block))
    return "\n".join(texts)


def format_single_message(msg: ChatMessage | str | Any) -> str:
    """
    Format a single message (assistant, user, system, or tool).
    """
    if isinstance(msg, str):
        return msg

    parts = []
    content_str = format_content(getattr(msg, "content", ""))

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


def format_message(msg: ChatMessage | list[ChatMessage] | str | Any) -> str:
    """
    Format the message object or list of messages into a readable string.
    Supports single messages, message sequences (transcripts), tool calls, and tool outputs.
    """
    if not isinstance(msg, list):
        return format_single_message(msg)
    
    formatted_msgs = []
    for m in msg:
        role = getattr(m, "role", "message").capitalize()
        fmt = format_single_message(m)
        formatted_msgs.append(f"[{role}]: {fmt}")
    return "\n\n".join(formatted_msgs)


def extract_transcript(sample: Sample) -> str:
    """
    Extract the complete response transcript from a sample,
    supporting both single message outputs and multi-turn agent histories with tool calls.
    """
    messages = getattr(sample, "messages", None)
    if messages:
        # Filter out system and initial user prompt
        turns = [m for m in messages if getattr(m, "role", None) not in ("system", "user")]
        if turns:
            return format_message(turns)
        return format_message(messages)

    # Fallback to output.choices[0].message
    if hasattr(sample, "output") and sample.output and getattr(sample.output, "choices", None):
        return format_message(sample.output.choices[0].message)

    return ""


def extract_score(sample: Sample) -> bool:
    """
    Extract the score from the sample's output.
    """
    score = sample.score
    return score.value == "C"