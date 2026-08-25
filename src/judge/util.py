import hashlib
import json
from typing import Any
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import Sample
from inspect_ai.scorer import value_to_float

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


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _message_subparts(msg: ChatMessage | str | Any, include_reasoning: bool) -> list[dict[str, Any]]:
    """Rendered pieces of one message, retaining tool metadata without hidden reasoning."""
    if isinstance(msg, str):
        return [{"kind": "content", "content": msg, "call_id": None,
                 "function": None, "raw_payload": None}]
    content_str = format_content(getattr(msg, "content", ""), include_reasoning=include_reasoning)
    if getattr(msg, "role", None) == "tool":
        func = getattr(msg, "function", "tool")
        err = getattr(msg, "error", None)
        err_msg = getattr(err, "message", str(err)) if err else ""
        err_str = f" (Error: {err_msg})" if err else ""
        return [{"kind": "tool_output", "content": f"[Tool Output ({func}){err_str}]: {content_str}",
                 "call_id": getattr(msg, "tool_call_id", None), "function": func,
                 "raw_payload": _jsonable(getattr(msg, "content", ""))}]

    parts = []
    if content_str:
        parts.append({"kind": "content", "content": content_str, "call_id": None,
                      "function": None, "raw_payload": None})
    for call in getattr(msg, "tool_calls", None) or ():
        func, args = getattr(call, "function", "tool"), getattr(call, "arguments", {})
        parts.append({"kind": "tool_call", "content": f"[Tool Call: {func}({args})]",
                      "call_id": getattr(call, "id", None), "function": func,
                      "raw_payload": _jsonable(args)})
    return parts


def format_single_message(msg: ChatMessage | str | Any, include_reasoning: bool = True) -> str:
    """Format a single assistant, user, system, or tool message."""
    return "\n".join(part["content"] for part in _message_subparts(msg, include_reasoning))


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


def _transcript_messages(sample: Sample) -> tuple[list[Any], bool]:
    """Messages visible after the prompt, and whether one output is rendered bare."""
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
            return turns, False
        return list(messages), False

    # Fallback to output.choices[0].message
    if hasattr(sample, "output") and sample.output and getattr(sample.output, "choices", None):
        return [sample.output.choices[0].message], True

    return [], False


def extract_transcript_parts(sample: Sample, include_reasoning: bool = True) -> list[dict[str, Any]]:
    """Structured visible messages with exact content spans in the rendered transcript."""
    messages, bare = _transcript_messages(sample)
    parts, cursor = [], 0
    for message in messages:
        role = str(getattr(message, "role", "assistant" if bare else "message")).lower()
        subparts = _message_subparts(message, include_reasoning)
        content = "\n".join(subpart["content"] for subpart in subparts)
        prefix = "" if bare else f"[{role.capitalize()}]: "
        start = cursor + len(prefix)
        offset = start
        for subpart in subparts:
            subpart["start"], subpart["end"] = offset, offset + len(subpart["content"])
            offset = subpart["end"] + 1
        parts.append({"role": role, "content": content, "start": start,
                      "end": start + len(content), "bare": bare, "subparts": subparts})
        cursor += len(prefix) + len(content) + 2
    return parts


def render_transcript_parts(parts: list[dict[str, Any]]) -> str:
    """Render parts returned by :func:`extract_transcript_parts`."""
    return "\n\n".join(
        part["content"] if part["bare"] else f"[{part['role'].capitalize()}]: {part['content']}"
        for part in parts
    )


def transcript_content_hash(question: str, transcript: str, parts: list[dict[str, Any]]) -> str:
    """Stable identity of every byte that controls scoring or assistant exposure."""
    payload = json.dumps({"question": question, "transcript": transcript, "parts": parts},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_transcript(sample: Sample, include_reasoning: bool = True) -> str:
    """Extract exactly the transcript visible to the judge."""
    return render_transcript_parts(extract_transcript_parts(sample, include_reasoning))


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
    if not scores:
        return False
    value = next(iter(scores.values())).value
    if isinstance(value, dict):
        value = value.get("score", 0)
    return value_to_float()(value) >= 1.0
