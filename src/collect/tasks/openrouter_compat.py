"""OpenRouter provider fixes for agentic collection (import for side effect; collect/cli.py does).

1. Anthropic prompt caching. inspect's OpenRouter provider marks the last system block, the last
   tool definition and a message-level breakpoint -- but only on list-form content. In an agent
   loop the last message is a string-content ``tool`` message, so the rolling breakpoint never
   lands and only the static prefix (~2.3K tokens) is ever read back: 6-8% of prompt tokens on
   Terminal-Bench (src/challenges/frontier/README.md). OpenRouter honours
   ``cache_control`` on a tool message whose content is a list of text blocks, and the whole
   prefix is read back on the next turn. So convert the last message's string content to list
   form and mark its last block. At most 4 breakpoints (Anthropic's limit): system, tools,
   inspect's rolling one, this one.

2. Flaky upstream 400s. OpenRouter relays an upstream failure as HTTP 400 ``Provider returned
   error`` with ``metadata.raw: 'ERROR'`` (stealth/union-alpha: ~7% of calls, at
   random). inspect treats 400 as final, so the sample dies. Retry that exact shape as transient;
   a real 400 (context overflow, bad schema) carries a descriptive raw message and stays final.
"""

from typing import Any

from inspect_ai.model._providers import openrouter
from inspect_ai.model._model import RetryDecision

_inspect_markers = openrouter._add_anthropic_cache_markers
_inspect_should_retry = openrouter.OpenRouterAPI.should_retry


def _add_anthropic_cache_markers(request: dict[str, Any]) -> None:
    _inspect_markers(request)
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if isinstance(content, str) and content:
        last["content"] = [{"type": "text", "text": content, "cache_control": openrouter._ephemeral()}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = openrouter._ephemeral()


def _should_retry(self: Any, ex: BaseException) -> bool | RetryDecision:
    decision = _inspect_should_retry(self, ex)
    message = str(ex)
    if "Provider returned error" in message and "'raw': 'ERROR'" in message:
        return RetryDecision.transient()
    return decision


openrouter._add_anthropic_cache_markers = _add_anthropic_cache_markers
openrouter.OpenRouterAPI.should_retry = _should_retry
