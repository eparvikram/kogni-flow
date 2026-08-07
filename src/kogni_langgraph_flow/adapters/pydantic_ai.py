"""pydantic-ai's own usage shapes -- both the plain-attribute form
current versions use, and a callable `result.usage()` form some
versions/wrappers expose."""

from typing import Any, Optional

from ..models import TokenUsage


def pydantic_ai_adapter(result: Any) -> Optional[TokenUsage]:
    """result.usage.input_tokens / .output_tokens / .total_tokens --
    a plain attribute, not a method, in current pydantic-ai versions."""
    usage = getattr(result, "usage", None)
    if usage is None or callable(usage):
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=getattr(usage, "total_tokens", None), available=True,
    )


def pydantic_ai_callable_usage_adapter(result: Any) -> Optional[TokenUsage]:
    """result.usage() as a CALLABLE returning an object with
    request_tokens/response_tokens (some pydantic-ai versions/wrappers)
    or input_tokens/output_tokens/prompt_tokens/completion_tokens."""
    usage_attr = getattr(result, "usage", None)
    if not callable(usage_attr):
        return None
    try:
        usage = usage_attr()
    except TypeError:
        return None
    if usage is None:
        return None
    input_tokens = (
        getattr(usage, "request_tokens", None)
        or getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
    )
    output_tokens = (
        getattr(usage, "response_tokens", None)
        or getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
    )
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=getattr(usage, "total_tokens", None), available=True,
    )
