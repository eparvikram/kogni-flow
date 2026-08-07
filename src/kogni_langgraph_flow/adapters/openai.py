"""The raw OpenAI/Azure OpenAI SDK response shape."""

from typing import Any, Optional

from ..models import TokenUsage


def openai_adapter(result: Any) -> Optional[TokenUsage]:
    """result.usage.prompt_tokens / .completion_tokens / .total_tokens."""
    usage = getattr(result, "usage", None)
    if usage is None or callable(usage):
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if input_tokens is None and output_tokens is None:
        return None
    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=getattr(usage, "total_tokens", None), available=True,
    )
