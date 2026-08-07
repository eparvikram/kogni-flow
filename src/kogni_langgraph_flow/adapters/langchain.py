"""LangChain-style response objects: response_metadata/additional_kwargs
dicts carrying a 'token_usage' or 'usage' dict."""

from typing import Any, Optional

from ..models import TokenUsage


def langchain_adapter(result: Any) -> Optional[TokenUsage]:
    for attr in ("response_metadata", "additional_kwargs"):
        meta = getattr(result, attr, None)
        if not isinstance(meta, dict):
            continue
        usage = meta.get("token_usage") or meta.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if input_tokens is None and output_tokens is None:
            continue
        return TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens"), available=True,
        )
    return None
