"""Normalizes different SDKs' token-usage shapes into one TokenUsage, so
instrumentation.py's auto-traced agent calls work across frameworks
without tracer.py caring which SDK produced the result. Tried in order,
custom-registered ones first; the first adapter that recognizes the
shape wins.

Each SDK's shape lives in its own module (pydantic_ai.py, openai.py,
langchain.py) so adding support for another framework means adding one
file here, not editing a growing single one.
"""

from typing import Any, Callable, List, Optional

from ..models import TokenUsage
from .langchain import langchain_adapter
from .openai import openai_adapter
from .pydantic_ai import pydantic_ai_adapter, pydantic_ai_callable_usage_adapter

UsageAdapter = Callable[[Any], Optional[TokenUsage]]

_DEFAULT_ADAPTERS: List[UsageAdapter] = [
    pydantic_ai_adapter,
    openai_adapter,
    pydantic_ai_callable_usage_adapter,
    langchain_adapter,
]

_custom_adapters: List[UsageAdapter] = []


def register_adapter(adapter: UsageAdapter) -> None:
    """Add a custom adapter, tried before the built-in ones -- for a
    result shape none of the above cover. adapter(result) -> TokenUsage
    or None (None means 'not my shape, try the next one')."""
    _custom_adapters.insert(0, adapter)


def extract_usage(result: Any) -> TokenUsage:
    for adapter in _custom_adapters + _DEFAULT_ADAPTERS:
        try:
            usage = adapter(result)
        except Exception:
            continue
        if usage is not None:
            return usage
    return TokenUsage(available=False)


__all__ = ["register_adapter", "extract_usage", "UsageAdapter"]
