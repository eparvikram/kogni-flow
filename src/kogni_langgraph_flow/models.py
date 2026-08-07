"""Canonical record shapes for this package -- kept separate from
tracer.py's storage, adapters/'s extraction, and formatter.py's display,
so none of them reach into each other's internals.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenUsage:
    """What adapters/ normalizes every framework's usage object into.
    available=False (not zero) means "this SDK's usage shape wasn't
    recognized" -- formatter.py renders that as '-', never a fabricated 0."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    available: bool = False


@dataclass
class AgentTrace:
    """One traced unit -- a LangGraph node, or (best-effort) a
    pydantic-ai agent call. Deliberately minimal -- name, tokens,
    latency, status. No prompts, no full outputs, no business payloads:
    this is a developer debugging aid, not an audit trail."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    sequence: int
    agent_name: str
    model_name: Optional[str] = None  # None for a plain LangGraph node -- only agent calls have a model
    # Captured only for a pydantic-ai agent call whose prompt/output are
    # plain strings (see instrumentation.py's _extract_prompt_text/
    # _extract_output_text) -- None for a plain LangGraph node, or when
    # the prompt/output isn't a simple string (message_history-based
    # calls, structured output_type, multi-modal content). Kept in-memory
    # only, same as the rest of this record (see tracer.py's storage
    # model) -- nothing here is persisted to disk.
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    status: str = "completed"
    error_type: Optional[str] = None
    started_at: float = field(default_factory=time.time)


@dataclass
class TurnRecord:
    trace_id: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    executions: List[AgentTrace] = field(default_factory=list)
