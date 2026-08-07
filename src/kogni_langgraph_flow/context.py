"""Ambient per-turn identity -- a contextvar, not a parameter threaded
through every function call. enable_flow() starts a turn automatically
around each app.invoke()/app.ainvoke() call (see api.py); everything
traced anywhere below that (any node, any depth, any pydantic-ai agent
call) picks up the same trace_id without any code needing to pass it
along.

Two contextvars, not one: current_trace_id (which turn) and
current_span_id (which node/call is the "parent" of anything traced
while it's running) -- the latter is what makes a node that itself
triggers another traced call show up as real parent/child hierarchy
instead of a flat list.

set_span()/reset_span() (not just a `with` block) exist because
LangGraph fires its own node start/end as two SEPARATE callback
invocations (see instrumentation.py) -- there's no single Python scope
to wrap with a context manager there.

contextvars.ContextVar (not a plain global) so this is safe under
asyncio/concurrent turns too -- each task gets its own current trace_id/
span_id, unlike a plain module-level variable which would leak across
concurrent requests.
"""

import contextlib
import contextvars
import uuid
from typing import Iterator, Optional

from . import tracer  # tracer.py has no imports from this module, so no cycle

_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "kogni_langgraph_flow_current_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "kogni_langgraph_flow_current_span_id", default=None
)


def start_turn(trace_id: Optional[str] = None) -> str:
    """Call once at the top of one graph invocation. Prefer
    `with trace_turn():` when the turn's end is easy to mark
    structurally -- see below."""
    trace_id = trace_id or f"TRACE-{uuid.uuid4().hex[:8]}"
    _current_trace_id.set(trace_id)
    _current_span_id.set(None)
    tracer.start_trace(trace_id)
    return trace_id


def end_turn(trace_id: Optional[str] = None) -> None:
    """Marks the turn's wall-clock end -- what formatter.py's 'End-to-end
    turn' figure is measured against."""
    tracer.mark_turn_ended(trace_id or current_trace_id())


@contextlib.contextmanager
def trace_turn(trace_id: Optional[str] = None) -> Iterator[str]:
    """with trace_turn() as trace_id: ... -- start_turn() + end_turn(),
    as a context manager so the turn boundary can't be forgotten."""
    trace_id = start_turn(trace_id)
    try:
        yield trace_id
    finally:
        end_turn(trace_id)


def current_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def current_span_id() -> Optional[str]:
    return _current_span_id.get()


def set_span(span_id: Optional[str]) -> contextvars.Token:
    """Pair with reset_span(token). Use push_span() instead when start
    and end share one Python scope."""
    return _current_span_id.set(span_id)


def reset_span(token: contextvars.Token) -> None:
    _current_span_id.reset(token)


@contextlib.contextmanager
def push_span(span_id: str) -> Iterator[None]:
    """`with` form of set_span/reset_span, for a caller whose start and
    end DO share one Python scope (e.g. instrumentation.py's wrapped
    pydantic-ai call)."""
    token = set_span(span_id)
    try:
        yield
    finally:
        reset_span(token)
