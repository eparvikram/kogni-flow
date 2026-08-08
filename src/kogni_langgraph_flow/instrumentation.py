"""Everything that actually captures a trace without the caller writing
any tracing code: FlowCallbackHandler (LangGraph node start/end) and
patch_pydantic_ai_agent (best-effort per-agent token/latency). Both feed
the same trace_id/span hierarchy (see context.py), so a node that
internally calls an agent shows up as real parent/child nesting.

FlowCallbackHandler
--------------------
Turns LangGraph's own node start/end events into trace records -- zero
code changes inside any node function. Verified directly (not assumed):
invoking a compiled LangGraph app with this handler in
config["callbacks"] fires on_chain_start/on_chain_end for each node,
with the node's name available at metadata["langgraph_node"]. There's
also an outer chain_start/chain_end pair for the whole-graph invocation
itself (metadata has no "langgraph_node" key) -- skipped here so the
graph wrapper doesn't get recorded as if it were a node. Depends on
langchain_core (LangGraph's own callback base class) -- a real,
unavoidable dependency for this piece; a project using LangGraph already
has langchain_core installed regardless, since LangGraph is built on it.

patch_pydantic_ai_agent
-------------------------
Patches Agent.run_sync ONCE, at the class level (idempotent -- a second
enable_flow() call is a no-op here) -- so every Agent instance, already
constructed or created later, gets traced without the caller wrapping
each call site. Purely optional: if pydantic_ai isn't installed, this
just returns False and FlowCallbackHandler still gives full node-level
tracing -- an app using a different LLM library loses only the per-agent
token/latency detail, not the whole feature.

Agent naming: pydantic-ai's own Agent(infer_name=True) (the default)
normally sets agent.name to the variable it was assigned to, the first
time it's run -- BUT its own inference (Agent._infer_name, see
pydantic_ai/agent/__init__.py) walks up exactly ONE stack frame from
inside run_sync to find that variable, and since our wrapper sits
between the real caller and the original run_sync, that one frame up
lands on OUR wrapper's own `self` parameter instead of the caller's
actual variable -- verified directly (not assumed): without the fix
below, every traced agent showed up named literally "self" in the
trace. _infer_caller_name() replicates the same frame-walk one level
higher (our wrapper's caller, not pydantic-ai's idea of the caller) and
sets agent.name BEFORE calling through, so pydantic-ai's own inference
(which only runs `if self.name is None`) never fires and never
overwrites it.
"""

import inspect
import time
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from . import context as flow_context
from . import tracer as flow_tracer
from .adapters import extract_usage
from .models import AgentTrace


class FlowCallbackHandler(BaseCallbackHandler):
    """One instance per enable_flow()-wrapped app. Instance state
    (_starts) tracks in-flight node runs by LangChain's own run_id, so
    concurrent/nested node executions (parallel branches, subgraphs)
    don't cross-contaminate each other's timing or span parenting."""

    def __init__(self) -> None:
        self._starts: Dict[UUID, Dict[str, Any]] = {}

    def on_chain_start(
        self, serialized: Optional[dict], inputs: Any, *, run_id: UUID,
        parent_run_id: Optional[UUID] = None, tags: Optional[list] = None,
        metadata: Optional[dict] = None, **kwargs: Any,
    ) -> None:
        node_name = (metadata or {}).get("langgraph_node")
        if not node_name:
            return  # the whole-graph wrapper, not an individual node

        trace_id = flow_context.current_trace_id() or flow_context.start_turn()
        parent_span_id = flow_context.current_span_id()
        # Full run_id, not truncated -- verified directly that truncating
        # to 8 hex chars was NOT safe: two different nodes' run_ids
        # collided on their first 8 characters in a real test (LangGraph's
        # run_ids for nodes within one invocation aren't independently
        # random enough for a short prefix to be reliably unique), which
        # made two distinct node spans look like the same parent.
        span_id = f"SPAN-{run_id.hex}"
        sequence = flow_tracer.next_sequence(trace_id)
        token = flow_context.set_span(span_id)

        self._starts[run_id] = {
            "trace_id": trace_id, "span_id": span_id, "parent_span_id": parent_span_id,
            "sequence": sequence, "node_name": node_name, "token": token,
            "start": time.monotonic(),
        }

    def _finish(self, run_id: UUID, status: str, error: Optional[BaseException] = None) -> None:
        entry = self._starts.pop(run_id, None)
        if entry is None:
            return
        flow_context.reset_span(entry["token"])
        latency_ms = (time.monotonic() - entry["start"]) * 1000
        flow_tracer.record(entry["trace_id"], AgentTrace(
            trace_id=entry["trace_id"], span_id=entry["span_id"], parent_span_id=entry["parent_span_id"],
            sequence=entry["sequence"], agent_name=entry["node_name"], latency_ms=latency_ms,
            status=status, error_type=type(error).__name__ if error else None,
        ))

    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._finish(run_id, "completed")

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._finish(run_id, "failed", error)


_patched = False


def _extract_prompt_text(args: Any, kwargs: Dict[str, Any]) -> Optional[str]:
    """run_sync's first positional arg (or its user_prompt kwarg) is the
    prompt for a plain string call -- None for anything else (a
    message_history-based call with no new prompt, multi-modal content,
    etc.), rather than guessing at a stringified representation."""
    candidate = args[0] if args else kwargs.get("user_prompt")
    return candidate if isinstance(candidate, str) else None


def _extract_output_text(result: Any) -> Optional[str]:
    """result.output is a plain string when the agent has no output_type
    set; a structured output_type gives back a Pydantic model (the
    common case) or some other Python object instead. A Pydantic model
    is serialized via model_dump_json() -- a real, readable
    representation of the actual output, not a Python repr; anything
    else non-string falls back to str(). Both get truncated for display
    the same way plain-string output already is (see formatter.py) --
    None only for a genuinely absent output."""
    output = getattr(result, "output", None)
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


def _infer_caller_name(target: Any) -> Optional[str]:
    """Same frame-walk as pydantic-ai's own Agent._infer_name, but from
    THIS function's caller (traced_run_sync) up to ITS caller -- i.e.
    the code that actually wrote `some_agent.run_sync(...)`."""
    this_frame = inspect.currentframe()
    if this_frame is None or this_frame.f_back is None:
        return None
    caller_frame = this_frame.f_back.f_back  # skip traced_run_sync's own frame too
    if caller_frame is None:
        return None
    for name, item in caller_frame.f_locals.items():
        if item is target:
            return name
    if caller_frame.f_locals is not caller_frame.f_globals:
        for name, item in caller_frame.f_globals.items():
            if item is target:
                return name
    return None


def patch_pydantic_ai_agent() -> bool:
    global _patched
    if _patched:
        return True
    try:
        from pydantic_ai import Agent
    except ImportError:
        return False

    original_run_sync = Agent.run_sync

    def traced_run_sync(self: "Agent", *args: Any, **kwargs: Any) -> Any:
        if self.name is None:
            inferred = _infer_caller_name(self)
            if inferred:
                self.name = inferred  # set BEFORE calling through, so pydantic-ai's
                # own infer_name (which only fires `if self.name is None`) never
                # overwrites this with its own (wrong, one-frame-too-shallow) guess.
        trace_id = flow_context.current_trace_id() or flow_context.start_turn()
        parent_span_id = flow_context.current_span_id()
        sequence = flow_tracer.next_sequence(trace_id)
        span_id = f"SPAN-{id(self):x}-{sequence}"
        # getattr, not self.model.model_name directly -- not every Model
        # subclass is guaranteed to expose model_name, and a missing name
        # should show as "-" downstream (formatter.py), never crash the
        # whole trace.
        model_name = getattr(self.model, "model_name", None)
        input_text = _extract_prompt_text(args, kwargs)
        start = time.monotonic()
        try:
            with flow_context.push_span(span_id):
                result = original_run_sync(self, *args, **kwargs)
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            agent_name = self.name or f"pydantic_ai_agent_{id(self):x}"
            flow_tracer.record(trace_id, AgentTrace(
                trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id, sequence=sequence,
                agent_name=agent_name, model_name=model_name, input_text=input_text, latency_ms=latency_ms,
                status="failed", error_type=type(exc).__name__,
            ))
            raise

        latency_ms = (time.monotonic() - start) * 1000
        agent_name = self.name or f"pydantic_ai_agent_{id(self):x}"
        usage = extract_usage(result)
        flow_tracer.record(trace_id, AgentTrace(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id, sequence=sequence,
            agent_name=agent_name, model_name=model_name,
            input_text=input_text, output_text=_extract_output_text(result),
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        ))
        return result

    Agent.run_sync = traced_run_sync
    _patched = True
    return True
