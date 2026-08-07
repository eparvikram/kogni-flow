"""Renders one TurnRecord as a plain-text developer view: a flow
(sequential '->' chain, or an indented tree if any node/agent call
actually nested another traced call inside it), then a token/latency/
status table, then totals -- including BOTH the summed per-call latency
and the actual wall-clock turn latency, since they diverge whenever
work runs concurrently.
"""

from typing import List, Optional

from .context import current_trace_id
from .models import AgentTrace, TurnRecord
from .tracer import get_trace


def _format_flow(executions: List[AgentTrace]) -> List[str]:
    by_parent: dict = {}
    for e in executions:
        by_parent.setdefault(e.parent_span_id, []).append(e)

    has_nesting = any(e.parent_span_id is not None for e in executions)
    if not has_nesting:
        return [" -> ".join(e.agent_name for e in executions)]

    lines: List[str] = []

    def _render(parent_span_id: Optional[str], depth: int) -> None:
        for e in by_parent.get(parent_span_id, []):
            # Plain ASCII, not box-drawing characters -- a caller's
            # stdout isn't guaranteed to be UTF-8 (e.g. Windows' legacy
            # cp1252 console default raises UnicodeEncodeError on
            # unicode tree characters).
            prefix = "" if depth == 0 else ("  " * (depth - 1)) + "`-- "
            lines.append(f"{prefix}{e.agent_name}")
            _render(e.span_id, depth + 1)

    _render(None, 0)
    return lines


def _fmt_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "-"


def format_flow(trace_id: Optional[str] = None) -> str:
    trace_id = trace_id or current_trace_id()
    if not trace_id:
        return "No trace_id given and no current turn -- call enable_flow(app) and invoke it first."

    turn: Optional[TurnRecord] = get_trace(trace_id)
    if not turn or not turn.executions:
        return f"No node/agent executions recorded for trace_id={trace_id!r}."

    executions = sorted(turn.executions, key=lambda e: e.sequence)

    lines = [f"FLOW {trace_id}", ""]
    lines.append("Flow")
    lines.extend(_format_flow(executions))
    lines.append("")

    name_width = max(len("Node/Agent"), max(len(e.agent_name) for e in executions))
    model_width = max(len("Model"), max(len(e.model_name or "-") for e in executions))
    header = (
        f"{'Node/Agent':<{name_width}}  {'Model':<{model_width}}  "
        f"{'Input':>8}  {'Output':>8}  {'Latency':>10}  {'Status':>7}"
    )
    separator = "-" * len(header)
    lines.append(header)
    lines.append(separator)

    total_in, total_out, top_level_latency = 0, 0, 0.0
    for e in executions:
        status = "OK" if e.status == "completed" else "ERROR"
        model_str = e.model_name or "-"
        in_str = _fmt_int(e.input_tokens)
        out_str = _fmt_int(e.output_tokens)
        latency_str = f"{e.latency_ms:.0f}ms" if e.latency_ms is not None else "-"
        lines.append(
            f"{e.agent_name:<{name_width}}  {model_str:<{model_width}}  "
            f"{in_str:>8}  {out_str:>8}  {latency_str:>10}  {status:>7}"
        )
        if e.status != "completed" and e.error_type:
            lines.append(f"{'':<{name_width}}  Error: {e.error_type}")
        total_in += e.input_tokens or 0
        total_out += e.output_tokens or 0
        # Only top-level (no parent) entries go into the latency total --
        # a nested entry's time is already CONTAINED within its parent's
        # window (e.g. a node that calls an agent internally: the node's
        # own latency already includes the agent call), so summing every
        # row regardless of depth would double-count that time and could
        # legitimately exceed the real wall-clock turn latency below,
        # which would read as a bug rather than the expected result of a
        # real parent/child hierarchy.
        if e.parent_span_id is None:
            top_level_latency += e.latency_ms or 0.0

    lines.append(separator)
    lines.append(f"{'Total':<{name_width}}  {'':<{model_width}}  {_fmt_int(total_in):>8}  {_fmt_int(total_out):>8}")
    lines.append("")
    lines.append(f"Top-level time:   {top_level_latency / 1000:.2f}s")
    turn_latency_ms = _turn_latency_ms(turn)
    if turn_latency_ms is not None:
        lines.append(f"End-to-end turn:  {turn_latency_ms / 1000:.2f}s")
    return "\n".join(lines)


def _turn_latency_ms(turn: TurnRecord) -> Optional[float]:
    if turn.ended_at is None:
        return None
    return (turn.ended_at - turn.started_at) * 1000


def print_flow(trace_id: Optional[str] = None) -> None:
    print(format_flow(trace_id))
