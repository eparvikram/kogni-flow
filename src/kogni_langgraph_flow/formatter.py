"""Renders one TurnRecord as a plain-text developer view: a flow
(sequential '->' chain, or an indented tree if any node/agent call
actually nested another traced call inside it), then a token/latency/
status table, then totals -- including BOTH the summed per-call latency
and the actual wall-clock turn latency, since they diverge whenever
work runs concurrently.
"""

from dataclasses import replace
from typing import Dict, List, Optional

from .context import current_trace_id
from .models import AgentTrace, TurnRecord
from .tracer import get_trace


def _merge_node_agent_pairs(executions: List[AgentTrace]) -> List[AgentTrace]:
    """Collapse a plain node whose ONLY child is exactly one agent call
    into a single row -- the common "one node, just wraps one model
    call" shape, where the node is pure plumbing and showing it as a
    separate row from the agent just repeats the same latency/status
    twice for no new information. Takes the node's span_id/parent_span_id
    (so it still nests correctly under any real ancestor) but the
    agent's name/model/text/tokens (the actually meaningful data).

    Left alone (real hierarchy preserved) whenever there's something
    genuinely to show: a node with zero children, multiple children, or
    a child that isn't a plain leaf agent call -- merging those would
    either have nothing to merge or would silently drop information."""
    by_parent: Dict[Optional[str], List[AgentTrace]] = {}
    for e in executions:
        by_parent.setdefault(e.parent_span_id, []).append(e)

    merged_child_span_ids = set()
    result: List[AgentTrace] = []
    for e in executions:
        if e.span_id in merged_child_span_ids:
            continue
        children = by_parent.get(e.span_id, [])
        if e.model_name is None and len(children) == 1:
            child = children[0]
            if child.model_name is not None and not by_parent.get(child.span_id):
                merged_child_span_ids.add(child.span_id)
                result.append(replace(
                    e,
                    agent_name=child.agent_name,
                    model_name=child.model_name,
                    input_text=child.input_text,
                    output_text=child.output_text,
                    input_tokens=child.input_tokens,
                    output_tokens=child.output_tokens,
                    status=child.status if e.status == "completed" else e.status,
                    error_type=child.error_type or e.error_type,
                ))
                continue
        result.append(e)
    return result


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


_TEXT_WIDTH = 32  # truncated display width for the Input/Output text columns


def _fmt_text(value: Optional[str], width: int = _TEXT_WIDTH) -> str:
    if value is None:
        return "-"
    value = value.replace("\n", " ").strip()
    return value if len(value) <= width else value[: width - 1] + "…"


def _iteration_counts(executions: List[AgentTrace]) -> Dict[str, int]:
    """How many times each Node/Agent name appears in this trace -- the
    generic signal that a loop (e.g. a repair/retry edge routing back to
    an earlier node) fired more than once, without this package needing
    to know anything about what the loop means. Computed AFTER merging
    (see _merge_node_agent_pairs) so a merged node+agent pair counts as
    one occurrence per real iteration, not two. Only names that actually
    repeated are returned -- a normal, loop-free flow has none."""
    counts: Dict[str, int] = {}
    for e in executions:
        counts[e.agent_name] = counts.get(e.agent_name, 0) + 1
    return {name: count for name, count in counts.items() if count > 1}


def format_flow(trace_id: Optional[str] = None) -> str:
    trace_id = trace_id or current_trace_id()
    if not trace_id:
        return "No trace_id given and no current turn -- call enable_flow(app) and invoke it first."

    turn: Optional[TurnRecord] = get_trace(trace_id)
    if not turn or not turn.executions:
        return f"No node/agent executions recorded for trace_id={trace_id!r}."

    executions = _merge_node_agent_pairs(sorted(turn.executions, key=lambda e: e.sequence))

    lines = [f"FLOW {trace_id}", ""]
    lines.append("Flow")
    lines.extend(_format_flow(executions))
    lines.append("")

    name_width = max(len("Node/Agent"), max(len(e.agent_name) for e in executions))
    model_width = max(len("Model"), max(len(e.model_name or "-") for e in executions))
    token_width = 13  # fits "Output(token)", the wider of the two token column labels
    in_text_width = out_text_width = _TEXT_WIDTH
    header = (
        f"{'Node/Agent':<{name_width}}  {'Model':<{model_width}}  "
        f"{'Input':<{in_text_width}}  {'Output':<{out_text_width}}  "
        f"{'Input(token)':>{token_width}}  {'Output(token)':>{token_width}}  {'Latency':>10}  {'Status':>7}"
    )
    separator = "-" * len(header)
    lines.append(header)
    lines.append(separator)

    total_in, total_out, top_level_latency = 0, 0, 0.0
    for e in executions:
        status = "OK" if e.status == "completed" else "ERROR"
        model_str = e.model_name or "-"
        in_text_str = _fmt_text(e.input_text, in_text_width)
        out_text_str = _fmt_text(e.output_text, out_text_width)
        in_str = _fmt_int(e.input_tokens)
        out_str = _fmt_int(e.output_tokens)
        latency_str = f"{e.latency_ms:.0f}ms" if e.latency_ms is not None else "-"
        lines.append(
            f"{e.agent_name:<{name_width}}  {model_str:<{model_width}}  "
            f"{in_text_str:<{in_text_width}}  {out_text_str:<{out_text_width}}  "
            f"{in_str:>{token_width}}  {out_str:>{token_width}}  {latency_str:>10}  {status:>7}"
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
    lines.append(
        f"{'Total':<{name_width}}  {'':<{model_width}}  {'':<{in_text_width}}  {'':<{out_text_width}}  "
        f"{_fmt_int(total_in):>{token_width}}  {_fmt_int(total_out):>{token_width}}"
    )
    lines.append("")
    lines.append(f"Top-level time:   {top_level_latency / 1000:.2f}s")
    turn_latency_ms = _turn_latency_ms(turn)
    if turn_latency_ms is not None:
        lines.append(f"End-to-end turn:  {turn_latency_ms / 1000:.2f}s")

    repeats = _iteration_counts(executions)
    if repeats:
        lines.append("")
        lines.append("Iterations (ran more than once this turn -- e.g. a retry/repair loop):")
        for name, count in repeats.items():
            lines.append(f"  {name}: {count}x")

    return "\n".join(lines)


def _turn_latency_ms(turn: TurnRecord) -> Optional[float]:
    if turn.ended_at is None:
        return None
    return (turn.ended_at - turn.started_at) * 1000


def print_flow(trace_id: Optional[str] = None) -> None:
    print(format_flow(trace_id))
