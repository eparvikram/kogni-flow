"""In-memory store, deliberately not SQLite/any DB -- this is a
developer-debugging aid for "what just happened," not an audit trail --
nothing here needs to survive a process restart.

Traces are kept in a bounded ring of the last _MAX_TRACES turns so a
long-running process doesn't leak memory turn after turn; each
individual turn's execution list is unbounded (a single turn isn't
going to have thousands of node/agent calls).
"""

import time
from collections import OrderedDict
from typing import Dict, Optional

from .models import AgentTrace, TurnRecord

_MAX_TRACES = 200

_traces: "OrderedDict[str, TurnRecord]" = OrderedDict()
_sequence_counters: Dict[str, int] = {}


def _get_or_create(trace_id: str) -> TurnRecord:
    if trace_id not in _traces:
        if len(_traces) >= _MAX_TRACES:
            oldest, _ = _traces.popitem(last=False)  # evict oldest
            _sequence_counters.pop(oldest, None)
        _traces[trace_id] = TurnRecord(trace_id=trace_id)
    return _traces[trace_id]


def start_trace(trace_id: str) -> None:
    """Explicit turn start -- ensures TurnRecord.started_at reflects the
    real beginning of the turn, not whenever the first node happens to
    get traced."""
    _get_or_create(trace_id)


def next_sequence(trace_id: str) -> int:
    _sequence_counters[trace_id] = _sequence_counters.get(trace_id, 0) + 1
    return _sequence_counters[trace_id]


def record(trace_id: str, execution: AgentTrace) -> None:
    _get_or_create(trace_id).executions.append(execution)


def mark_turn_ended(trace_id: Optional[str]) -> None:
    if trace_id and trace_id in _traces:
        _traces[trace_id].ended_at = time.time()


def get_trace(trace_id: str) -> Optional[TurnRecord]:
    return _traces.get(trace_id)


def clear_trace(trace_id: str) -> None:
    _traces.pop(trace_id, None)
    _sequence_counters.pop(trace_id, None)


def clear_all() -> None:
    _traces.clear()
    _sequence_counters.clear()
