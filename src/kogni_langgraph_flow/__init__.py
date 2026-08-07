"""kogni_langgraph_flow -- the entire integration surface is one call:

    from kogni_langgraph_flow import enable_flow
    app = graph.compile()
    enable_flow(app)
    app.invoke(...)             # traced automatically from here on
    print_flow()

No decorator, no manual wrapping of any node function or LLM call -- see
api.py for how (LangGraph callbacks for node-level tracing, a
best-effort pydantic-ai patch for per-agent token/latency, both in
instrumentation.py).
"""

from .adapters import register_adapter
from .api import enable_flow
from .formatter import format_flow, print_flow
from .tracer import clear_all, clear_trace, get_trace

__all__ = [
    "enable_flow",
    "print_flow", "format_flow",
    "get_trace", "clear_trace", "clear_all",
    "register_adapter",
]

__version__ = "0.1.0"
