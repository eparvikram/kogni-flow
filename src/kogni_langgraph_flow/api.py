"""enable_flow(app) -- the entire integration surface:

    from kogni_langgraph_flow import enable_flow
    app = graph.compile()
    enable_flow(app)
    app.invoke(...)             # traced automatically from here on

One call. Wraps app's invoke/ainvoke so every call:
  1. starts (and ends) one trace turn automatically,
  2. gets a FlowCallbackHandler injected into its LangGraph config, so
     every NODE that runs gets recorded with zero changes inside any
     node function (see instrumentation.py -- verified against a real
     compiled graph, not assumed),
  3. best-effort: has every pydantic-ai Agent.run_sync call anywhere
     traced too, with real tokens, nested under whichever node is
     currently running if any (see instrumentation.py).

Then read it back with print_flow()/format_flow().
"""

import functools
from typing import Any

from . import context as flow_context
from .instrumentation import FlowCallbackHandler, patch_pydantic_ai_agent

# invoke/ainvoke ONLY -- verified directly (not assumed) that LangGraph's
# own Pregel.invoke() is implemented internally by calling self.stream()
# (a traceback surfaced this: "Pregel.invoke ... for chunk in
# self.stream(...)"). Wrapping stream/astream TOO would mean invoke's own
# internal call picks up our already-wrapped stream a second time,
# double-injecting config and crashing ("got multiple values for
# argument 'config'"). Since invoke already delegates to stream
# internally, every node still gets traced through invoke's callback
# injection alone -- stream() called directly (not via invoke) just
# isn't a wrapped entry point yet: a real, separate limitation, not a
# bug -- a caller streaming directly, never calling invoke/ainvoke,
# won't get automatic turn boundaries. Worth revisiting if that turns
# out to matter in practice.
_WRAPPED_METHODS = ("invoke", "ainvoke")


def enable_flow(app: Any) -> Any:
    """Mutates app in place and returns it too, so both

        enable_flow(app)

    and

        app = enable_flow(graph.compile())

    work. Idempotent -- calling it again on the same app (or a second
    app, in the same process) is safe; the pydantic-ai patch only ever
    applies once regardless of how many apps/how many times this runs."""
    if getattr(app, "_kogni_flow_enabled", False):
        return app

    patch_pydantic_ai_agent()  # best-effort; False (silently) if pydantic_ai isn't installed
    handler = FlowCallbackHandler()

    for method_name in _WRAPPED_METHODS:
        original = getattr(app, method_name, None)
        if original is None:
            continue
        is_async = method_name.startswith("a")
        setattr(app, method_name, _wrap(original, handler, is_async=is_async))

    app._kogni_flow_enabled = True
    return app


def _inject_callback(handler: FlowCallbackHandler, kwargs: dict) -> dict:
    config = dict(kwargs.get("config") or {})
    config["callbacks"] = list(config.get("callbacks") or []) + [handler]
    kwargs["config"] = config
    return kwargs


def _wrap(original, handler: FlowCallbackHandler, is_async: bool):
    if is_async:
        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs = _inject_callback(handler, kwargs)
            with flow_context.trace_turn():
                return await original(*args, **kwargs)
        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        kwargs = _inject_callback(handler, kwargs)
        with flow_context.trace_turn():
            return original(*args, **kwargs)
    return wrapper
