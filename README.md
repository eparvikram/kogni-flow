# kogni-flow

Auto-instrumentation for LangGraph apps. The entire integration:

```bash
pip install kogni-flow
```

```python
from kogni_langgraph_flow import enable_flow

app = graph.compile()
app = enable_flow(app)
app.invoke(...)             # traced automatically from here on
```

No decorator, no wrapping any node function or LLM call by hand.

## What gets traced automatically

1. **Every LangGraph node** — which ran, in what order, how long, whether
   it errored. Via LangGraph's own callback system
   (`config={"callbacks": [...]}`), verified directly against a real
   compiled graph: LangGraph fires `on_chain_start`/`on_chain_end` per
   node with the node's name at `metadata["langgraph_node"]`. Works for
   any LangGraph app, regardless of what's inside each node.
2. **Every pydantic-ai `Agent.run_sync()` call, anywhere** — best-effort,
   via a one-time monkey-patch of `Agent.run_sync` (only if `pydantic_ai`
   is importable; silently skipped otherwise, node-level tracing still
   works). Real input/output tokens, real latency, and the agent's own
   variable name (`classifier_agent`, not a generic placeholder).

Both feed the same trace, so a node that internally calls an agent shows
up as real parent/child hierarchy, not a flat list.

## Output

```
FLOW TRACE-5ed8fdcf

Flow
classify
`-- classifier_agent
respond
`-- responder_agent

Node/Agent           Input    Output     Latency   Status
---------------------------------------------------------
classify                 -         -      2250ms       OK
classifier_agent        23         3      2250ms       OK
respond                  -         -      1265ms       OK
responder_agent         19        10      1265ms       OK
---------------------------------------------------------
Total                   42        13

Top-level time:   3.51s
End-to-end turn:  3.53s
```

`Top-level time` sums only top-level (no-parent) entries' latency, not
every row — a nested entry's time is already contained within its
parent's window (the node's own latency already includes the agent call
inside it), so summing every row regardless of depth would double-count
and could read as a bug. `End-to-end turn` is the real wall-clock time
for the whole `invoke()` call; the two diverge once anything runs
concurrently.

```python
from kogni_langgraph_flow import print_flow, format_flow

print_flow()          # prints the table above for the last-run turn
format_flow()          # same, but returns the string instead of printing
```

## Custom token-usage shapes

Built-in adapters cover pydantic-ai, raw OpenAI, and LangChain-style
responses (`kogni_langgraph_flow/adapters/`). For anything else:

```python
from kogni_langgraph_flow import register_adapter
from kogni_langgraph_flow.models import TokenUsage

def my_adapter(result):
    if not hasattr(result, "my_usage_field"):
        return None
    return TokenUsage(
        input_tokens=result.my_usage_field.input,
        output_tokens=result.my_usage_field.output,
        available=True,
    )

register_adapter(my_adapter)
```

Adapters are tried in order (custom ones first); the first one that
recognizes the shape wins. If none do, tokens show as `-`, never a
guessed number.

## Agent naming

pydantic-ai's own `Agent(infer_name=True)` (the default) normally sets
`agent.name` to whatever variable it's assigned to, the first time it
runs. Patching `run_sync` breaks that (pydantic-ai's own inference walks
up exactly one stack frame, which lands on our wrapper's own parameter
instead of your code's variable) — verified directly: without a fix,
every agent showed up literally named `"self"`. This package's
instrumentation replicates the same frame-walk one level higher, so it
still resolves to `classifier_agent`, not a generic placeholder — no
naming convention required on your part, as long as the agent is a
plain module/function-local variable (not, say, an item inside a list
literal that's never bound to its own name, which falls back to an
id-based placeholder).

## Known limitation

Only `app.invoke()`/`app.ainvoke()` are wrapped, not `app.stream()`/
`app.astream()` directly. Verified directly (not assumed) that
LangGraph's own `Pregel.invoke()` is implemented by internally calling
`self.stream()` — wrapping both would mean invoke's own internal call
picks up the already-wrapped stream a second time, double-injecting
config and crashing. Since invoke already delegates to stream
internally, every node still gets traced through invoke's callback
injection alone. A caller that calls `.stream()` directly, never
`.invoke()`/`.ainvoke()`, won't get automatic turn boundaries yet.

## Storage

In-memory only, deliberately not a database (`tracer.py`, bounded to the
last 200 turns). This is a developer-debugging aid for "what just
happened," not an audit trail — nothing here needs to survive a process
restart.

## Development

```bash
pip install -e ".[dev]"
pytest
python examples/basic_usage.py
```

## Package layout

```
src/kogni_langgraph_flow/
├── __init__.py          # public API re-exports
├── api.py                # enable_flow()
├── instrumentation.py     # FlowCallbackHandler + patch_pydantic_ai_agent
├── models.py               # TokenUsage, AgentTrace, TurnRecord
├── context.py               # per-turn/per-span contextvars
├── tracer.py                  # in-memory store
├── formatter.py                # print_flow()/format_flow()
└── adapters/                    # per-SDK token-usage extraction
    ├── __init__.py
    ├── pydantic_ai.py
    ├── openai.py
    └── langchain.py
```

## License

MIT
