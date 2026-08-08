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
FLOW TRACE-58ce6b38

Flow
classifier_agent -> responder_agent

Node/Agent        Model  Input                             Output                             Input(token)  Output(token)     Latency   Status
----------------------------------------------------------------------------------------------------------------------------------------------
classifier_agent  test   What time is it?                  success (no tool calls)                      55              4         0ms       OK
responder_agent   test   What time is it?                  success (no tool calls)                      55              4         0ms       OK
----------------------------------------------------------------------------------------------------------------------------------------------
Total                                                                                                  110              8

Top-level time:   0.00s
End-to-end turn:  0.02s
```

(`Model` reads the real model identifier off the agent, e.g. `gpt-4o-mini` for a real OpenAI-backed agent — shown as `test` here since this example uses pydantic-ai's no-API-key test model. `Input` is the real prompt text (only for a plain-string call — `-` for a message_history-based call with no new prompt). `Output` is the real reply: plain text as-is, or `model_dump_json()` when the agent has a structured `output_type` (a Pydantic model) — either way, truncated to 32 characters for display.)

A plain LangGraph node whose only child is exactly one agent call is shown as a single row, not two — see [Node/agent merging](#nodeagent-merging) below for when that does and doesn't apply.

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

## Node/agent merging

A LangGraph node whose only child is exactly one agent call — the common
"one node, one model call" shape — is collapsed into a single row in
`print_flow()`/`format_flow()`, instead of showing the node and its
agent as two separate rows with the same latency and status repeated.
The node's real span/parent-span identity is kept (so it still nests
correctly under any real ancestor); the agent's name, model, text, and
tokens are what's shown.

This only ever affects display, never the underlying trace: `get_trace()`
still returns both entries, untouched, exactly as recorded. And it only
merges when there's genuinely nothing to lose — a node with zero
children, more than one child, or a child that isn't a plain leaf agent
call (e.g. it calls another agent internally) keeps its full real
hierarchy, since collapsing those would silently drop information.

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
