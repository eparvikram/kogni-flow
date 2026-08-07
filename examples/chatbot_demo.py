"""Interactive chatbot demo: the same 5-agent pipeline as before, wrapped
in a chat loop instead of a single one-shot call. Trace display is
filtered to agent calls only -- kogni-flow's FlowCallbackHandler still
traces every LangGraph node underneath (that's the library's whole
point, and other consumers of this trace may still want it), this demo
just prints a simplified view that only shows the LLM calls, not the
plain node wrappers around them (see print_agent_trace() below).

Input/Output TEXT columns are tracked here in the demo, not inside
kogni_langgraph_flow itself -- the library's AgentTrace deliberately
never stores prompts/full outputs (a real privacy/security choice: "no
prompts, no full outputs, no business payloads" applies to every
consumer of the library, not just this demo). Token COUNTS still come
from the real trace (get_trace); the text alongside them is captured
directly in each node function below and joined in by agent_name for
display only.

Uses a real model (openai:gpt-4o-mini), not pydantic-ai's no-API-key
'test' model -- canned "success (no tool calls)" replies would feel
broken in an actual back-and-forth chat. Needs OPENAI_API_KEY set (see
.env loading below).

Run with:  python chatbot_demo.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent

# Windows terminals often default stdout to a non-UTF-8 codepage (cp1252),
# which can't encode characters a model's response commonly includes
# (smart quotes, em dashes, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kogni_langgraph_flow import enable_flow, get_trace
from kogni_langgraph_flow.context import current_trace_id
from kogni_langgraph_flow.models import AgentTrace

# Looks for a .env file in this directory (or the current working
# directory) -- copy .env.example to .env here and set OPENAI_API_KEY,
# or just export it in your shell before running this script.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()  # also picks up a .env in the current working directory

MODEL = "openai:gpt-4o-mini"

TEXT_COL_WIDTH = 32  # truncated display width for the Input/Output text columns


class Exchange(TypedDict):
    input: str
    output: str


class State(TypedDict):
    message: str
    reply: str
    exchanges: Dict[str, Exchange]  # agent_name -> {input, output} text, display-only


intake_agent = Agent(MODEL, system_prompt="Classify the user's message in a few words (question, request, greeting, etc). Reply with just the classification.")
planner_agent = Agent(MODEL, system_prompt="Write a one-line plan for how to respond to the user's message.")
researcher_agent = Agent(MODEL, system_prompt="Note one relevant fact or consideration for replying to the user's message. One sentence.")
writer_agent = Agent(MODEL, system_prompt="Write a short, friendly reply to the user's message. Two sentences at most.")
reviewer_agent = Agent(MODEL, system_prompt="You are given a draft reply. Lightly polish it for tone and clarity, and return ONLY the final reply text, nothing else.")


def _record(state: State, agent_name: str, prompt: str, output: str) -> Dict[str, Exchange]:
    return {**state["exchanges"], agent_name: {"input": prompt, "output": output}}


def intake_node(state: State) -> dict:
    result = intake_agent.run_sync(state["message"])
    return {"exchanges": _record(state, "intake_agent", state["message"], result.output)}


def planner_node(state: State) -> dict:
    result = planner_agent.run_sync(state["message"])
    return {"exchanges": _record(state, "planner_agent", state["message"], result.output)}


def researcher_node(state: State) -> dict:
    result = researcher_agent.run_sync(state["message"])
    return {"exchanges": _record(state, "researcher_agent", state["message"], result.output)}


def writer_node(state: State) -> dict:
    result = writer_agent.run_sync(state["message"])
    return {"exchanges": _record(state, "writer_agent", state["message"], result.output), "reply": result.output}


def reviewer_node(state: State) -> dict:
    prompt = f"Draft reply: {state['reply']}"
    result = reviewer_agent.run_sync(prompt)
    return {"exchanges": _record(state, "reviewer_agent", prompt, result.output), "reply": result.output}


def build_app():
    graph = StateGraph(State)
    graph.add_node("intake", intake_node)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)
    return graph.compile()


def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def print_agent_trace(trace_id: Optional[str], exchanges: Dict[str, Exchange]) -> None:
    """Agent calls only -- filters out the plain LangGraph node wrappers
    kogni-flow also traces by default (see module docstring). Deliberately
    a small, self-contained printer rather than reusing kogni_langgraph_flow
    .formatter directly, since that one is built to show node+agent
    hierarchy together, not a filtered subset, and doesn't carry text
    content at all."""
    turn = get_trace(trace_id) if trace_id else None
    if not turn or not turn.executions:
        print("(no trace recorded)")
        return

    agents: List[AgentTrace] = [
        e for e in sorted(turn.executions, key=lambda e: e.sequence) if e.model_name is not None
    ]
    if not agents:
        print("(no agent calls recorded)")
        return

    print(f"AGENT TRACE {trace_id}")
    print()
    print(" -> ".join(a.agent_name for a in agents))
    print()

    name_w = max(len("Agent"), max(len(a.agent_name) for a in agents))
    model_w = max(len("Model"), max(len(a.model_name or "-") for a in agents))
    in_text_w = out_text_w = TEXT_COL_WIDTH
    header = (
        f"{'Agent':<{name_w}}  {'Model':<{model_w}}  "
        f"{'Input':<{in_text_w}}  {'Output':<{out_text_w}}  "
        f"{'Input(token)':>13}  {'Output(token)':>13}  {'Latency':>10}  {'Status':>7}"
    )
    print(header)
    print("-" * len(header))

    total_in = total_out = 0
    total_latency_ms = 0.0
    for a in agents:
        status = "OK" if a.status == "completed" else "ERROR"
        exchange = exchanges.get(a.agent_name, {"input": "-", "output": "-"})
        in_text = _truncate(exchange["input"], in_text_w)
        out_text = _truncate(exchange["output"], out_text_w)
        in_tok = f"{a.input_tokens:,}" if a.input_tokens is not None else "-"
        out_tok = f"{a.output_tokens:,}" if a.output_tokens is not None else "-"
        lat_s = f"{a.latency_ms:.0f}ms" if a.latency_ms is not None else "-"
        print(
            f"{a.agent_name:<{name_w}}  {a.model_name or '-':<{model_w}}  "
            f"{in_text:<{in_text_w}}  {out_text:<{out_text_w}}  "
            f"{in_tok:>13}  {out_tok:>13}  {lat_s:>10}  {status:>7}"
        )
        total_in += a.input_tokens or 0
        total_out += a.output_tokens or 0
        total_latency_ms += a.latency_ms or 0.0

    print("-" * len(header))
    print(
        f"{'Total':<{name_w}}  {'':<{model_w}}  {'':<{in_text_w}}  {'':<{out_text_w}}  "
        f"{total_in:>13,}  {total_out:>13,}"
    )
    print()
    # "Agent time" here, not "Top-level time" -- with node rows filtered
    # out, every remaining row's parent_span_id still points at its
    # (hidden) node, so none of them read as "top-level" anymore; this
    # is a straight sum of the agent rows actually shown above instead.
    print(f"Agent time:       {total_latency_ms / 1000:.2f}s")
    if turn.ended_at is not None:
        turn_latency_ms = (turn.ended_at - turn.started_at) * 1000
        print(f"End-to-end turn:  {turn_latency_ms / 1000:.2f}s")


if __name__ == "__main__":
    app = enable_flow(build_app())

    print("Kogni Flow demo chatbot -- 5 agents run on every message. Type 'exit' to quit.\n")
    while True:
        user_input = input("You > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        result = app.invoke({"message": user_input, "reply": "", "exchanges": {}})
        print(f"Bot > {result['reply']}\n")
        print_agent_trace(current_trace_id(), result["exchanges"])
        print()
