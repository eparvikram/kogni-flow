"""Runnable end to end with no API key -- uses pydantic-ai's built-in
'test' model (https://ai.pydantic.dev/testing/). Swap "test" for a real
model string (e.g. "openai:gpt-4o-mini", with OPENAI_API_KEY set) to see
real model output; the tracing/token-capture code below is unchanged
either way.

Run with:  python examples/basic_usage.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent

from kogni_langgraph_flow import enable_flow, print_flow


class State(TypedDict):
    question: str
    answer: str


classifier_agent = Agent("test", system_prompt="Reply with one word: GREETING or QUESTION.")
responder_agent = Agent("test", system_prompt="Reply in one short sentence.")


def classify_node(state: State) -> dict:
    result = classifier_agent.run_sync(state["question"])
    return {"answer": result.output}


def respond_node(state: State) -> dict:
    result = responder_agent.run_sync(state["question"])
    return {"answer": result.output}


def build_app():
    graph = StateGraph(State)
    graph.add_node("classify", classify_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_app()

    # ---- the entire integration ----
    app = enable_flow(app)
    # ---------------------------------

    result = app.invoke({"question": "hello there", "answer": ""})
    print("RESULT:", result)
    print()
    print_flow()
