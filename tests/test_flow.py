"""End-to-end tests against a real (tiny) LangGraph app + real pydantic-ai
Agent objects using the 'test' model, which needs no API key and no
network call -- see https://ai.pydantic.dev/testing/. No mocking of
kogni_langgraph_flow itself: these exercise the actual LangGraph
callback wiring and the actual Agent.run_sync patch, the same way
README.md's example does.
"""

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent

from kogni_langgraph_flow import clear_all, enable_flow, format_flow, get_trace
from kogni_langgraph_flow.context import current_trace_id


class _State(TypedDict):
    question: str
    answer: str


@pytest.fixture(autouse=True)
def _clear_traces():
    clear_all()
    yield
    clear_all()


def _build_app():
    classifier_agent = Agent("test", system_prompt="classify")
    responder_agent = Agent("test", system_prompt="respond")

    def classify_node(state: _State) -> dict:
        result = classifier_agent.run_sync(state["question"])
        return {"answer": result.output}

    def respond_node(state: _State) -> dict:
        result = responder_agent.run_sync(state["question"])
        return {"answer": result.output}

    graph = StateGraph(_State)
    graph.add_node("classify", classify_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)
    return enable_flow(graph.compile()), classifier_agent, responder_agent


def test_enable_flow_traces_every_node_and_agent():
    app, classifier_agent, responder_agent = _build_app()
    app.invoke({"question": "hello", "answer": ""})

    trace_id = current_trace_id()
    turn = get_trace(trace_id)
    names = sorted(e.agent_name for e in turn.executions)
    assert names == ["classifier_agent", "classify", "respond", "responder_agent"]


def test_agent_calls_nest_under_their_node():
    app, *_ = _build_app()
    app.invoke({"question": "hello", "answer": ""})

    turn = get_trace(current_trace_id())
    by_name = {e.agent_name: e for e in turn.executions}
    assert by_name["classifier_agent"].parent_span_id == by_name["classify"].span_id
    assert by_name["responder_agent"].parent_span_id == by_name["respond"].span_id
    assert by_name["classify"].parent_span_id is None
    assert by_name["respond"].parent_span_id is None


def test_real_tokens_are_captured():
    app, *_ = _build_app()
    app.invoke({"question": "hello", "answer": ""})

    turn = get_trace(current_trace_id())
    agent_records = [e for e in turn.executions if e.agent_name in ("classifier_agent", "responder_agent")]
    assert len(agent_records) == 2
    for record in agent_records:
        assert record.input_tokens is not None and record.input_tokens > 0
        assert record.output_tokens is not None and record.output_tokens > 0
        assert record.status == "completed"


def test_node_error_is_recorded_as_failed():
    def boom_node(state: _State) -> dict:
        raise ValueError("simulated failure")

    graph = StateGraph(_State)
    graph.add_node("boom", boom_node)
    graph.add_edge(START, "boom")
    graph.add_edge("boom", END)
    app = enable_flow(graph.compile())

    with pytest.raises(ValueError):
        app.invoke({"question": "x", "answer": ""})

    turn = get_trace(current_trace_id())
    assert len(turn.executions) == 1
    assert turn.executions[0].status == "failed"
    assert turn.executions[0].error_type == "ValueError"


def test_enable_flow_is_idempotent():
    app, *_ = _build_app()
    same_app = enable_flow(app)  # calling again shouldn't double-wrap or error
    assert same_app is app
    result = app.invoke({"question": "hello", "answer": ""})
    assert result["answer"]


def test_format_flow_renders_a_table():
    app, *_ = _build_app()
    app.invoke({"question": "hello", "answer": ""})

    text = format_flow(current_trace_id())
    assert "Flow" in text
    assert "classify" in text
    assert "classifier_agent" in text
    assert "Top-level time:" in text
