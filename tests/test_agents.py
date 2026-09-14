from types import SimpleNamespace as NS

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from goodmem_rag.agents import answer_text, build_agentic_graph, build_react_agent, tool_calls


@tool
def langgraph_docs_tool(query: str) -> str:
    """Search graph documentation."""
    return "Graph evidence: " + query


@tool
def langchain_docs_tool(query: str) -> str:
    """Search chain documentation."""
    return "Chain evidence: " + query


TOOLS = [langgraph_docs_tool, langchain_docs_tool]


class LoopModel:
    """Always asks for both tools when bound, so the graph must enforce its own cap."""

    def __init__(self, relevant=True):
        self.count = 0
        self.relevant = relevant
        self.grading_inputs = []
        self.final_inputs = []

    def bind_tools(self, tools):
        def invoke(messages):
            self.count += 1
            return AIMessage(content="", tool_calls=[
                {"name": t.name, "args": {"query": "evidence"}, "id": f"{self.count}-{t.name}"}
                for t in tools
            ])
        return RunnableLambda(invoke)

    def with_structured_output(self, schema):
        def invoke(messages):
            self.grading_inputs.append(messages[-1].text)
            return NS(binary_score="yes" if self.relevant else "no")
        return RunnableLambda(invoke)

    def invoke(self, messages):
        self.final_inputs.append(messages)
        return AIMessage(content="Final answer from evidence.")


@pytest.mark.parametrize("relevant", [True, False])
def test_graph_caps_hops_completes_calls_and_preserves_the_original_question(relevant):
    model = LoopModel(relevant)
    graph = build_agentic_graph(model, TOOLS, max_hops=2)
    result = graph.invoke({"messages": [HumanMessage(content="Original question")]})
    assert result["hops"] == model.count == 2
    assert answer_text(result) == result["last_answer"] == "Final answer from evidence."
    calls = tool_calls(result)
    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert {c["id"] for c in calls} == {m.tool_call_id for m in replies}
    assert all("Original question" in text for text in model.grading_inputs)
    assert all("Graph evidence" in text and "Chain evidence" in text for text in model.grading_inputs)
    assert "retrieval budget is exhausted" in model.final_inputs[-1][-1].text


class BindableFake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_prebuilt_agent_limits_a_model_that_keeps_calling_tools():
    responses = [AIMessage(content="", tool_calls=[
        {"name": "langgraph_docs_tool", "args": {"query": "q"}, "id": f"call-{i}"}
    ]) for i in range(20)]
    result = build_react_agent(BindableFake(responses=responses), TOOLS).invoke(
        {"messages": [HumanMessage(content="Question")]}, {"recursion_limit": 60}
    )
    assert answer_text(result)
    calls = tool_calls(result)
    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert {c["id"] for c in calls} == {m.tool_call_id for m in replies}
    assert sum(m.text.startswith("Graph evidence") for m in replies) <= 2


def test_unanswered_tool_call_is_not_a_final_answer():
    with pytest.raises(RuntimeError, match="finish"):
        answer_text({"messages": [AIMessage(content="", tool_calls=[
            {"name": "tool", "args": {}, "id": "1"}
        ])]})


def test_source_provenance_does_not_depend_on_model_following_citation_instructions():
    result = {"messages": [
        ToolMessage(content="Source: https://example.org/langgraph/overview\nTitle: Overview\n"
                            "Evidence mentioning https://example.org/unretrieved-page",
                    tool_call_id="1", artifact=[Document(page_content="Evidence",
                        metadata={"source": "https://example.org/langgraph/overview"})]),
        AIMessage(content="Answer without inline citations."),
    ]}
    answer = answer_text(result)
    assert "Sources consulted:" in answer
    assert "https://example.org/langgraph/overview" in answer
    assert "https://example.org/unretrieved-page" not in answer
