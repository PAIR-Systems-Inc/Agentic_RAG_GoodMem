"""The shared integration owns retrieval semantics; test the app's space bindings."""

from unittest.mock import Mock

import pytest
from goodmem import Goodmem
from langchain_core.messages import ToolMessage

from goodmem_rag.retrieval import make_tools


@pytest.mark.parametrize("rerank", [True, False])
def test_tools_use_the_selected_collection_and_reranker(rerank):
    client = Mock(spec=Goodmem)
    client.memories = Mock()
    client.memories.retrieve.return_value = []
    state = {"spaces": {"langgraph": "graph-space", "langchain": "chain-space"},
             "reranker_id": "reranker"}
    tools = make_tools(client, state, rerank=rerank)
    for tool, space in zip(tools, state["spaces"].values(), strict=True):
        assert set(tool.get_input_schema().model_fields) == {"query"}
        result = tool.invoke({"type": "tool_call", "id": "call", "name": tool.name,
                              "args": {"query": "question"}})
        assert isinstance(result, ToolMessage) and result.artifact == []
        request = client.memories.retrieve.call_args.kwargs
        assert request["space_ids"] == [space]
        assert request.get("reranker_id") == ("reranker" if rerank else None)
        assert request["requested_size"] == (20 if rerank else 5)
