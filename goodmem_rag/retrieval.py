"""Normalize GoodMem's event stream into cited LangChain Documents."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.tools import tool


class RetrievalFailure(RuntimeError):
    """GoodMem reported a failed/partial retrieval, rather than a true empty match."""


def documents_from_events(events) -> list[Document]:
    events = list(events)
    # Definitions and chunks are separate events. Join by memory UUID, not event order.
    memories = {event.memory_definition.memory_id: event.memory_definition
                for event in events if event.memory_definition is not None}
    documents = []
    seen = set()
    for event in events:
        status = event.status
        summary_disabled = (status is not None and status.code == "FEATURE_DISABLED"
                            and (status.details or {}).get("feature") == "summarization"
                            and (status.details or {}).get("required_param") == "llm_id")
        # The rerank-only postprocessor emits this notice even though reranking succeeded.
        if status and not summary_disabled and status.code != "LLM_CAPABILITY_INFERRED":
            raise RetrievalFailure(
                f"GoodMem retrieval status {event.status.code}: {event.status.message}"
            )
        item = event.retrieved_item
        if item is None or item.chunk is None or item.chunk.chunk is None:
            continue
        hit = item.chunk
        chunk = hit.chunk
        if chunk.chunk_id in seen or not chunk.chunk_text:
            continue
        memory = memories.get(chunk.memory_id)
        if memory is None:
            raise RetrievalFailure(f"Missing memory metadata for chunk {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        documents.append(Document(
            page_content=chunk.chunk_text,
            metadata=dict(memory.metadata or {}) | {
                "memory_id": chunk.memory_id, "chunk_id": chunk.chunk_id,
                "space_id": memory.space_id, "score": hit.relevance_score,
            },
        ))
    return documents


class GoodMemRetriever:
    def __init__(self, client, space_id: str, *, k: int = 5, reranker_id: str | None = None):
        if k < 1:
            raise ValueError("k must be positive")
        self.client, self.space_id, self.k, self.reranker_id = client, space_id, k, reranker_id

    def invoke(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        options = {}
        if self.reranker_id:
            options = {"reranker_id": self.reranker_id, "max_results": self.k,
                       "chronological_resort": False}
        events = self.client.memories.retrieve(
            message=query, space_ids=[self.space_id],
            requested_size=self.k * 4 if self.reranker_id else self.k,
            fetch_memory=True, fetch_memory_content=False, stream=False, **options,
        )
        return documents_from_events(events)


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "No relevant documents found in this GoodMem space."
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Title: {doc.metadata.get('title', '')}\n{doc.page_content}" for doc in documents
    )


def make_tools(client, state: dict, *, rerank: bool = True):
    reranker_id = state.get("reranker_id") if rerank else None
    graph = GoodMemRetriever(client, state["spaces"]["langgraph"], reranker_id=reranker_id)
    chain = GoodMemRetriever(client, state["spaces"]["langchain"], reranker_id=reranker_id)

    @tool
    def langgraph_docs_tool(query: str) -> str:
        """Search LangGraph docs for StateGraph, nodes, edges, reducers, persistence and workflows."""
        return format_documents(graph.invoke(query))

    @tool
    def langchain_docs_tool(query: str) -> str:
        """Search LangChain docs for models, tools, prompts, built-in agents and tool-calling loops."""
        return format_documents(chain.invoke(query))

    return [langgraph_docs_tool, langchain_docs_tool]
