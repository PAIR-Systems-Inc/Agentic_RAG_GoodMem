"""Application descriptions for the shared GoodMem retrieval tools."""

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_goodmem import GoodMemRetriever


def make_tools(client, state: dict, *, rerank: bool = True):
    descriptions = {
        "langgraph": "Search LangGraph docs for StateGraph, nodes, edges, reducers, persistence and workflows.",
        "langchain": "Search LangChain docs for models, tools, prompts, built-in agents and tool-calling loops.",
    }
    return [
        create_retriever_tool(
            GoodMemRetriever(
                client=client, space_ids=[state["spaces"][collection]],
                filter="CAST(val('$.application') AS TEXT) = 'agentic-rag-goodmem'",
                reranker_id=state.get("reranker_id") if rerank else None,
            ),
            name=f"{collection}_docs_tool", description=description,
            document_prompt=PromptTemplate.from_template(
                "Source: {source}\nTitle: {title}\n{page_content}"
            ),
            document_separator="\n\n---\n\n",
            response_format="content_and_artifact",
        )
        for collection, description in descriptions.items()
    ]
