"""Small acceptance suite, not a claim of superiority over the upstream baseline."""

from __future__ import annotations

import importlib.metadata
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from .agents import (
    answer_text,
    build_agentic_graph,
    build_react_agent,
    evidence_sources,
    tool_calls,
)
from .config import chat_model, save_json
from .retrieval import make_tools

RETRIEVAL_CASES = [
    ("state", "langgraph", "What are state, nodes and edges in StateGraph?", "graph-api", ["state", "nodes", "edges"]),
    ("reducers", "langgraph", "How does the add_messages reducer handle message updates?", "graph-api", ["add_messages"]),
    ("persistence", "langgraph", "What does a checkpointer do when compiling a graph?", "graph-api", ["checkpointer"]),
    ("fanout", "langgraph", "How does the Send API implement map-reduce?", "graph-api", ["Send"]),
    ("tools", "langchain", "How do I define a custom tool with the @tool decorator?", "tools", ["@tool"]),
    ("binding", "langchain", "How do bind_tools and tool_calls work on a chat model?", "models", ["bind_tools", "tool_calls"]),
    ("agents", "langchain", "How do I create an agent using create_agent?", "overview", ["create_agent"]),
    ("structured", "langchain", "How do I get structured output from a chat model?", "models", ["with_structured_output"]),
]
COMPARISON_QUESTION = (
    "Using details from both docs, explain how a LangGraph StateGraph differs from "
    "a LangChain agent's tool-calling loop. Cite both sources."
)
SEQUENTIAL_QUESTION = (
    "First search the LangGraph overview to identify the higher-level framework it recommends "
    "for prebuilt agent architectures. After reading that result, search the recommended "
    "framework's documentation to identify its agent constructor and explain how its "
    "tool-calling loop works. Cite both sources."
)


def _citations(text: str) -> set[str]:
    return {url.rstrip(".,;:!?")
            for url in re.findall(r"https://docs\.langchain\.com/[^\s)\]>]+", text)}


def evaluate(client, settings, state: dict, output: str, retrieval_only: bool = False) -> bool:
    model = None if retrieval_only else chat_model()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": client.system.info().model_dump(mode="json", exclude_none=True),
        "packages": {p: importlib.metadata.version(p) for p in ["goodmem", "langchain-goodmem", "langchain", "langgraph"]},
        "chat_model": getattr(model, "model", None),
        "embedder_model": client.embedders.get(id=state["embedder_id"]).model_identifier,
        "reranker_model": (client.rerankers.get(id=state["reranker_id"]).model_identifier
                           if state.get("reranker_id") else None),
        "documents": [{"source": d["source"], "sha256": d["sha256"]} for d in state["documents"]],
        "retrieval_configuration": {"base_top_k": 5, "rerank_candidates": 20, "rerank_top_k": 5},
        "retrieval": [], "agents": [],
        "limitations": "Small acceptance suite. No Chroma/bge-m3 baseline or answer-quality judge. "
                        "Latency includes network/provider variability. Citation checks verify provenance, "
                        "not that every generated claim is entailed by its source.",
    }
    modes = [False, True] if state.get("reranker_id") else [False]
    for rerank in modes:
        tools = {tool.name: tool for tool in make_tools(client, state, rerank=rerank)}
        for name, collection, query, expected_page, keywords in RETRIEVAL_CASES:
            start = time.monotonic()
            tool = tools[f"{collection}_docs_tool"]
            docs = tool.invoke({"type": "tool_call", "id": f"eval-{name}", "name": tool.name,
                                "args": {"query": query}}).artifact
            text = "\n".join(d.page_content for d in docs).lower()
            sources = [d.metadata.get("source", "") for d in docs]
            passed = (any(s.endswith("/" + expected_page) for s in sources)
                      and all(k.lower() in text for k in keywords)
                      and all(d.metadata["space_id"] == state["spaces"][collection] for d in docs))
            report["retrieval"].append({
                "case": name, "reranked": rerank, "passed": passed,
                "seconds": round(time.monotonic() - start, 3), "sources": sources,
                "scores": [d.metadata["score"] for d in docs],
                "expected_page": expected_page, "keywords": keywords,
            })
            save_json(Path(output), report)
            print(f"Retrieval {name} rerank={rerank}: {'PASS' if passed else 'FAIL'}", flush=True)
    if not retrieval_only:
        cases = [
            ("direct", "What is the capital of France?", set()),
            ("single_space", "What is a checkpointer used for in LangGraph? Cite the docs.", {"langgraph"}),
            ("cross_space", COMPARISON_QUESTION, {"langgraph", "langchain"}),
            ("sequential", SEQUENTIAL_QUESTION, {"langgraph", "langchain"}),
        ]
        for agent_name, builder in [("graph", build_agentic_graph), ("react", build_react_agent)]:
            agent = builder(model, make_tools(client, state))
            for name, question, expected in cases:
                start = time.monotonic()
                result = agent.invoke({"messages": [HumanMessage(content=question)]},
                                      {"recursion_limit": 60})
                answer = answer_text(result)
                calls = tool_calls(result)
                used = {call["name"].removesuffix("_docs_tool") for call in calls}
                replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                source_urls = set(evidence_sources(result))
                cited_urls = _citations(answer)
                rounds = sum(bool(getattr(m, "tool_calls", [])) for m in result["messages"])
                matched = {c["id"] for c in calls} == {m.tool_call_id for m in replies}
                citations_ok = (not expected or (bool(cited_urls) and cited_urls <= source_urls
                                and all(any(f"/{s}/" in u for u in cited_urls) for s in expected)))
                passed = (used == expected and matched and citations_ok
                          and (name != "sequential" or rounds >= 2)
                          and (name != "direct" or "paris" in answer.lower()))
                report["agents"].append({
                    "agent": agent_name, "case": name, "question": question, "passed": passed,
                    "seconds": round(time.monotonic() - start, 3), "tool_calls": calls,
                    "retrieval_rounds": rounds, "citations": sorted(cited_urls),
                    "all_tool_calls_answered": matched, "answer": answer,
                    "model_answer": result["messages"][-1].text,
                    "model_inline_citations": sorted(_citations(result["messages"][-1].text)),
                })
                # Save partial results if a later provider request fails.
                save_json(Path(output), report)
                print(f"Agent {agent_name}/{name}: {'PASS' if passed else 'FAIL'} "
                      f"({len(calls)} calls, {rounds} rounds)", flush=True)
    report["passed"] = all(c["passed"] for c in report["retrieval"] + report["agents"])
    save_json(Path(output), report)
    print(f"Report saved to {output}")
    return report["passed"]
