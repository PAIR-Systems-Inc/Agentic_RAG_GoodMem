"""Reproduce boundary behavior against the configured GoodMem server."""

import uuid
from pathlib import Path

from goodmem import Goodmem
from goodmem.errors import AuthenticationError, NotFoundError

from goodmem_rag.config import Settings, save_json
from goodmem_rag.retrieval import GoodMemRetriever, RetrievalFailure, documents_from_events

settings = Settings.from_env()
state = settings.state()
report = {}
with settings.client() as client:
    # A valid empty space and an inaccessible/missing space must not look the same.
    empty = client.spaces.create(
        name="Agentic RAG temporary empty-space probe",
        space_embedders=[{"embedder_id": state["embedder_id"]}],
    )
    try:
        report["empty_space_returns_empty"] = GoodMemRetriever(client, empty.space_id).invoke("state") == []
    finally:
        client.spaces.delete(id=empty.space_id)
    try:
        GoodMemRetriever(client, str(uuid.uuid4())).invoke("state")
        report["missing_space_detected"] = False
    except (RetrievalFailure, NotFoundError) as exc:
        report["missing_space_detected"] = True
        report["missing_space_diagnostic"] = str(exc)
    if state.get("reranker_id"):
        events = client.memories.retrieve(
            message="What are state, nodes and edges in StateGraph?",
            space_ids=[state["spaces"]["langgraph"]],
            reranker_id=state["reranker_id"], requested_size=5, max_results=5,
            chronological_resort=False, fetch_memory=True, fetch_memory_content=False, stream=False,
        )
        report["rerank_without_llm"] = {
            "chunks": len(documents_from_events(events)),
            "statuses": [e.status.model_dump(mode="json", exclude_none=True) for e in events if e.status],
            "stages": [e.result_set_boundary.stage_name for e in events
                       if e.result_set_boundary and e.result_set_boundary.kind == "BEGIN"],
            "configured_llms": len(client.llms.list()),
        }
    counts = {name: len(list(client.memories.list(space_id=id))) for name, id in state["spaces"].items()}
    report["corpus_memory_counts"] = counts
try:
    with Goodmem(base_url=settings.base_url, api_key="gm_invalid_probe_key", timeout=30) as bad:
        GoodMemRetriever(bad, state["spaces"]["langgraph"]).invoke("state")
    report["invalid_key_rejected"] = False
except AuthenticationError:
    report["invalid_key_rejected"] = True
report["passed"] = all(report[k] for k in ["empty_space_returns_empty", "missing_space_detected",
                                           "invalid_key_rejected"])
save_json(Path(".runtime/probes.json"), report)
print(report)
if not report["passed"]:
    raise SystemExit(1)
