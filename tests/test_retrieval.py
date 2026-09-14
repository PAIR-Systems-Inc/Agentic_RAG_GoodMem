from types import SimpleNamespace as NS

import pytest

from goodmem_rag.retrieval import GoodMemRetriever, RetrievalFailure, documents_from_events


def event(**kwargs):
    return NS(memory_definition=None, retrieved_item=None, status=None, **kwargs)


def definition():
    e = event()
    e.memory_definition = NS(memory_id="memory-1", space_id="space-1",
                             metadata={"source": "https://example.org/docs", "title": "Docs"})
    return e


def chunk():
    e = event()
    e.retrieved_item = NS(chunk=NS(
        chunk=NS(chunk_id="chunk-1", memory_id="memory-1", chunk_text="retrieved evidence"),
        relevance_score=-0.82,
    ))
    return e


def status(code, details=None):
    e = event()
    e.status = NS(code=code, message="Diagnostic", details=details)
    return e


def test_definitions_can_follow_chunks_and_duplicate_hits_are_removed():
    docs = documents_from_events([chunk(), definition(), chunk()])
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "https://example.org/docs"
    # Base vector scores are not normalized probabilities; do not apply a 0..1 cutoff.
    assert docs[0].metadata["score"] == -0.82


@pytest.mark.parametrize("code", ["EMBEDDER_FAILED", "RERANKING_FAILED", "SPACE_NOT_FOUND",
                                 "VECTOR_SEARCH_PARTIAL", "FEATURE_DISABLED"])
def test_status_failures_are_not_misrepresented_as_empty_results(code):
    with pytest.raises(RetrievalFailure, match=code):
        documents_from_events([definition(), chunk(), status(code)])


def test_rerank_only_summary_notice_is_informational():
    notice = status("FEATURE_DISABLED", {"feature": "summarization", "required_param": "llm_id"})
    assert len(documents_from_events([notice, definition(), chunk()])) == 1


def test_missing_metadata_does_not_produce_an_uncited_answer():
    with pytest.raises(RetrievalFailure, match="Missing memory metadata"):
        documents_from_events([chunk()])


def test_empty_space_is_a_valid_empty_result():
    assert documents_from_events([]) == []


def test_retrieval_uses_only_selected_space_and_requests_metadata():
    requests = []

    def retrieve(**kwargs):
        requests.append(kwargs)
        return [definition(), chunk()]

    client = NS(memories=NS(retrieve=retrieve))
    GoodMemRetriever(client, "space-1", reranker_id="reranker-1").invoke("question")
    assert requests[0]["space_ids"] == ["space-1"]
    assert requests[0]["fetch_memory"] is True
    assert requests[0]["fetch_memory_content"] is False
    assert requests[0]["chronological_resort"] is False
    assert requests[0]["max_results"] == 5
    assert requests[0]["requested_size"] == 20


def test_empty_query_is_rejected_before_a_network_call():
    with pytest.raises(ValueError, match="empty"):
        GoodMemRetriever(None, "space").invoke(" ")
