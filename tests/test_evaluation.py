from goodmem_rag.evaluation import _citations


def test_citation_provenance_accepts_punctuation_between_bare_urls():
    assert _citations(
        "Sources: https://docs.langchain.com/oss/python/langchain/models, "
        "https://docs.langchain.com/oss/python/langgraph/overview."
    ) == {
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langgraph/overview",
    }
