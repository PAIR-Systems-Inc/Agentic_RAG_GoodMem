from types import SimpleNamespace as NS

import httpx
import pytest

from goodmem_rag.config import Settings, save_json
from goodmem_rag.ingestion import source_document, wait_for_memory


def test_failed_indexing_and_timeout_are_not_reported_ready():
    failed = NS(memories=NS(get=lambda **kw: NS(processing_status="FAILED")))
    with pytest.raises(RuntimeError, match="FAILED"):
        wait_for_memory(failed, "memory", 1)
    pending = NS(memories=NS(get=lambda **kw: NS(processing_status="PENDING")))
    with pytest.raises(TimeoutError, match="exceeded"):
        wait_for_memory(pending, "memory", 0)


def test_waits_for_completed_indexing():
    states = iter(["PENDING", "PROCESSING", "COMPLETED"])
    client = NS(memories=NS(get=lambda **kw: NS(processing_status=next(states))))
    assert wait_for_memory(client, "memory", 5, interval=0).processing_status == "COMPLETED"


def test_markdown_downloads_preserve_source_and_content_digest():
    def handler(request):
        assert str(request.url) == "https://example.org/docs.md"
        return httpx.Response(200, text="# Demo\n" + "Documentation content. " * 20)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        document = source_document("https://example.org/docs", http)
    assert document["source"] == "https://example.org/docs"
    assert document["title"] == "Demo"
    assert len(document["sha256"]) == 64


def test_http_failures_are_not_ingested_as_documents():
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503))) as http:
        with pytest.raises(httpx.HTTPStatusError):
            source_document("https://example.org/docs", http)


def test_html_fallback_excludes_navigation():
    def handler(request):
        if request.url.path.endswith(".md"):
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=(
            "<title>Docs</title><nav>navigation noise</nav><main>" + "real content " * 30 + "</main>"
        ))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        assert "navigation" not in source_document("https://example.org/docs", http)["text"]


def test_saved_credentials_are_private_and_bound_to_the_endpoint(tmp_path):
    save_json(tmp_path / "credentials.json", {"base_url": "http://one", "api_key": "test-key"})
    assert (tmp_path / "credentials.json").stat().st_mode & 0o777 == 0o600
    assert Settings(base_url="http://one", runtime=tmp_path).key() == "test-key"
    with pytest.raises(ValueError, match="GOODMEM_API_KEY"):
        Settings(base_url="http://two", runtime=tmp_path).key()
