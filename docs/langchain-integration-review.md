# Published GoodMem LangChain integration

The demo installs [langchain-goodmem 0.2.0 from PyPI](https://pypi.org/project/langchain-goodmem/0.2.0/). Its lockfile contains the public wheel and source-distribution checksums; no sibling checkout or editable integration dependency is required. The integration source and [release notes](https://github.com/PAIR-Systems-Inc/goodmem-langchain/releases/tag/v0.2.0) are published, and the [GoodMem documentation guide](https://docs.goodmem.ai/docs/integrations/agent-frameworks/langchain/) is live.

The shared library supplies `GoodMemRetriever`, Document metadata joins, native metadata filters and `wait_for_memory`. LangChain's standard `create_retriever_tool` exposes a query-only search tool bound to each collection. The demo filters for `application=agentic-rag-goodmem`; model tool calls cannot change that filter or the configured space.

Setup uses the SDK for deterministic memory IDs, source references and replacement policy, and the shared readiness helper for indexing. URL loading, content hashing, namespace provisioning and graph workflows belong to the application. There are no demo collection names or prompts in the shared library. The library also provides `add_documents` for ordinary LangChain Documents; it preserves metadata and optional UUIDs, batches writes, and waits by default.

The clean 0.2 API removes the compatibility client, JSON envelopes, deprecated argument handling and automatic space reuse. Low-level retrieval tools preserve SDK events, including statuses and chunks. The Document retriever raises on incomplete retrieval. Native async remains future work; LangChain's executor supports async invocation today. See the [migration notes](https://github.com/PAIR-Systems-Inc/goodmem-langchain/blob/v0.2.0/CHANGELOG.md).

## Size

The application's retrieval module is 96 → 28 lines. All production application code is 739 → 656 lines (11.2% smaller). The integration is 1,994 → 915 lines (54.1% smaller), including Document ingestion and filtering. Combined, that is 1,162 fewer lines. These counts include docstrings and blanks; [statement counts and per-file measurements](validation/shared-integration/code-size.json) are also recorded. The integration README has 372 words including code.

## Release validation

- Integration: 114 offline tests pass on Python 3.10 and 3.13, including 71 inherited LangChain standard checks. Ten live tests cover all eleven tools, Document ingestion, metadata filters, and plain/reranked retrieval without an LLM. GitHub CI and trusted PyPI publishing passed.
- Fresh public installation: the downloaded wheel checksum matches PyPI, and all 17 Python source files match the release and installed package. The ingestion/filtering/reranking smoke example and all five documentation snippets pass against live temporary spaces. [Package verification](validation/shared-integration/release-0.2.0/published-package.json).
- Application: 12 unit tests pass with the public package. Setup indexes the six official pages in two spaces. The acceptance suite exercises the application's actual filtered tools: retrieval passes 16/16, and the graph and ReAct agents pass 8/8, including dependent two-round searches. [Evaluation](validation/shared-integration/release-0.2.0/evaluation.json).
- Notebooks: all four pass in fresh kernels using the published integration. Repeated setup leaves three indexed memories per space and zero registered GoodMem LLMs. [Notebook runs](validation/shared-integration/release-0.2.0/notebooks.json) and [verification](validation/shared-integration/release-0.2.0/verification.json).
- Same-event comparison: the published normalizer preserves the original text, metadata, scores and order in 16/16 cases. [Parity report](validation/shared-integration/release-0.2.0/contract-parity.json).
- Public docs: the deployed page contains the 0.2 ingestion, retrieval and reranking guide; the site's build, link and search-index checks pass. [Deployment verification](validation/shared-integration/release-0.2.0/documentation.json).

The earlier [development evaluation](validation/shared-integration/evaluation.json) remains unchanged as historical evidence. That run passed 7/8 agent cases because ReAct invented an additional `/langgraph/persistence` citation not present in retrieved artifacts. The release run passed without changing the prompts or relaxing citation checks; this does not establish that the generation limitation is fixed. No controlled comparison against Chroma has been run.

The older [published 0.1 package investigation](validation/native-integration/review.json) is also retained as historical evidence. Current server filter-expression rough edges are recorded in the [migration findings](goodmem-rough-edges.md).
