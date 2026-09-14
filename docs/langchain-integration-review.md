# Local adoption of the GoodMem LangChain integration

The demo consumes the sibling integration's clean 0.2 development API. Both repositories remain local and uncommitted for review.

The shared library supplies `GoodMemRetriever` and `wait_for_memory`. The demo uses LangChain's standard `create_retriever_tool`, adding its own collection descriptions and citation formatting. Source loading, hashing, replacement policy, namespace provisioning and graph workflows stay in the application. There are no demo collection names or prompts in the library.

The published 0.1.0 integration already provided agent tools and reranking without an LLM. The improvement is standard Document retrieval, source metadata joins, explicit indexing readiness and using the official SDK throughout. The clean break removes the compatibility client, JSON envelopes, deprecated argument handling and name-based space reuse. See the [current shared integration review](../../goodmem-langchain/docs/integration-review.md) and [migration notes](../../goodmem-langchain/CHANGELOG.md).

## Size

The application's retrieval module is 96 → 27 lines. It includes the seven lines of formatting configuration formerly hidden in the removed factory. All production application code is 739 → 656 lines (11.2% smaller). The integration is 1,994 → 824 lines (58.7% smaller) against the original release, including its new modules. Combined, that is 1,253 fewer lines. These counts include docstrings and blanks; [statement counts and per-file measurements](validation/shared-integration/code-size.json) are also recorded.

## Validation

For the clean break:

- Integration: 37 unit tests pass on Python 3.10 and 3.13; a live workflow exercises all eleven tools and plain/reranked Document tools without an LLM.
- Application: 12 tests pass with the standard LangChain factory.
- Live same-event comparison: 16/16 preserve the original text, metadata, scores and order. [Parity report](validation/shared-integration/contract-parity.json).

The earlier [full evaluation](validation/shared-integration/evaluation.json) and [fresh-kernel notebook runs](validation/shared-integration/notebooks.json) are historical adoption evidence, not reruns of this cleanup. Retrieval acceptance passed 16/16, both changed notebooks passed, and full agent acceptance passed 7/8. The failing ReAct case invented an additional `/langgraph/persistence` citation; the retrieved artifact contained only `/langgraph/graph-api`. That generation limitation remains recorded. These changes do not establish improved answer quality.

The older [published-package investigation](validation/native-integration/review.json) is retained as historical evidence.

## Review state

The dependency remains `langchain-goodmem==0.2.0.dev0` from `../goodmem-langchain`. Use sibling checkouts with `uv sync --locked`. Replace the local path with an approved release or immutable commit before publishing the app update. Nothing should be committed, pushed or published before user review.
