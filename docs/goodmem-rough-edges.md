# GoodMem migration findings

**Update for the local integration review:** the shared `goodmem-langchain` package now implements document normalization and ingestion waiting, and the application uses it. The results below describe the original committed adaptation. The [new assessment](langchain-integration-review.md) records the proposed library fixes, smaller app, 16/16 retrieval checks, and 7/8 agent checks (one model-generated link to a page that was not retrieved). Neither set of results proves a quality improvement over Chroma.

The migration works against a real GoodMem instance. The application uses the published `goodmem==0.1.34` Python SDK and a pinned `server-v1.0.311` container. Cohere `embed-v4.0` supplies embeddings, `rerank-v3.5` supplies optional reranking, and the application uses `command-a-03-2025` for chat. **No LLM is registered in GoodMem.** Groq remains configurable as the upstream chat default; the recorded live run used Cohere.

The four notebook files and their original Git history are retained. GoodMem owns whole-document storage, chunking, embedding, retrieval, and reranking. LangGraph still owns the agent's decisions, grading, drafting, and follow-up lookups.

## Observed results

| Check | Result |
| --- | --- |
| Fresh local Compose startup, bootstrap, embedder/reranker registration | Passed |
| Six official pages, split between two GoodMem spaces | All indexed to `COMPLETED` |
| Repeated setup / notebook ingestion | Still three memories per space |
| Eight retrieval questions, plain vector search | 8/8 source-and-keyword checks passed |
| Same eight questions, with reranking | 8/8 passed |
| Both agents: direct, single-space, cross-space, sequential lookup | 8/8 passed in final evaluation |
| Sequential case | Two retrieval rounds, one tool per documentation space, for each agent |
| Empty space / missing space / invalid key | Empty result / HTTP 404 / authentication rejection |
| Reranking without a GoodMem LLM | Five reranked chunks; zero configured LLMs |
| Notebook execution | All four executed in fresh kernels; see notebook results artifact |

The recorded median retrieval latency was 0.158 seconds without reranking and 0.333 seconds with it. The hand-built graph took 6.453–18.990 seconds on the documentation cases; ReAct took 5.202–9.060 seconds. These are single-run observations, affected by provider/network variability and concurrent local work. Plain search returns five candidates. Reranking selects five from twenty, so this is not a controlled reranking latency comparison.

The suite checks retrieved source coverage, space isolation, expected keywords, completed tool calls, citation provenance, and retrieval rounds. It does not grade every answer claim for entailment. Displayed answers include a deterministic list of sources consulted; raw model answers and their inline citations are recorded separately.

Artifacts: [final evaluation](validation/evaluation.json), [boundary probes](validation/probes.json), [notebook execution](validation/notebooks.json). The [initial evaluation](validation/initial-evaluation.json) is retained to show the problems that prompted changes.

## 1. Reranking succeeds without an LLM, but emits a misleading notice

**Reproduced against the live server.** A request with a reranker and no `llm_id` returned a valid `rerank` result set containing five scored chunks, alongside this status:

```json
{
  "code": "FEATURE_DISABLED",
  "message": "Abstract reply generation disabled: no LLM configured. Add 'llm_id' parameter to enable AI-generated summaries.",
  "details": {"feature": "summarization", "required_param": "llm_id"}
}
```

Reranking itself is supported and worked. My first adapter treated every noninformational status as a retrieval failure; that was too strict for this notice. It now ignores only the structured summarization-disabled notice, plus the SDK's capability-inference notice. Embedder failures, partial vector searches, and reranker failures still surface as errors. A regression test covers this distinction.

Product improvement: suppress the summarization notice when only reranking was requested, or give informational statuses an explicit severity. A caller should not be prompted to configure an LLM for an operation that does not require one.

## 2. Scores have different meanings across retrieval stages

**Observed in live responses.** The first plain vector response had scores `-0.622, -0.522, -0.511, -0.506, -0.492`. Reranked results use a different scoring scale. The generated SDK's `ChunkReference.relevance_score` description says “0.0 to 1.0,” which does not match those plain-search values.

The adapter treats scores as opaque and preserves GoodMem's returned ordering. It does not sort across stages or impose a universal relevance threshold.

Product improvement: document the scoring direction and scale for each stage, and correct the generated field description.

## 3. Cited retrieval requires joining separate event types

**Exercised in live retrieval and unit tests.** The retrieval endpoint returns an event stream even with `stream=False`; the SDK materializes a list of events. Memory metadata arrives in `memory_definition`, while text and score arrive under `retrieved_item.chunk.chunk`. A chunk alone is not a ready-to-use cited LangChain document.

The adapter explicitly requests `fetch_memory=True`, joins by memory UUID in two passes, deduplicates chunks, and fails if citation metadata is missing. Unit tests include definitions arriving after chunks. `fetch_memory_content=False` avoids shipping original document bodies into every retrieval result.

Product improvement: a supported helper returning text, score, memory metadata, and source together would make common RAG integrations smaller.

## 4. Upload success is not indexing completion

**Exercised during ingestion.** Creation returns before indexing completes. The app must poll each memory's `processing_status`, stop on failure, and enforce a timeout before presenting the corpus as ready. The adapter does this, and tests cover pending, processing, completed, failed, and timed-out memories.

Stable IDs and content hashes make reruns repeatable. Replaced document versions are removed only after the replacement reaches `COMPLETED`.

Product improvement: an SDK `create_and_wait` or `wait_until_ready` helper with processing diagnostics would remove common application glue. The current API contract is workable; this is convenience friction rather than a retrieval defect.

## 5. A source URL does not ingest a web page

`original_content_ref` records provenance; it does not fetch the content. This is explicit in the SDK documentation, but easy to mistake for a URL ingestion feature when migrating a loader-based tutorial.

The app fetches each official Markdown page, falls back to article HTML if needed, checks the HTTP response, then uploads the actual content with the canonical source URL in metadata. GoodMem handles chunking and embedding after upload.

Product improvement: a clearly named source-reference field or a separate URL ingestion convenience method would make the distinction easier to discover.

## 6. Bootstrap, connection details, and resource identity need application state

The Python package used here is `goodmem`, not the separate legacy `goodmem-client` package. REST and CLI/gRPC URLs also differ. The quickstart pins the package and server, provides a dedicated loopback REST port, and stores initialized credentials separately from nonsecret resource state. Saved credentials are bound to their endpoint so a server URL change does not silently reuse another server's key.

GoodMem deduplicates equivalent embedder configurations in addition to checking client-supplied IDs. A second namespace should reuse an existing embedder ID rather than assume a different UUID permits an equivalent model registration. The live probe records this behavior.

A smaller observed metadata inconsistency: `/v1/system/info` returned `version=server-v1.0.311` and `git_describe=server-v1.0.311`, but all three numeric version fields were zero. Validation records the full version string and commit rather than using the numeric fields.

## Agent and upstream issues, separate from GoodMem

The upstream explicit graph can hit its hop limit immediately after requesting a tool and end without answering that call. It also grades/generates from only the last tool result if several tools run in a round. The adaptation completes requested calls, caps retrieval rounds before asking for more tools, grades all results from the latest round, and drafts from accumulated evidence.

The first live evaluation exposed vague search queries and a ReAct answer that omitted inline citations. Search guidance now asks for focused queries and dependent lookups. The adapter adds a clearly labeled list of sources consulted independently of the model's formatting. This preserves provenance; it does not magically verify every generated sentence.

One initial sequential test was flawed: it asked the LangGraph overview to name an agent constructor, when that page only recommends the higher-level framework. The corrected case first identifies the framework and then finds its constructor in that framework's docs. The initial and final evaluations therefore use different sequential questions; they are debugging evidence, not a before/after quality benchmark.

## What “better” remains unproven

This migration demonstrates persistent, reusable server-managed retrieval with working reranking and source metadata. It removes local Chroma and BGE-M3 model management from the notebooks.

It does not establish superior retrieval or answer quality over upstream. A fair comparison would hold the corpus snapshot, extraction, chunking, embedding model, chat model, prompts, and query set fixed, then measure retrieval relevance, claim support, cost, and latency across both backends. The current run changes the embedder and prefers Markdown extraction, so an improvement cannot be attributed to GoodMem alone.

No GoodMem server fixes were made or issues posted as part of this adaptation. The workarounds and reproducers are contained in this repository.
