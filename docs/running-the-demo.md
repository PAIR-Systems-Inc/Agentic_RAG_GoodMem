# Running the demo

The [README](../README.md) covers the first run and notebook setup. This guide covers other configurations, repeat runs, and troubleshooting.

## Use an existing GoodMem server

Set these values in `.env`, keeping your chat-provider settings:

```dotenv
GOODMEM_BASE_URL=https://your-goodmem-server.example.com
GOODMEM_API_KEY=your-goodmem-key
GOODMEM_EMBEDDER_ID=your-embedder-uuid
```

Use the REST server root, without `/v1` or `/mcp`. The CLI's gRPC address is a different interface. The credential must be allowed to create spaces and ingest documents.

Then load the demo's documentation:

```bash
uv run goodmem-rag setup
```

Leave out `--init` when using an existing server. If you omit `GOODMEM_EMBEDDER_ID`, setup registers the model in `GOODMEM_EMBEDDING_MODEL` (default `embed-v4.0`) with `EMBEDDING_API_KEY` or `COHERE_API_KEY`.

To use an existing reranker, set `GOODMEM_RERANKER_ID`. Otherwise, `setup --with-reranker` registers `rerank-v3.5` using `COHERE_API_KEY`. Reranking needs no GoodMem LLM registration.

## Use Groq for chat

Replace the chat settings in `.env`:

```dotenv
CHAT_PROVIDER=groq
CHAT_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your-groq-key
```

Groq is the original project's chat provider. GoodMem still needs an embedder, and optionally a reranker: keep the Cohere configuration or supply existing GoodMem resource IDs. Changing chat providers does not change the stored documentation.

## Restart, refresh, or inspect retrieval

Restart the local server without rebuilding the corpus:

```bash
docker compose up -d --wait
```

You can then run `ask` or open the RAG notebooks. To fetch the latest versions of the six source pages:

```bash
uv run goodmem-rag setup
```

Setup reuses unchanged documents and waits for changed pages to finish indexing before removing their old versions. A previously selected reranker remains selected when setup runs again.

Inspect the passages directly:

```bash
uv run goodmem-rag search langgraph "How does the add_messages reducer work?"
uv run goodmem-rag search langgraph "How does the add_messages reducer work?" --no-rerank
```

Use `langchain` instead of `langgraph` to search the other collection. `--no-rerank` lets you compare plain retrieval with reranked results.

Stop the local server with `docker compose stop`. The database lives in a Docker volume. The ignored `.runtime` directory stores its initialized API key and the demo's resource IDs; keep that directory with the database when restarting the demo.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv` cannot read the lockfile | [Update uv](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv), then run `uv sync --locked` again. |
| Docker will not start | Make sure Docker Desktop or Docker Engine is running. Use `docker compose logs goodmem db` to inspect startup errors. |
| Port 8088 is occupied | Set both `GOODMEM_PORT=8089` and `GOODMEM_BASE_URL=http://localhost:8089` in `.env`, then start Compose again. Saved keys are tied to the server address; supply that server's key if you change the address after setup. |
| Chat asks for a Groq key after you configured Cohere | Change both `CHAT_PROVIDER` and `CHAT_MODEL` to the README values; the template defaults to Groq. |
| One notebook reports missing imports | Select this project's `.venv` kernel in that notebook. In JupyterLab, select **GoodMem RAG** after registering it as shown in the README. |
| `No matching corpus state` | Run `uv run goodmem-rag setup` for the configured server and namespace. |
| Server already initialized, but no key is available | Supply its existing `GOODMEM_API_KEY`. `--init` does not recover keys from an initialized server. |

## Separate copies of the corpus

`GOODMEM_NAMESPACE` selects a corpus namespace, and `GOODMEM_RUNTIME_DIR` selects where its local state is stored. Use a different runtime directory for each copy. Reuse an existing `GOODMEM_EMBEDDER_ID`: GoodMem rejects equivalent embedder registrations even if their IDs differ.

The default configuration is for local learning. For an application serving untrusted users, separate provisioning from retrieval and give retrieval a restricted GoodMem credential.

## Validation and implementation

The project uses the published `langchain-goodmem` integration for Document retrieval, metadata filters, and indexing readiness. The application keeps its source loading and agent workflows in [goodmem_rag](../goodmem_rag/). See the [integration assessment](langchain-integration-review.md) for the design and code comparison.

Run offline checks:

```bash
uv run pytest -q
uv run ruff check goodmem_rag tests scripts
```

With GoodMem configured and provider credentials available, run the live checks:

```bash
uv run goodmem-rag evaluate
uv run python scripts/probe_goodmem.py
uv run python scripts/run_notebooks.py
```

Evaluation checks source coverage, keywords, completed tool calls, citations, and dependent retrieval rounds for both agents. It records answers and configuration in `.runtime/evaluation.json`. Plain retrieval returns five candidates; reranking selects five from twenty. The checks verify citation provenance, not whether every generated claim follows from its cited passage.

Notebook execution uses a fresh kernel per notebook and saves executed copies under `.runtime/executed/`. Committed notebooks have clean outputs. CI runs offline checks; live checks use provider APIs.
