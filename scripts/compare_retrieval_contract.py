"""Compare old and shared normalization on the same live events, without model calls."""

import ast
import json
import subprocess
from pathlib import Path

from langchain_goodmem.retrievers import documents_from_events

from goodmem_rag.config import Settings
from goodmem_rag.evaluation import RETRIEVAL_CASES

BASE = "7a1640b38b32e8402561f30a8ad028dc02366d81"
source = subprocess.check_output(["git", "show", f"{BASE}:goodmem_rag/retrieval.py"], text=True)
# Load only the baseline normalizer; neither its tools nor its retriever are executed.
module = ast.parse(source)
module.body = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
               or getattr(node, "name", "") in {"RetrievalFailure", "documents_from_events"}]
baseline = {}
exec(compile(module, "baseline_retrieval.py", "exec"), baseline)
settings = Settings.from_env()
state = settings.state()
rows = []
with settings.client() as client:
    for rerank in [False, True]:
        for name, collection, query, _, _ in RETRIEVAL_CASES:
            options = {"reranker_id": state["reranker_id"], "max_results": 5,
                       "chronological_resort": False} if rerank else {}
            events = client.memories.retrieve(
                message=query, space_ids=[state["spaces"][collection]], requested_size=20 if rerank else 5,
                fetch_memory=True, fetch_memory_content=False, stream=False, **options,
            )
            old = baseline["documents_from_events"](events)
            new = documents_from_events(events)
            identical = [(d.page_content, d.metadata) for d in old] == [
                (d.page_content, d.metadata) for d in new
            ]
            rows.append({"case": name, "rerank": rerank, "same_text_metadata_scores_order": identical,
                         "new_document_ids_match_chunks": all(d.id == d.metadata["chunk_id"] for d in new)})
report = {"baseline_commit": BASE, "cases": rows,
          "passed": all(r["same_text_metadata_scores_order"] and r["new_document_ids_match_chunks"] for r in rows)}
path = Path("docs/validation/shared-integration/contract-parity.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(f"{sum(r['same_text_metadata_scores_order'] for r in rows)}/{len(rows)} identical retrieval contracts")
if not report["passed"]:
    raise SystemExit(1)
