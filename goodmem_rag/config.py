"""Environment settings and endpoint-bound local state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from goodmem import Goodmem


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    # Create private from the outset, including when the file contains a bootstrap key.
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(value, out, indent=2)
        out.write("\n")
    temp.replace(path)


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://localhost:8088"
    api_key: str = field(default="", repr=False)
    namespace: str = "agentic-rag-goodmem"
    runtime: Path = Path(".runtime")
    timeout: float = 120
    ingest_timeout: float = 600

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(Path(".env"))
        url = os.getenv("GOODMEM_BASE_URL", "http://localhost:8088").strip().rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path:
            raise ValueError("GOODMEM_BASE_URL must be the REST server root, without /v1 or /mcp")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("GOODMEM_BASE_URL must not contain credentials, query, or fragment")
        return cls(
            base_url=url,
            api_key=os.getenv("GOODMEM_API_KEY", "").strip(),
            namespace=os.getenv("GOODMEM_NAMESPACE", "agentic-rag-goodmem"),
            runtime=Path(os.getenv("GOODMEM_RUNTIME_DIR", ".runtime")),
            timeout=float(os.getenv("GOODMEM_TIMEOUT", "120")),
            ingest_timeout=float(os.getenv("GOODMEM_INGEST_TIMEOUT", "600")),
        )

    def key(self) -> str:
        if self.api_key:
            return self.api_key
        saved = read_json(self.runtime / "credentials.json")
        if saved.get("base_url") == self.base_url and saved.get("api_key"):
            return saved["api_key"]
        raise ValueError("Set GOODMEM_API_KEY, or run setup --init for a new local server")

    def client(self) -> Goodmem:
        return Goodmem(base_url=self.base_url, api_key=self.key(), timeout=self.timeout)

    def state(self) -> dict:
        state = read_json(self.runtime / "state.json")
        if state.get("base_url") != self.base_url or state.get("namespace") != self.namespace:
            raise ValueError("No matching corpus state. Run goodmem-rag setup for this server/namespace")
        return state


def chat_model():
    """Keep the upstream Groq default; Cohere supports a single-provider demo."""
    load_dotenv(Path(".env"))
    provider = os.getenv("CHAT_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Set GROQ_API_KEY (or select CHAT_PROVIDER=cohere and COHERE_API_KEY)")
        return ChatGroq(
            model=os.getenv("CHAT_MODEL") or "openai/gpt-oss-120b", temperature=0, max_retries=2
        )
    if provider == "cohere":
        from langchain_cohere import ChatCohere

        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("Set COHERE_API_KEY for CHAT_PROVIDER=cohere")
        return ChatCohere(
            model=os.getenv("CHAT_MODEL") or "command-a-03-2025", temperature=0, timeout_seconds=120
        )
    raise ValueError("CHAT_PROVIDER must be groq or cohere")
