# Application configuration and small .env loader.
# The project intentionally avoids a heavy settings framework. A single frozen
# dataclass keeps paths, model names, chunk sizes, and generation settings easy
# to inspect and defend in an interview.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REFUSAL_MESSAGE = "Not found in the provided documents."


# Load KEY=VALUE pairs without printing secrets or overriding env vars.
def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Runtime settings for corpus paths, embeddings, vector store, and LLM.
@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    sources_path: Path
    raw_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    llm_model: str
    llm_temperature: float
    chunk_tokens: int
    overlap_tokens: int

    # Build config from the workspace root and optional environment files.
    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "AppConfig":
        root = (root_dir or Path.cwd()).resolve()

        # .env.local is for machine-specific secrets; .env is a fallback.
        load_env_file(root / ".env.local")
        load_env_file(root / ".env")

        return cls(
            root_dir=root,
            sources_path=root / "data" / "sources.json",
            raw_dir=root / "data" / "raw",
            chroma_dir=root / "data" / "chroma",
            collection_name=os.getenv("RAG_COLLECTION", "grounded_pdf_chunks"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            llm_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            chunk_tokens=int(os.getenv("CHUNK_TOKENS", "220")),
            overlap_tokens=int(os.getenv("OVERLAP_TOKENS", "40")),
        )
