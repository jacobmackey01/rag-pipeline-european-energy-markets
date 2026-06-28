"""Chroma vector-store operations for indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb

from rag_pipeline.config import AppConfig
from rag_pipeline.documents import DocumentChunk
from rag_pipeline.embeddings import LocalEmbedder


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned from vector search with its source metadata."""

    id: str
    text: str
    source: str
    title: str
    source_url: str
    chunk_index: int
    page_start: int
    page_end: int
    distance: float

    @property
    def citation_label(self) -> str:
        """Human-readable label used in CLI retrieval previews."""
        return f"{self.source}, page {self.page_start}, chunk {self.chunk_index}"


def get_client(config: AppConfig):
    """Create a persistent Chroma client rooted in data/chroma."""
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.chroma_dir))


def get_collection(config: AppConfig):
    """Open the chunk collection using cosine distance for semantic similarity."""
    client = get_client(config)
    return client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(config: AppConfig) -> None:
    """Drop the existing collection so ingestion can rebuild it from scratch."""
    client = get_client(config)
    try:
        client.delete_collection(config.collection_name)
    except Exception:
        pass


def index_chunks(
    config: AppConfig,
    chunks: list[DocumentChunk],
    embedder: LocalEmbedder | None = None,
    batch_size: int = 64,
) -> int:
    """Embed chunks in batches and upsert text, vectors, and metadata into Chroma."""
    embedder = embedder or LocalEmbedder(config.embedding_model)
    collection = get_collection(config)

    for start in range(0, len(chunks), batch_size):
        # Batch embedding keeps memory use modest while avoiding one call per chunk.
        batch = chunks[start : start + batch_size]
        embeddings = embedder.embed([chunk.text for chunk in batch])
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            embeddings=embeddings,
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
        )

    return len(chunks)


def _metadata_value(metadata: dict[str, Any], key: str, default: str | int) -> Any:
    """Read Chroma metadata defensively because values can be missing or None."""
    value = metadata.get(key, default)
    return default if value is None else value


def retrieve(
    config: AppConfig,
    question: str,
    top_k: int = 4,
    embedder: LocalEmbedder | None = None,
) -> list[RetrievedChunk]:
    """Embed a question and return the top-k nearest stored chunks."""
    embedder = embedder or LocalEmbedder(config.embedding_model)
    collection = get_collection(config)

    # Query and documents use the same embedding model, so distances are comparable.
    query_embedding = embedder.embed([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        retrieved.append(
            RetrievedChunk(
                id=chunk_id,
                text=text,
                source=str(_metadata_value(metadata, "source", "")),
                title=str(_metadata_value(metadata, "title", "")),
                source_url=str(_metadata_value(metadata, "source_url", "")),
                chunk_index=int(_metadata_value(metadata, "chunk_index", -1)),
                page_start=int(_metadata_value(metadata, "page_start", -1)),
                page_end=int(_metadata_value(metadata, "page_end", -1)),
                distance=float(distance),
            )
        )
    return retrieved
