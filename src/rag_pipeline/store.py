# =============================================================================
# store.py — The vector database layer (ChromaDB): store and search embeddings.
#
# A vector store keeps each chunk's embedding (its meaning-vector) alongside its
# text and metadata, and can quickly answer: "given this query vector, which
# stored vectors are closest?" Closeness here = cosine similarity = similar meaning.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The ChromaDB client library.
import chromadb

from rag_pipeline.config import AppConfig
from rag_pipeline.documents import DocumentChunk, load_sources
from rag_pipeline.embeddings import LocalEmbedder
from rag_pipeline.embedding_context import CONTEXT_VERSION, contextual_text
from rag_pipeline.hybrid import rank_records
from rag_pipeline.retrieval_scope import MissingReportError, OUTLOOK_REFERENCE, source_scope


# One search result: a stored chunk plus how far it was from the query.
@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    title: str
    source_url: str
    chunk_index: int
    page_start: int
    page_end: int
    # Cosine distance; the combined retrieval score determines final ranking.
    distance: float
    retrieval_score: float | None = None

    # A short human-readable label like "report.pdf, page 18, chunk 45", shown in
    # CLI previews so you can eyeball where a chunk came from.
    @property
    def citation_label(self) -> str:
        return f"{self.source}, page {self.page_start}, chunk {self.chunk_index}"


# Create (or reopen) a Chroma client that persists to data/chroma on disk.
def get_client(config: AppConfig):
    # Ensure the storage folder exists.
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    # PersistentClient saves to disk, so the index survives between runs.
    return chromadb.PersistentClient(path=str(config.chroma_dir))


# Get the collection (Chroma's equivalent of a table) that holds our chunks,
# creating it if it doesn't exist yet.
def get_collection(config: AppConfig):
    client = get_client(config)
    # `hnsw:space: cosine` tells Chroma to rank similarity by cosine distance —
    # the right metric for normalised text embeddings.
    return client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine", "embedding_context_version": CONTEXT_VERSION},
    )


# Delete the whole collection so `ingest --reset` can rebuild from a clean slate.
def reset_collection(config: AppConfig) -> None:
    client = get_client(config)
    # Wrapped in try/except because deleting a collection that doesn't exist
    # raises — and "already gone" is a perfectly fine outcome here.
    try:
        client.delete_collection(config.collection_name)
    except Exception:
        pass


# Embed all chunks and write them into Chroma. Returns how many were stored.
def index_chunks(
    config: AppConfig,
    chunks: list[DocumentChunk],
    embedder: LocalEmbedder | None = None,
    batch_size: int = 64,
) -> int:
    # Use the passed-in embedder, or build a default one.
    embedder = embedder or LocalEmbedder(config.embedding_model)
    # Open the collection to write into.
    collection = get_collection(config)
    if collection.count() and (collection.metadata or {}).get("embedding_context_version") != CONTEXT_VERSION:
        raise RuntimeError("Index uses passage-only embeddings. Run `rag-pipeline ingest --reset`.")

    # Process chunks in batches of 64 (memory-friendly, and one embed call per
    # batch instead of one per chunk).
    for start in range(0, len(chunks), batch_size):
        # Slice out this batch.
        batch = chunks[start : start + batch_size]
        # Embed all chunk texts in the batch at once.
        embeddings = embedder.embed([contextual_text(chunk.title, chunk.text) for chunk in batch])
        # `upsert` = insert-or-update by id. Because ids are stable, re-running
        # ingestion overwrites existing rows instead of creating duplicates.
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            embeddings=embeddings,
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
        )

    # Report the total number stored.
    return len(chunks)


# Safely read a metadata value, falling back to a default if it's missing or None.
def _metadata_value(metadata: dict[str, Any], key: str, default: str | int) -> Any:
    # `.get` returns the default if the key is absent...
    value = metadata.get(key, default)
    # ...and we also coerce an explicit None to the default.
    return default if value is None else value


# Retrieve the top-k passages by contextual semantic and lexical ranking.
def retrieve(
    config: AppConfig,
    question: str,
    top_k: int = 4,
    embedder: LocalEmbedder | None = None,
    sources: list[str] | None = None,
    infer_scope: bool = True,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    scope = None
    if sources is not None or (infer_scope and OUTLOOK_REFERENCE.search(question)):
        try:
            scope = source_scope(question, load_sources(config.sources_path), sources)
        except MissingReportError:
            return []
    # Same model as indexing — crucial, because two vectors are only comparable
    # if the same model produced them (same coordinate space).
    embedder = embedder or LocalEmbedder(config.embedding_model)
    collection = get_collection(config)

    if collection.count() and (collection.metadata or {}).get("embedding_context_version") != CONTEXT_VERSION:
        raise RuntimeError("Index uses passage-only embeddings. Run `rag-pipeline ingest --reset`.")

    read_options: dict[str, Any] = {}
    if scope:
        read_options["where"] = {"source": {"$in": scope}}
    records = collection.get(include=["documents", "metadatas", "embeddings"], **read_options)
    if not records["ids"]:
        return []
    query_vector = embedder.embed([question])[0]
    ranked = rank_records(question, query_vector, records)[:top_k]

    # Build typed RetrievedChunk objects from the parallel result lists.
    retrieved: list[RetrievedChunk] = []
    for index, distance, score in ranked:
        chunk_id = records["ids"][index]
        text = records["documents"][index]
        metadata = records["metadatas"][index] or {}
        retrieved.append(
            RetrievedChunk(
                id=chunk_id,
                text=text,
                # Pull each metadata field defensively with a sensible default.
                source=str(_metadata_value(metadata, "source", "")),
                title=str(_metadata_value(metadata, "title", "")),
                source_url=str(_metadata_value(metadata, "source_url", "")),
                chunk_index=int(_metadata_value(metadata, "chunk_index", -1)),
                page_start=int(_metadata_value(metadata, "page_start", -1)),
                page_end=int(_metadata_value(metadata, "page_end", -1)),
                # The cosine distance for this result.
                distance=float(distance),
                retrieval_score=score,
            )
        )
    # Return the ranked list (closest first).
    return retrieved
