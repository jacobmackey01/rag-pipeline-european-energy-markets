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
from rag_pipeline.documents import DocumentChunk
from rag_pipeline.embeddings import LocalEmbedder


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
    # Cosine DISTANCE from the query (lower = more similar). Chroma returns this.
    distance: float

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
        metadata={"hnsw:space": "cosine"},
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

    # Process chunks in batches of 64 (memory-friendly, and one embed call per
    # batch instead of one per chunk).
    for start in range(0, len(chunks), batch_size):
        # Slice out this batch.
        batch = chunks[start : start + batch_size]
        # Embed all chunk texts in the batch at once.
        embeddings = embedder.embed([chunk.text for chunk in batch])
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


# The RETRIEVAL step: embed the question and fetch the top-k most similar chunks.
def retrieve(
    config: AppConfig,
    question: str,
    top_k: int = 4,
    embedder: LocalEmbedder | None = None,
) -> list[RetrievedChunk]:
    # Same model as indexing — crucial, because two vectors are only comparable
    # if the same model produced them (same coordinate space).
    embedder = embedder or LocalEmbedder(config.embedding_model)
    collection = get_collection(config)

    # Embed the question. `embed` takes a list and returns a list, so we pass
    # [question] and take element [0].
    query_embedding = embedder.embed([question])[0]
    # Ask Chroma for the top_k nearest stored vectors, requesting the text,
    # metadata, and distances back.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns lists-of-lists (one inner list per query). We sent one
    # query, so we take element [0] of each. `.get(..., [[]])` guards missing keys.
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Build typed RetrievedChunk objects from the parallel result lists.
    retrieved: list[RetrievedChunk] = []
    # `zip` walks all four lists together, one result at a time.
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        # Guard against a None metadata dict.
        metadata = metadata or {}
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
            )
        )
    # Return the ranked list (closest first).
    return retrieved
