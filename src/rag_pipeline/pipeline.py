"""High-level orchestration functions used by the CLI."""

from __future__ import annotations

from rag_pipeline.config import AppConfig
from rag_pipeline.documents import load_pdf_chunks
from rag_pipeline.embeddings import LocalEmbedder
from rag_pipeline.generation import answer_from_context
from rag_pipeline.store import index_chunks, reset_collection, retrieve


def build_index(config: AppConfig, reset: bool = False) -> int:
    """Load PDF chunks and index them in Chroma."""
    if reset:
        reset_collection(config)
    chunks = load_pdf_chunks(config)
    if not chunks:
        raise RuntimeError(
            f"No PDF chunks found in {config.raw_dir}. Run `rag-pipeline download` first."
        )
    return index_chunks(config, chunks)


def ask_question(config: AppConfig, question: str, top_k: int = 4) -> dict[str, object]:
    """Retrieve evidence for a question, then generate a grounded answer."""
    embedder = LocalEmbedder(config.embedding_model)
    chunks = retrieve(config, question, top_k=top_k, embedder=embedder)
    result = answer_from_context(config, question, chunks)
    result["retrieved_chunks"] = chunks
    return result
