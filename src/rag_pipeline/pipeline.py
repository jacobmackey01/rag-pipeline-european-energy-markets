"""Orchestrate index construction and grounded question answering."""

from __future__ import annotations

from rag_pipeline.config import AppConfig
from rag_pipeline.documents import load_pdf_chunks
from rag_pipeline.embeddings import LocalEmbedder
from rag_pipeline.generation import answer_from_context
from rag_pipeline.store import index_chunks, reset_collection, retrieve


# Indexing flow: load and chunk PDFs, then store their embeddings in Chroma.
def build_index(config: AppConfig, reset: bool = False) -> int:
    # If asked, wipe the existing collection first for a clean rebuild.
    if reset:
        reset_collection(config)
    # Load and chunk every manifest PDF.
    chunks = load_pdf_chunks(config)
    # Report the likely setup issue when the corpus has no extractable chunks.
    if not chunks:
        raise RuntimeError(
            f"No PDF chunks found in {config.raw_dir}. Run `rag-pipeline download` first."
        )
    # Embed and store the chunks; return how many were indexed.
    return index_chunks(config, chunks)


# Answering flow: retrieve relevant chunks, then generate a grounded answer.
def ask_question(config: AppConfig, question: str, top_k: int = 4) -> dict[str, object]:
    # Build the embedder once and pass it to retrieve (so the model loads once).
    embedder = LocalEmbedder(config.embedding_model)
    # Retrieve the top-k most relevant chunks for the question.
    chunks = retrieve(config, question, top_k=top_k, embedder=embedder)
    # Generate an answer grounded in those chunks (or a refusal).
    result = answer_from_context(config, question, chunks)
    # Attach the retrieved chunks to the result so the CLI can show its sources.
    result["retrieved_chunks"] = chunks
    return result
