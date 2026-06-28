"""Prompt construction and citation validation for grounded answers."""

from __future__ import annotations

import re

from rag_pipeline.config import REFUSAL_MESSAGE
from rag_pipeline.store import RetrievedChunk


# Match bare PDF filenames without swallowing preceding prose such as "According to".
PDF_PATTERN = re.compile(r"(?<![\w.-])([\w.-]+\.pdf)(?![\w.-])", re.IGNORECASE)


# This instruction is the core anti-hallucination contract for generation.
GROUNDING_INSTRUCTION = (
    "Answer ONLY using the provided context. Cite the source filename for each claim. "
    f"If the answer is not in the context, reply exactly: '{REFUSAL_MESSAGE}' "
    "Do not use outside knowledge."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks with source tags before passing them to the LLM."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            "\n".join(
                [
                    f"[{index}] Source: {chunk.source}",
                    f"Title: {chunk.title}",
                    f"Page: {chunk.page_start}",
                    f"Chunk: {chunk.chunk_index}",
                    "Text:",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Combine retrieved context and the user's question into one prompt."""
    return (
        "Use the context below to answer the question.\n\n"
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def extract_cited_sources(answer: str) -> set[str]:
    """Return every PDF filename cited by the model answer."""
    return {match.group(1).strip() for match in PDF_PATTERN.finditer(answer)}


def citation_check(answer: str, retrieved: list[RetrievedChunk]) -> tuple[bool, str]:
    """Ensure cited filenames came from the chunks the model was given."""
    if answer.strip() == REFUSAL_MESSAGE:
        return True, "Refusal answer does not require citations."

    cited = extract_cited_sources(answer)
    retrieved_sources = {chunk.source for chunk in retrieved}

    if not cited:
        return False, "No PDF filename citation found in the answer."

    unknown = cited - retrieved_sources
    if unknown:
        return False, f"Cited source(s) not present in retrieved context: {sorted(unknown)}"

    return True, "All cited PDF filenames were present in retrieved context."
