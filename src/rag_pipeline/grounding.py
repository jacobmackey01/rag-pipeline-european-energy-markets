"""Build grounded prompts and verify cited PDF filenames."""

from __future__ import annotations

import re

from rag_pipeline.config import REFUSAL_MESSAGE
from rag_pipeline.store import RetrievedChunk


# Regex for PDF filenames in generated answers:
# - the negative lookbehind prevents a match from starting inside a longer token;
# - the capture group selects filename characters ending in ".pdf";
# - the negative lookahead prevents a match from ending inside a longer token.
# Matching is case-insensitive so ".PDF" is also accepted.
PDF_PATTERN = re.compile(r"(?<![\w.-])([\w.-]+\.pdf)(?![\w.-])", re.IGNORECASE)


# Grounding instruction shared with the refusal validation.
GROUNDING_INSTRUCTION = (
    "Answer ONLY using the provided context. Cite the source filename for each claim. "
    f"If the answer is not in the context, reply exactly: '{REFUSAL_MESSAGE}' "
    "Do not use outside knowledge."
)


# Format the retrieved chunks into a readable, source-tagged block for the prompt.
def format_context(chunks: list[RetrievedChunk]) -> str:
    # We'll build a list of formatted chunk strings.
    parts: list[str] = []
    # Number the chunks from 1 for readability inside the prompt.
    for index, chunk in enumerate(chunks, start=1):
        # Each chunk shows its number, source file, title, page, chunk index, and text.
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
    # Join the chunks with a clear "---" separator between them.
    return "\n\n---\n\n".join(parts)


# Assemble the full user prompt: the context block followed by the question.
def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "Use the context below to answer the question.\n\n"
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


# Pull every PDF filename the model cited out of its answer text.
def extract_cited_sources(answer: str) -> set[str]:
    # `finditer` finds all matches; group(1) is the captured filename; we strip
    # whitespace and use a set to de-duplicate.
    return {match.group(1).strip() for match in PDF_PATTERN.finditer(answer)}


# Verify that every source the model cited was actually among the chunks we gave
# it. This catches a model "citing" a document it invented or never saw.
def citation_check(answer: str, retrieved: list[RetrievedChunk]) -> tuple[bool, str]:
    # A pure refusal does not require a citation.
    if answer.strip() == REFUSAL_MESSAGE:
        return True, "Refusal answer does not require citations."

    # What the model cited...
    cited = extract_cited_sources(answer)
    # ...versus what it was actually given.
    retrieved_sources = {chunk.source for chunk in retrieved}

    # A non-refusal answer without a citation fails the grounding check.
    if not cited:
        return False, "No PDF filename citation found in the answer."

    # Any cited file not in the retrieved set is a fabricated/incorrect citation.
    unknown = cited - retrieved_sources
    if unknown:
        return False, f"Cited source(s) not present in retrieved context: {sorted(unknown)}"

    # Otherwise every citation is legitimate.
    return True, "All cited PDF filenames were present in retrieved context."
