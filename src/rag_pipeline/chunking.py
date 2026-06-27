from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int


def tokenize(text: str) -> list[str]:
    """Token-like splitter used for deterministic chunk boundaries."""
    return TOKEN_PATTERN.findall(text)


def _token_spans(text: str) -> list[re.Match[str]]:
    return list(TOKEN_PATTERN.finditer(text))


def chunk_text(text: str, chunk_tokens: int = 220, overlap_tokens: int = 40) -> list[TextChunk]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    spans = _token_spans(text)
    if not spans:
        return []

    chunks: list[TextChunk] = []
    step = chunk_tokens - overlap_tokens
    start = 0
    index = 0

    while start < len(spans):
        end = min(start + chunk_tokens, len(spans))
        start_char = spans[start].start()
        end_char = spans[end - 1].end()
        chunk = text[start_char:end_char].strip()
        if chunk:
            chunks.append(TextChunk(text=chunk, chunk_index=index))
            index += 1
        if end >= len(spans):
            break
        start += step

    return chunks
