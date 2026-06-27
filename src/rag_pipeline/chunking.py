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


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def chunk_text(text: str, chunk_tokens: int = 220, overlap_tokens: int = 40) -> list[TextChunk]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    tokens = tokenize(text)
    if not tokens:
        return []

    chunks: list[TextChunk] = []
    step = chunk_tokens - overlap_tokens
    start = 0
    index = 0

    while start < len(tokens):
        window = tokens[start : start + chunk_tokens]
        chunk = detokenize(window)
        if chunk:
            chunks.append(TextChunk(text=chunk, chunk_index=index))
            index += 1
        if start + chunk_tokens >= len(tokens):
            break
        start += step

    return chunks
