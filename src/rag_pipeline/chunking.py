"""Text chunking helpers used before embedding PDF content.

The important design choice here is preserving original source text. We use
token offsets to choose chunk boundaries, then slice the original string, so
numeric evidence like 5.25%, 2025-2026, and bidding-zone names is not rebuilt
or accidentally changed before it is embedded or cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Fallback splitter used in unit tests and when no model tokenizer is supplied.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TextChunk:
    """A chunk plus its original character span and token count."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int


def tokenize(text: str) -> list[str]:
    """Return regex token-like units for lightweight deterministic tests."""
    return TOKEN_PATTERN.findall(text)


def _token_spans(text: str) -> list[re.Match[str]]:
    """Return regex matches so fallback chunks can keep source character offsets."""
    return list(TOKEN_PATTERN.finditer(text))


def _chunks_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    """Turn token offsets into overlapping chunks while slicing original text."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    offsets = [(start, end) for start, end in offsets if end > start]
    if not offsets:
        return []

    chunks: list[TextChunk] = []
    step = chunk_tokens - overlap_tokens
    start = 0
    index = 0

    while start < len(offsets):
        # Convert token-window positions back into source-text character spans.
        end = min(start + chunk_tokens, len(offsets))
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]

        # Strip only leading/trailing whitespace from the selected source span.
        raw_chunk = text[start_char:end_char]
        chunk = raw_chunk.strip()
        if chunk:
            leading_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            trailing_trim = len(raw_chunk) - len(raw_chunk.rstrip())
            chunks.append(
                TextChunk(
                    text=chunk,
                    chunk_index=index,
                    start_char=start_char + leading_trim,
                    end_char=end_char - trailing_trim,
                    token_count=end - start,
                )
            )
            index += 1
        if end >= len(offsets):
            break
        start += step

    return chunks


def chunk_text(text: str, chunk_tokens: int = 220, overlap_tokens: int = 40) -> list[TextChunk]:
    """Chunk text with regex token-like offsets, preserving the original text."""
    spans = _token_spans(text)
    offsets = [(span.start(), span.end()) for span in spans]
    return _chunks_from_offsets(text, offsets, chunk_tokens, overlap_tokens)


def chunk_text_with_tokenizer(
    text: str,
    tokenizer,
    chunk_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[TextChunk]:
    """Chunk text with a Hugging Face fast tokenizer's real token offsets.

    This is the ingestion path. It aligns chunk size with the same tokenizer
    family used by the MiniLM embedding model, rather than relying on a rough
    word/punctuation count.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return _chunks_from_offsets(text, offsets, chunk_tokens, overlap_tokens)
