# =============================================================================
# chunking.py — Split long document text into small, overlapping pieces.
#
# WHY CHUNK? The embedding model can only "read" a limited amount of text at
# once, and one vector for a whole 50-page PDF would blur all its detail. So we
# cut each document into small passages ("chunks"); each chunk becomes one
# searchable unit with its own embedding.
#
# THE KEY TRICK here: we decide chunk *boundaries* using token offsets, but we
# build each chunk by slicing the ORIGINAL text. That preserves exact wording —
# figures like 5.25%, year ranges like 2025-2026, and zone codes like PL-DE stay
# intact instead of being mangled by re-joining tokens.
# =============================================================================

from __future__ import annotations

# `re` is Python's regular-expression engine, used for the fallback tokenizer.
import re
# `dataclass` to make a small, typed data-holder (TextChunk) with no boilerplate.
from dataclasses import dataclass


# A regex matching EITHER a run of "word" characters (\w+ = letters/digits/_)
# OR a single non-word, non-space character ([^\w\s] = punctuation). Together
# they chop text into rough "tokens". This is a lightweight stand-in used in
# tests and whenever no real model tokenizer is supplied.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# One chunk of text plus the bookkeeping about where it came from.
@dataclass(frozen=True)
class TextChunk:
    # The chunk's actual text (sliced from the original document).
    text: str
    # Its position in the sequence of chunks (0, 1, 2, ...).
    chunk_index: int
    # Character index in the original text where this chunk starts...
    start_char: int
    # ...and where it ends. These let us map a chunk back to its PDF page later.
    end_char: int
    # How many tokens this chunk spans (informational / for debugging).
    token_count: int


# Split text into the rough token strings themselves (used by unit tests).
def tokenize(text: str) -> list[str]:
    # `findall` returns every substring matching the pattern, in order.
    return TOKEN_PATTERN.findall(text)


# Like tokenize, but return the regex MATCH objects (not just the strings) so
# callers can read each token's character position via .start() / .end().
def _token_spans(text: str) -> list[re.Match[str]]:
    # `finditer` yields match objects; list() materialises them into a list.
    return list(TOKEN_PATTERN.finditer(text))


# The core engine. Given a list of (start_char, end_char) token offsets, group
# them into overlapping windows and slice the original text for each window.
# Both public chunkers below funnel their offsets into this one function.
def _chunks_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    # --- Validate the settings up front, failing loudly on nonsense values. ---
    # A chunk must hold at least one token.
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")
    # Overlap can't be negative.
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    # Overlap must be smaller than the chunk; otherwise the window never moves
    # forward (see `step` below) and we'd loop forever.
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    # Drop zero-width offsets (end <= start). Some tokenizers emit (0,0) for
    # special tokens, and an empty span carries no text.
    offsets = [(start, end) for start, end in offsets if end > start]
    # If there were no real tokens, there's nothing to chunk.
    if not offsets:
        return []

    # The finished chunks we'll return.
    chunks: list[TextChunk] = []
    # How far the window advances each step. With chunk=220, overlap=40 the step
    # is 180 — so each new chunk repeats the last 40 tokens of the previous one.
    step = chunk_tokens - overlap_tokens
    # Index of the first token in the current window.
    start = 0
    # Running counter for chunk_index.
    index = 0

    # Slide the window across the token list until everything is covered.
    while start < len(offsets):
        # The window's end token index (capped so we don't run past the list).
        end = min(start + chunk_tokens, len(offsets))
        # Character position where the FIRST token of the window begins...
        start_char = offsets[start][0]
        # ...and where the LAST token of the window ends. Slicing between these
        # captures every character of the original text in this window.
        end_char = offsets[end - 1][1]

        # Slice the ORIGINAL text — this is the preservation trick. raw_chunk may
        # carry leading/trailing whitespace, which we handle next.
        raw_chunk = text[start_char:end_char]
        # Remove surrounding whitespace for a clean chunk string.
        chunk = raw_chunk.strip()
        # Keep the chunk only if it still has real content after trimming.
        if chunk:
            # How many characters were trimmed off the front...
            leading_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            # ...and off the back, so we can shift the recorded character offsets
            # to match the trimmed text exactly (keeps page mapping accurate).
            trailing_trim = len(raw_chunk) - len(raw_chunk.rstrip())
            # Build and store the chunk record.
            chunks.append(
                TextChunk(
                    text=chunk,
                    chunk_index=index,
                    # Nudge the offsets inward by the trimmed amounts.
                    start_char=start_char + leading_trim,
                    end_char=end_char - trailing_trim,
                    # Number of tokens actually in this window.
                    token_count=end - start,
                )
            )
            # Only advance the chunk counter when we actually added one.
            index += 1
        # If this window already reached the end of the tokens, we're done.
        if end >= len(offsets):
            break
        # Otherwise slide the window forward by `step` (leaving the overlap).
        start += step

    # Return all the chunks we built.
    return chunks


# Public chunker #1: uses the lightweight regex tokenizer. Handy for unit tests
# and as a fallback when no model tokenizer is available.
def chunk_text(text: str, chunk_tokens: int = 220, overlap_tokens: int = 40) -> list[TextChunk]:
    # Get the regex token match objects.
    spans = _token_spans(text)
    # Convert each match into a (start, end) character-offset pair.
    offsets = [(span.start(), span.end()) for span in spans]
    # Reuse the shared engine to build the chunks.
    return _chunks_from_offsets(text, offsets, chunk_tokens, overlap_tokens)


# Public chunker #2 (the real ingestion path): uses a Hugging Face "fast"
# tokenizer so chunk sizes are measured in the SAME tokens the embedding model
# uses. That means a "220-token" chunk really is ~220 of the model's tokens,
# safely under all-MiniLM-L6-v2's 256-token input limit (so nothing is truncated).
def chunk_text_with_tokenizer(
    text: str,
    tokenizer,
    chunk_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[TextChunk]:
    # Run the tokenizer over the whole text, asking for:
    encoded = tokenizer(
        text,
        # ...no [CLS]/[SEP] special tokens (we only want real content tokens),
        add_special_tokens=False,
        # ...the character offsets of every token (the (start,end) we need),
        return_offsets_mapping=True,
        # ...and silence the "sequence longer than model max" warning, which is
        # expected here since we deliberately tokenize the full document.
        verbose=False,
    )
    # Pull out the list of (start_char, end_char) pairs, forcing them to ints.
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    # Reuse the same shared engine to build the chunks.
    return _chunks_from_offsets(text, offsets, chunk_tokens, overlap_tokens)
