# =============================================================================
# documents.py — From public PDFs to source-traceable, page-tagged chunks.
#
# This module is the front half of the INGESTION pipeline. It:
#   1. downloads the PDFs listed in data/sources.json,
#   2. verifies each file against a pinned SHA256 checksum (reproducibility),
#   3. extracts text page-by-page (remembering where each page starts/ends),
#   4. chunks the text, and
#   5. attaches metadata (filename, URL, page range) so answers can cite sources.
# =============================================================================

from __future__ import annotations

# `hashlib` provides SHA256, used to fingerprint downloaded files.
import hashlib
# `json` reads the sources manifest file.
import json
# `re` tidies up messy whitespace from PDF text extraction.
import re
# `urllib.request` is Python's built-in HTTP client (no extra dependency needed).
import urllib.request
# `dataclass` for typed, boilerplate-free data holders.
from dataclasses import dataclass
# `Path` for cross-platform filesystem paths.
from pathlib import Path
# `Any` means "any type" — used for the tokenizer, whose exact class we'd rather
# not import just for a type hint.
from typing import Any

# `PdfReader` from the pypdf library opens a PDF and exposes its pages/text.
from pypdf import PdfReader

# Our own chunking helpers and the config object.
from rag_pipeline.chunking import TextChunk, chunk_text, chunk_text_with_tokenizer
from rag_pipeline.config import AppConfig


# One entry from data/sources.json: a source PDF we intend to ingest.
@dataclass(frozen=True)
class SourceDocument:
    # The local filename to save/look up the PDF as.
    filename: str
    # A human-readable title (shown in prompts and citations).
    title: str
    # The public URL to download it from.
    url: str
    # Expected SHA256 checksum; optional (None means "don't verify this one").
    sha256: str | None = None


# Records which character range in the combined text belongs to one PDF page.
# We build one big text string per document, so we need this to map any character
# position back to its page number later.
@dataclass(frozen=True)
class PageSpan:
    # The 1-based page number in the PDF.
    page_number: int
    # Character index where this page's text starts in the combined string...
    start_char: int
    # ...and where it ends.
    end_char: int


# A finished chunk ready to be embedded and stored, carrying all the metadata
# needed to cite it later.
@dataclass(frozen=True)
class DocumentChunk:
    # A unique id for this chunk, e.g. "entsoe-summer-outlook-2026-c12".
    id: str
    # The chunk text.
    text: str
    # The source filename (used for citations and the citation-integrity check).
    source: str
    # The document title.
    title: str
    # The source URL.
    source_url: str
    # Which chunk number this is within its document.
    chunk_index: int
    # First and last PDF page this chunk's text spans.
    page_start: int
    page_end: int

    # A computed view of the fields above as a flat dict of primitive values,
    # which is the only shape Chroma accepts for metadata (no nested objects).
    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "title": self.title,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


# Read data/sources.json and turn each JSON object into a SourceDocument.
def load_sources(path: Path) -> list[SourceDocument]:
    # Parse the JSON file into a list of plain dicts.
    records = json.loads(path.read_text(encoding="utf-8"))
    # `**record` unpacks each dict's keys as keyword arguments to the dataclass.
    return [SourceDocument(**record) for record in records]


# Compute a file's SHA256 by streaming it in 1MB blocks, so even very large PDFs
# never need to be fully loaded into memory at once.
def sha256_file(path: Path) -> str:
    # Create an empty SHA256 hasher.
    digest = hashlib.sha256()
    # Open the file in binary read mode.
    with path.open("rb") as file:
        # `iter(callable, sentinel)` calls the lambda repeatedly until it returns
        # b"" (end of file), feeding each 1MB block into the hasher.
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    # Return the final fingerprint as a hex string.
    return digest.hexdigest()


# Raise an error if a local file's checksum doesn't match the manifest. This is
# the reproducibility guarantee: we always validate against known-good bytes.
def verify_source_file(path: Path, source: SourceDocument) -> None:
    # If no checksum was pinned for this source, skip verification.
    if source.sha256 is None:
        return
    # Compute the file's actual checksum...
    actual = sha256_file(path)
    # ...and normalise the expected one to lowercase for a fair comparison.
    expected = source.sha256.lower()
    # If they differ, the file is wrong/corrupt/changed — stop immediately.
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}."
        )


# Download every manifest PDF (unless already present), then verify each one.
def download_sources(config: AppConfig, overwrite: bool = False) -> list[Path]:
    # Make sure the destination folder exists (create parents if needed).
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    # Collect the paths we end up with.
    downloaded: list[Path] = []
    # Loop over each source listed in the manifest.
    for source in load_sources(config.sources_path):
        # Where this PDF should live locally.
        target = config.raw_dir / source.filename
        # Download only if it's missing, or if we were told to overwrite.
        if not target.exists() or overwrite:
            # Build an HTTP request with a custom User-Agent (some servers reject
            # the default Python user agent).
            request = urllib.request.Request(
                source.url,
                headers={"User-Agent": "rag-pipeline-grounded-qa/0.1"},
            )
            # Open the URL (60s timeout) and write the downloaded bytes to disk.
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
        # Whether freshly downloaded or reused from cache, verify the checksum.
        verify_source_file(target, source)
        # Record the path.
        downloaded.append(target)
    # Return all local PDF paths.
    return downloaded


# Load the embedding model's tokenizer so chunking can measure sizes in the
# model's own tokens. Imported lazily because `transformers` is a heavy library.
def _load_chunk_tokenizer(model_name: str) -> Any:
    # Import here (not at the top) so importing this module stays cheap.
    from transformers import AutoTokenizer

    # `use_fast=True` requests the Rust-backed tokenizer, which is the only kind
    # that can return character offset mappings (which our chunker relies on).
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Defensive check: if we somehow got a slow tokenizer (no offsets), fail loudly.
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"Tokenizer for {model_name} must support offset mappings.")
    return tokenizer


# Clean up the raw text pypdf extracts, WITHOUT changing the actual words.
def _clean_pdf_text(text: str) -> str:
    # Replace NUL bytes (which some PDFs contain) with spaces.
    text = text.replace("\x00", " ")
    # Collapse runs of spaces/tabs into a single space.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines down to a single blank line (two newlines).
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim leading/trailing whitespace.
    return text.strip()


# Extract a PDF into ONE combined text string, plus a list of PageSpans telling
# us which slice of that string came from which page.
def _extract_pdf_text_with_pages(reader: PdfReader) -> tuple[str, list[PageSpan]]:
    # Pieces of text we'll join at the end (faster than repeated string +=).
    parts: list[str] = []
    # The page-span records.
    page_spans: list[PageSpan] = []
    # Running character count = where the next piece will start in the combined text.
    cursor = 0

    # Walk through the pages, numbering them from 1.
    for page_number, page in enumerate(reader.pages, start=1):
        # Extract this page's text (or "" if none) and clean it.
        page_text = _clean_pdf_text(page.extract_text() or "")
        # Skip pages that have no text (e.g. image-only pages).
        if not page_text:
            continue

        # If we've already added a page, insert a blank-line separator so words
        # from different pages don't run together. Advance the cursor by the 2
        # characters ("\n\n") we just added.
        if parts:
            parts.append("\n\n")
            cursor += 2
        # This page's text starts at the current cursor position.
        start_char = cursor
        # Add the page text and advance the cursor past it.
        parts.append(page_text)
        cursor += len(page_text)
        # Record where this page lives in the combined string.
        page_spans.append(
            PageSpan(page_number=page_number, start_char=start_char, end_char=cursor)
        )

    # Join all pieces into the final combined text; return it with the spans.
    return "".join(parts), page_spans


# Given a chunk's character span, find which page numbers it overlaps. A chunk
# can straddle a page boundary, so this may return two different pages.
def _page_range_for_chunk(page_spans: list[PageSpan], chunk: TextChunk) -> tuple[int, int]:
    # Collect every page whose character span overlaps the chunk's span. The
    # overlap test is "page ends after chunk starts AND page starts before chunk ends".
    pages = [
        span.page_number
        for span in page_spans
        if span.end_char > chunk.start_char and span.start_char < chunk.end_char
    ]
    # If somehow no page matched, return (-1, -1) as an "unknown" sentinel.
    if not pages:
        return -1, -1
    # Otherwise return the first and last page the chunk touches.
    return pages[0], pages[-1]


# Pick the right chunker: the real tokenizer-based one for ingestion, or the
# regex fallback when no tokenizer is supplied (e.g. in tests).
def _chunk_document_text(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    tokenizer: Any | None,
) -> list[TextChunk]:
    # No tokenizer -> use the lightweight regex chunker.
    if tokenizer is None:
        return chunk_text(text, chunk_tokens, overlap_tokens)
    # Otherwise use the model-tokenizer chunker.
    return chunk_text_with_tokenizer(text, tokenizer, chunk_tokens, overlap_tokens)


# Turn a single PDF file into a list of fully-described DocumentChunks.
def extract_chunks_from_pdf(
    path: Path,
    source: SourceDocument,
    chunk_tokens: int,
    overlap_tokens: int,
    tokenizer: Any | None = None,
) -> list[DocumentChunk]:
    # Open the PDF.
    reader = PdfReader(str(path))
    # Extract its combined text and the page-span map.
    full_text, page_spans = _extract_pdf_text_with_pages(reader)
    # Where we'll collect the chunks.
    chunks: list[DocumentChunk] = []

    # If the PDF had no extractable text, return an empty list.
    if not full_text:
        return chunks

    # Chunk the combined text, then enrich each chunk with metadata.
    for local_chunk in _chunk_document_text(
        full_text,
        chunk_tokens,
        overlap_tokens,
        tokenizer,
    ):
        # Work out which page(s) this chunk spans.
        page_start, page_end = _page_range_for_chunk(page_spans, local_chunk)
        # Build the full DocumentChunk with a stable, unique id and all metadata.
        chunks.append(
            DocumentChunk(
                # id is "<filename-without-extension>-c<chunk number>". Stable, so
                # re-ingesting upserts the same id instead of duplicating.
                id=f"{path.stem}-c{local_chunk.chunk_index}",
                text=local_chunk.text,
                source=source.filename,
                title=source.title,
                source_url=source.url,
                chunk_index=local_chunk.chunk_index,
                page_start=page_start,
                page_end=page_end,
            )
        )

    # Return all chunks for this document.
    return chunks


# Top-level ingestion helper: load and chunk EVERY manifest PDF found in data/raw.
def load_pdf_chunks(config: AppConfig) -> list[DocumentChunk]:
    # Build a filename -> SourceDocument lookup so we can match files to manifest entries.
    sources = {source.filename: source for source in load_sources(config.sources_path)}
    # Load the model tokenizer once and reuse it for every document.
    tokenizer = _load_chunk_tokenizer(config.embedding_model)
    # Accumulate chunks across all documents.
    chunks: list[DocumentChunk] = []
    # Iterate PDFs in sorted (deterministic) order.
    for path in sorted(config.raw_dir.glob("*.pdf")):
        # Find this file's manifest entry; skip any stray PDF not in the manifest.
        source = sources.get(path.name)
        if source is None:
            continue
        # Verify the file's checksum before trusting its contents.
        verify_source_file(path, source)
        # Extract this PDF's chunks and add them to the overall list.
        chunks.extend(
            extract_chunks_from_pdf(
                path,
                source,
                chunk_tokens=config.chunk_tokens,
                overlap_tokens=config.overlap_tokens,
                tokenizer=tokenizer,
            )
        )
    # Return all chunks across all documents.
    return chunks
