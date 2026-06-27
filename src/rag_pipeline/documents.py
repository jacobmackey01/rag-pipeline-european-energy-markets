from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag_pipeline.chunking import TextChunk, chunk_text, chunk_text_with_tokenizer
from rag_pipeline.config import AppConfig


@dataclass(frozen=True)
class SourceDocument:
    filename: str
    title: str
    url: str
    sha256: str | None = None


@dataclass(frozen=True)
class PageSpan:
    page_number: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    title: str
    source_url: str
    chunk_index: int
    page_start: int
    page_end: int

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


def load_sources(path: Path) -> list[SourceDocument]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [SourceDocument(**record) for record in records]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_file(path: Path, source: SourceDocument) -> None:
    if source.sha256 is None:
        return
    actual = sha256_file(path)
    expected = source.sha256.lower()
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}."
        )


def download_sources(config: AppConfig, overwrite: bool = False) -> list[Path]:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for source in load_sources(config.sources_path):
        target = config.raw_dir / source.filename
        if not target.exists() or overwrite:
            request = urllib.request.Request(
                source.url,
                headers={"User-Agent": "rag-pipeline-grounded-qa/0.1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
        verify_source_file(target, source)
        downloaded.append(target)
    return downloaded


def _load_chunk_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"Tokenizer for {model_name} must support offset mappings.")
    return tokenizer


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_text_with_pages(reader: PdfReader) -> tuple[str, list[PageSpan]]:
    parts: list[str] = []
    page_spans: list[PageSpan] = []
    cursor = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _clean_pdf_text(page.extract_text() or "")
        if not page_text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start_char = cursor
        parts.append(page_text)
        cursor += len(page_text)
        page_spans.append(
            PageSpan(page_number=page_number, start_char=start_char, end_char=cursor)
        )

    return "".join(parts), page_spans


def _page_range_for_chunk(page_spans: list[PageSpan], chunk: TextChunk) -> tuple[int, int]:
    pages = [
        span.page_number
        for span in page_spans
        if span.end_char > chunk.start_char and span.start_char < chunk.end_char
    ]
    if not pages:
        return -1, -1
    return pages[0], pages[-1]


def _chunk_document_text(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    tokenizer: Any | None,
) -> list[TextChunk]:
    if tokenizer is None:
        return chunk_text(text, chunk_tokens, overlap_tokens)
    return chunk_text_with_tokenizer(text, tokenizer, chunk_tokens, overlap_tokens)


def extract_chunks_from_pdf(
    path: Path,
    source: SourceDocument,
    chunk_tokens: int,
    overlap_tokens: int,
    tokenizer: Any | None = None,
) -> list[DocumentChunk]:
    reader = PdfReader(str(path))
    full_text, page_spans = _extract_pdf_text_with_pages(reader)
    chunks: list[DocumentChunk] = []

    if not full_text:
        return chunks

    for local_chunk in _chunk_document_text(
        full_text,
        chunk_tokens,
        overlap_tokens,
        tokenizer,
    ):
        page_start, page_end = _page_range_for_chunk(page_spans, local_chunk)
        chunks.append(
            DocumentChunk(
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

    return chunks


def load_pdf_chunks(config: AppConfig) -> list[DocumentChunk]:
    sources = {source.filename: source for source in load_sources(config.sources_path)}
    tokenizer = _load_chunk_tokenizer(config.embedding_model)
    chunks: list[DocumentChunk] = []
    for path in sorted(config.raw_dir.glob("*.pdf")):
        source = sources.get(path.name)
        if source is None:
            continue
        verify_source_file(path, source)
        chunks.extend(
            extract_chunks_from_pdf(
                path,
                source,
                chunk_tokens=config.chunk_tokens,
                overlap_tokens=config.overlap_tokens,
                tokenizer=tokenizer,
            )
        )
    return chunks
