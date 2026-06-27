from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from rag_pipeline.chunking import chunk_text
from rag_pipeline.config import AppConfig


@dataclass(frozen=True)
class SourceDocument:
    filename: str
    title: str
    url: str


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


def download_sources(config: AppConfig, overwrite: bool = False) -> list[Path]:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for source in load_sources(config.sources_path):
        target = config.raw_dir / source.filename
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue
        request = urllib.request.Request(
            source.url,
            headers={"User-Agent": "rag-pipeline-grounded-qa/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        downloaded.append(target)
    return downloaded


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_chunks_from_pdf(
    path: Path,
    source: SourceDocument,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    reader = PdfReader(str(path))
    chunks: list[DocumentChunk] = []
    global_index = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _clean_pdf_text(page.extract_text() or "")
        if not page_text:
            continue
        for local_chunk in chunk_text(page_text, chunk_tokens, overlap_tokens):
            chunks.append(
                DocumentChunk(
                    id=f"{path.stem}-p{page_number}-c{local_chunk.chunk_index}",
                    text=local_chunk.text,
                    source=source.filename,
                    title=source.title,
                    source_url=source.url,
                    chunk_index=global_index,
                    page_start=page_number,
                    page_end=page_number,
                )
            )
            global_index += 1

    return chunks


def load_pdf_chunks(config: AppConfig) -> list[DocumentChunk]:
    sources = {source.filename: source for source in load_sources(config.sources_path)}
    chunks: list[DocumentChunk] = []
    for path in sorted(config.raw_dir.glob("*.pdf")):
        source = sources.get(path.name)
        if source is None:
            continue
        chunks.extend(
            extract_chunks_from_pdf(
                path,
                source,
                chunk_tokens=config.chunk_tokens,
                overlap_tokens=config.overlap_tokens,
            )
        )
    return chunks
