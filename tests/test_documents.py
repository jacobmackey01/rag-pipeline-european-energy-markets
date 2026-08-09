# Unit tests for document checksums and page-aware tokenizer chunking.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_pipeline.chunking import chunk_text_with_tokenizer
from rag_pipeline.documents import (
    CorpusIntegrityError,
    PageSpan,
    SourceDocument,
    _page_range_for_chunk,
    sha256_file,
    verify_corpus,
    verify_source_file,
)


# Small stand-in for a Hugging Face fast tokenizer in offset-mapping tests.
class FakeTokenizer:
    is_fast = True

    # Return simple whitespace token offsets matching the tokenizer API shape.
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        offsets = []
        cursor = 0
        for token in text.split(" "):
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"offset_mapping": offsets}


# Tokenizer offsets should drive chunk windows and preserve source positions.
def test_chunk_text_with_tokenizer_uses_real_offsets():
    text = "alpha beta gamma delta epsilon"

    chunks = chunk_text_with_tokenizer(
        text,
        FakeTokenizer(),
        chunk_tokens=3,
        overlap_tokens=1,
    )

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma",
        "gamma delta epsilon",
    ]
    assert chunks[0].token_count == 3
    assert chunks[1].start_char == text.index("gamma")


# A chunk spanning the page separator should report both source pages.
def test_page_range_can_span_pages():
    text = "page one text\n\npage two text"
    chunks = chunk_text_with_tokenizer(
        text,
        FakeTokenizer(),
        chunk_tokens=6,
        overlap_tokens=1,
    )
    page_spans = [
        PageSpan(page_number=1, start_char=0, end_char=len("page one text")),
        PageSpan(
            page_number=2,
            start_char=text.index("page two"),
            end_char=len(text),
        ),
    ]

    assert _page_range_for_chunk(page_spans, chunks[0]) == (1, 2)


# Pinned corpus checksums should fail loudly on drift or corruption.
def test_verify_source_file_rejects_checksum_mismatch(tmp_path: Path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"report-bytes")
    source = SourceDocument(
        filename="report.pdf",
        title="Report",
        url="https://example.com/report.pdf",
        sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_source_file(path, source)


# Checksum output is normalized for direct comparison with sources.json.
def test_sha256_file_returns_lowercase_digest(tmp_path: Path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _corpus_config(tmp_path: Path, records: list[dict]) -> SimpleNamespace:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    sources_path = tmp_path / "data" / "sources.json"
    sources_path.write_text(json.dumps(records), encoding="utf-8")
    return SimpleNamespace(raw_dir=raw_dir, sources_path=sources_path)


def test_verify_corpus_checks_every_manifest_file(tmp_path: Path):
    config = _corpus_config(
        tmp_path,
        [
            {
                "filename": "report.pdf",
                "title": "Report",
                "url": "https://example.com/report.pdf",
                "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            }
        ],
    )
    (config.raw_dir / "report.pdf").write_bytes(b"abc")

    assert verify_corpus(config) == [config.raw_dir / "report.pdf"]


def test_verify_corpus_reports_missing_and_unpinned_sources(tmp_path: Path):
    config = _corpus_config(
        tmp_path,
        [
            {
                "filename": "missing.pdf",
                "title": "Missing",
                "url": "https://example.com/missing.pdf",
                "sha256": "0" * 64,
            },
            {
                "filename": "unpinned.pdf",
                "title": "Unpinned",
                "url": "https://example.com/unpinned.pdf",
            },
        ],
    )

    with pytest.raises(CorpusIntegrityError) as exc_info:
        verify_corpus(config)

    message = str(exc_info.value)
    assert "missing missing.pdf" in message
    assert "no SHA256 checksum pinned for unpinned.pdf" in message
