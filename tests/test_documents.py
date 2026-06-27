from pathlib import Path

import pytest

from rag_pipeline.chunking import chunk_text_with_tokenizer
from rag_pipeline.documents import (
    PageSpan,
    SourceDocument,
    _page_range_for_chunk,
    sha256_file,
    verify_source_file,
)


class FakeTokenizer:
    is_fast = True

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


def test_sha256_file_returns_lowercase_digest(tmp_path: Path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
