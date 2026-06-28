"""Unit tests for citation extraction and source-integrity checks."""

from rag_pipeline.config import REFUSAL_MESSAGE
from rag_pipeline.grounding import citation_check, extract_cited_sources
from rag_pipeline.store import RetrievedChunk


def _chunk(source: str) -> RetrievedChunk:
    """Build a minimal retrieved chunk for citation-check tests."""
    return RetrievedChunk(
        id="id",
        text="text",
        source=source,
        title="title",
        source_url="https://example.com",
        chunk_index=0,
        page_start=1,
        page_end=1,
        distance=0.1,
    )


def test_extract_cited_sources_finds_pdf_filenames():
    """PDF filenames in normal parenthesized citations should be detected."""
    answer = "The MPC maintained Bank Rate at 5.25% (boe-mpr-may-2024.pdf)."

    assert extract_cited_sources(answer) == {"boe-mpr-may-2024.pdf"}


def test_extract_cited_sources_does_not_capture_preceding_words():
    """Citation parsing should not swallow prose before the filename."""
    answer = "According to entsoe-winter-outlook-2025-2026.pdf, adequacy is monitored."

    assert extract_cited_sources(answer) == {"entsoe-winter-outlook-2025-2026.pdf"}


def test_citation_check_passes_when_cited_source_was_retrieved():
    """A citation is valid when it names a PDF present in retrieved context."""
    answer = "Ireland required monitoring (entsoe-summer-outlook-2024.pdf)."

    passed, message = citation_check(answer, [_chunk("entsoe-summer-outlook-2024.pdf")])

    assert passed is True
    assert "present" in message


def test_citation_check_fails_unknown_source():
    """A model cannot cite a filename that was not supplied in context."""
    answer = "Claim (unknown.pdf)."

    passed, message = citation_check(answer, [_chunk("known.pdf")])

    assert passed is False
    assert "not present" in message


def test_refusal_answer_does_not_need_citation():
    """The exact refusal message is allowed to omit source citations."""
    passed, _ = citation_check(REFUSAL_MESSAGE, [])

    assert passed is True
