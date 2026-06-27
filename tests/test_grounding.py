from rag_pipeline.config import REFUSAL_MESSAGE
from rag_pipeline.grounding import citation_check, extract_cited_sources
from rag_pipeline.store import RetrievedChunk


def _chunk(source: str) -> RetrievedChunk:
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
    answer = "The MPC maintained Bank Rate at 5.25% (boe-mpr-may-2024.pdf)."

    assert extract_cited_sources(answer) == {"boe-mpr-may-2024.pdf"}


def test_citation_check_passes_when_cited_source_was_retrieved():
    answer = "Ireland required monitoring (entsoe-summer-outlook-2024.pdf)."

    passed, message = citation_check(answer, [_chunk("entsoe-summer-outlook-2024.pdf")])

    assert passed is True
    assert "present" in message


def test_citation_check_fails_unknown_source():
    answer = "Claim (unknown.pdf)."

    passed, message = citation_check(answer, [_chunk("known.pdf")])

    assert passed is False
    assert "not present" in message


def test_refusal_answer_does_not_need_citation():
    passed, _ = citation_check(REFUSAL_MESSAGE, [])

    assert passed is True
