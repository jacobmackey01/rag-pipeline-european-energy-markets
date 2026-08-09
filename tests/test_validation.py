from pathlib import Path
from types import SimpleNamespace

import rag_pipeline.validation as validation
from rag_pipeline.documents import CorpusIntegrityError


def test_validation_stops_before_retrieval_when_corpus_fails(monkeypatch):
    def fail_corpus(config):
        raise CorpusIntegrityError("checksum drift")

    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("retrieval must not run against an invalid corpus")

    monkeypatch.setattr(validation, "verify_corpus", fail_corpus)
    monkeypatch.setattr(validation, "retrieve", unexpected_retrieval)

    results = validation.run_validation(SimpleNamespace())

    assert len(results) == 1
    assert results[0].name == "corpus_integrity"
    assert results[0].passed is False
    assert "checksum drift" in results[0].notes[0]


def test_validation_records_successful_corpus_check(monkeypatch):
    monkeypatch.setattr(
        validation,
        "verify_corpus",
        lambda config: [Path("first.pdf"), Path("second.pdf")],
    )
    monkeypatch.setattr(validation, "VALIDATION_CASES", ())

    results = validation.run_validation(SimpleNamespace())

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].retrieved_sources == ["first.pdf", "second.pdf"]
    assert results[0].checks == {"manifest_files_present_and_checksummed": True}
