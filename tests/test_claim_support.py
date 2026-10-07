import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from rag_pipeline.claim_support import assess_claim_support, split_claim_units
from rag_pipeline.config import AppConfig, REFUSAL_MESSAGE
from rag_pipeline.generation import answer_from_context
from rag_pipeline.store import RetrievedChunk


@pytest.fixture
def config(tmp_path):
    return AppConfig(tmp_path, tmp_path / "sources.json", tmp_path / "raw",
                     tmp_path / "chroma", "test", "embedding", "generation", 220, 40,
                     claim_support_enabled=True)


def chunk(source="energy.pdf", text="Capacity was 1000 MW in 2024."):
    return RetrievedChunk("chunk-1", text, source, "Energy report", "https://example.com",
                          0, 9, 9, 0.1)


def mock_client(probabilities, requests, *, response=None, status=200):
    def handler(request):
        assert request.url.path == "/v1/decisions"
        body = json.loads(request.content)
        requests.append(body)
        payload = response or {
            "model": "gpt-6-luna",
            "answers": [dict(type="predicate", name=q["name"], probability=p)
                        for q, p in zip(body["questions"], probabilities)],
            "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0},
                      "total_tokens": 100},
        }
        return httpx.Response(status, json=payload)
    return OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_units_preserve_decimals_pdf_names_and_uncited_prose():
    answer = "- Demand rose 5.25% (report.v2.pdf). Prices fell.\n(report.v2.pdf)\nUncited claim."
    assert split_claim_units(answer) == [
        "Demand rose 5.25% (report.v2.pdf).", "Prices fell. (report.v2.pdf)", "Uncited claim."]


def test_source_annotation_covers_preceding_sentences_and_bullets():
    answer = "ACER plans monitoring. Work includes:\n\n- LNG\n- Network codes\n\nDates are indicative. **Source: energy.pdf**"
    assert split_claim_units(answer) == [
        "ACER plans monitoring. Work includes: LNG Network codes Dates are indicative. **Source: energy.pdf**"]


def test_shared_citation_keeps_all_assertions_for_assessment(config):
    requests = []
    result = assess_claim_support(config, "Capacity was 1000 MW. It has since doubled. **(energy.pdf)**",
        [chunk()], client=mock_client([0.05], requests))
    record = json.loads(requests[0]["input"])["records"][0]
    assert "It has since doubled" in record["claim"]
    assert result["passed"] is False


def test_each_claim_uses_only_its_cited_passages(config):
    requests = []
    client = mock_client([0.95, 0.01], requests)
    result = assess_claim_support(config,
        "Capacity was 1000 MW (energy.pdf). Capacity was 9000 MW (other.pdf).",
        [chunk(), chunk("other.pdf", "Capacity was 5 MW.")], client=client)
    assert result["passed"] is False
    assert result["needs_review"] is True
    assert [x["status"] for x in result["claims"]] == ["supported", "unsupported"]
    records = json.loads(requests[0]["input"])["records"]
    assert records[0]["evidence"] == [{"source": "energy.pdf", "text": "Capacity was 1000 MW in 2024."}]
    assert records[1]["evidence"][0]["source"] == "other.pdf"
    assert result["claims"][0]["evidence_chunks"][0]["page_start"] == 9


@pytest.mark.parametrize("p,status", [(0.8,"supported"), (0.2,"unsupported"), (0.5,"uncertain")])
def test_thresholds(config, p, status):
    result = assess_claim_support(config, "Claim (energy.pdf).", [chunk()],
                                  client=mock_client([p], []))
    assert result["claims"][0]["status"] == status
    assert result["needs_review"] is (status != "supported")


def test_missing_citations_do_not_borrow_other_claims_evidence(config):
    requests = []
    result = assess_claim_support(config, "An uncited assertion. Another (unknown.pdf).", [chunk()],
                                  client=mock_client([], requests))
    assert requests == []
    assert all(x["status"] == "missing_evidence" for x in result["claims"])
    assert result["needs_review"] is True


def test_refusal_makes_no_decisions_request(config):
    requests = []
    result = assess_claim_support(config, REFUSAL_MESSAGE, [], client=mock_client([], requests))
    assert requests == []
    assert result["status"] == "not_applicable"
    assert result["passed"] is None


@pytest.mark.parametrize("answers", [[], [{"type":"predicate", "name":"wrong", "probability":0.99}],
    [{"type":"predicate", "name":"claim_1", "probability":1.5}],
    [{"type":"predicate", "name":"claim_1", "probability":"invalid"}]])
def test_malformed_results_require_review(config, answers):
    result = assess_claim_support(config, "Claim (energy.pdf).", [chunk()],
        client=mock_client([], [], response={"model":"gpt-6-luna", "answers":answers}))
    assert result["status"] == "unavailable"
    assert result["passed"] is False


def test_provider_refusal_requires_review(config):
    response = {"model":"gpt-6-luna", "answers":[{"type":"refusal", "name":"claim_1"}],
                "usage":{"input_tokens":10}}
    result = assess_claim_support(config, "Claim (energy.pdf).", [chunk()],
                                  client=mock_client([], [], response=response))
    assert result["claims"][0]["status"] == "refused"
    assert result["needs_review"] is True


def test_api_failure_is_not_a_pass_and_does_not_echo_error_body(config):
    result = assess_claim_support(config, "Claim (energy.pdf).", [chunk()],
        client=mock_client([], [], status=429,
                           response={"error":{"message":"secret-test-marker", "type":"rate_limit_error"}}))
    assert result["status"] == "unavailable"
    assert result["passed"] is False
    assert "secret-test-marker" not in json.dumps(result)


def test_overflow_units_are_flagged_not_dropped(config):
    requests = []
    answer = "\n".join(["Claim (energy.pdf)."] * 33)
    result = assess_claim_support(config, answer, [chunk()],
                                  client=mock_client([0.99] * 32, requests))
    assert len(result["claims"]) == 33
    assert result["claims"][-1]["status"] == "not_checked"
    assert result["passed"] is False


def test_empty_answer_requires_review(config):
    assert assess_claim_support(config, "", [], client=mock_client([], []))["needs_review"] is True


def test_bad_thresholds_rejected(config):
    with pytest.raises(ValueError):
        replace(config, unsupported_threshold=0.9, support_threshold=0.8)


@pytest.mark.parametrize("enabled", [False, True])
def test_generation_keeps_answer_and_citation_result(config, monkeypatch, enabled):
    from rag_pipeline import generation
    requests = []
    client = mock_client([0.01], requests)
    answer = "Capacity was 9000 MW (energy.pdf)."
    client.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=answer))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(generation, "OpenAI", lambda: client)
    result = answer_from_context(replace(config, claim_support_enabled=enabled), "Capacity?", [chunk()])
    assert result["answer"] == answer
    assert result["citation_check"] is True
    if enabled:
        assert result["claim_support"]["passed"] is False
        assert len(requests) == 1
    else:
        assert requests == []
        assert result["claim_support"]["status"] == "disabled"


def test_no_retrieval_skips_generation_and_support(config, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = answer_from_context(config, "Capacity?", [])
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["claim_support"]["status"] == "not_applicable"
