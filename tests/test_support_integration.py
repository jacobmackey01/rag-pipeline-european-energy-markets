import json
from dataclasses import replace

from rag_pipeline.config import AppConfig
from rag_pipeline import cli, support_evaluation, validation


def config(tmp_path):
    return AppConfig(tmp_path, tmp_path / "sources.json", tmp_path / "raw",
                     tmp_path / "chroma", "test", "embedding", "generation", 220, 40)


def test_validate_includes_failed_support_in_overall_result(tmp_path, monkeypatch):
    cfg = replace(config(tmp_path), claim_support_enabled=True)
    monkeypatch.setattr(validation, "VALIDATION_CASES", (
        validation.ValidationCase("case", "question"),))
    monkeypatch.setattr(validation, "retrieve", lambda *a, **kw: [])
    monkeypatch.setattr(validation, "answer_from_context", lambda *a: {
        "answer": "Claim (energy.pdf).", "citation_check": True,
        "citation_check_message": "Valid filename",
        "claim_support": {"status":"unavailable", "passed":False, "needs_review":True},
    })
    result = validation.run_validation(cfg)[0]
    assert result.checks["citation_integrity"] is True
    assert result.checks["claim_support"] is False
    assert result.passed is False
    assert result.claim_support["status"] == "unavailable"


def test_cli_override_and_json_keep_support_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.AppConfig, "from_env", lambda: config(tmp_path))
    def ask(cfg, *a, **kw):
        assert cfg.claim_support_enabled is True
        return {"answer":"Claim", "citation_check":True, "citation_check_message":"OK",
                "retrieved_chunks":[], "claim_support":{"needs_review":True}}
    monkeypatch.setattr(cli, "ask_question", ask)
    monkeypatch.setattr("sys.argv", ["rag-pipeline", "ask", "question", "--check-support", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["claim_support"]["needs_review"] is True


def test_no_check_support_overrides_environment_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.AppConfig, "from_env", lambda: replace(config(tmp_path), claim_support_enabled=True))
    def ask(cfg, *a, **kw):
        assert cfg.claim_support_enabled is False
        return {"answer":"Claim", "citation_check":True, "citation_check_message":"OK",
                "retrieved_chunks":[], "claim_support":{"status":"disabled"}}
    monkeypatch.setattr(cli, "ask_question", ask)
    monkeypatch.setattr("sys.argv", ["rag-pipeline", "ask", "question", "--no-check-support"])
    cli.main()


def test_evaluation_reports_operational_failures_separately(tmp_path, monkeypatch):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"label_status":"provisional", "evidence":{}, "cases":[
        {"name":"unsupported", "answer":"Claim (energy.pdf).", "evidence_ids":[],
         "expected_supported":False}]}))
    monkeypatch.setattr(support_evaluation, "citation_check", lambda *a: (True,"OK"))
    monkeypatch.setattr(support_evaluation, "assess_claim_support", lambda *a: {
        "status":"unavailable", "passed":False, "usage":{}, "latency_seconds":0.1})
    result = support_evaluation.evaluate_support(config(tmp_path), path)
    assert result["summary"]["baseline"]["unsupported_accepted"] == 1
    assert result["summary"]["augmented"]["unsupported_accepted"] == 0
    assert result["summary"]["unavailable_cases"] == 1
