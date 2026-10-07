import json
from dataclasses import replace

import pytest

from rag_pipeline import cli, validation
from test_support_integration import config


def result(checks):
    return validation.ValidationResult("known", "question", all(checks.values()), "answer", [], [],
                                       checks, [], xfail_reason="Missing passage",
                                       expected_failure_checks=("retrieval_expected_content", "expected_phrases"))


def test_xfail_does_not_hide_support_or_citation_failures():
    known = result({"retrieval_expected_content":False, "citation_integrity":True})
    assert known.status == "XFAIL" and not known.passed
    assert result({"retrieval_expected_content":False, "claim_support":False}).status == "FAIL"
    assert result({"retrieval_expected_content":False, "citation_integrity":False}).status == "FAIL"
    assert result({"retrieval_expected_content":True}).status == "XPASS"
    assert replace(known, xfail_reason="").status == "FAIL"


def test_json_preserves_failure_and_reason(tmp_path):
    path = tmp_path / "result.json"
    validation.write_validation_results([result({"expected_phrases":False})], path)
    saved = json.loads(path.read_text())[0]
    assert saved["status"] == "XFAIL" and saved["passed"] is False
    assert saved["xfail_reason"] == "Missing passage"


@pytest.mark.parametrize("checks,exit_code", [({"expected_phrases":False},0),
                                             ({"expected_phrases":True},1),
                                             ({"citation_integrity":False},1)])
def test_cli_xfail_visible_but_unexpected_outcomes_fail(tmp_path, monkeypatch, capsys, checks, exit_code):
    monkeypatch.setattr(cli.AppConfig, "from_env", lambda: config(tmp_path))
    monkeypatch.setattr(cli, "run_validation", lambda *a, **kw:[result(checks)])
    monkeypatch.setattr("sys.argv", ["rag-pipeline", "validate"])
    if exit_code:
        with pytest.raises(SystemExit, match="1"):
            cli.main()
    else:
        cli.main()
    assert "known gap: Missing passage" in capsys.readouterr().out


def test_original_questions_stay_broad_and_probes_explicit():
    cases = {c.name:c for c in validation.VALIDATION_CASES}
    broad = cases["grounding_acer_see_cross_zonal_capacity"]
    assert broad.question == "What does ACER say about cross-zonal capacity and flexibility in Southeast Europe?"
    assert broad.xfail_reason and broad.retrieval_top_k == 4
    monitoring = cases["grounding_acer_2026_market_developments"]
    assert monitoring.question == "What does ACER say about key developments in European electricity and gas markets in 2026?"
    assert monitoring.xfail_reason
    assert cases["retrieval_see_price_spike_top12_probe"].retrieval_top_k == 12
