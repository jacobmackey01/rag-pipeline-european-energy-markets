import csv
import json

import pytest

from rag_pipeline.blind_labels import FIELDS, prepare_blind_rows, write_blind_csv
from rag_pipeline.config import REFUSAL_MESSAGE
from rag_pipeline import cli


def saved_answer(question="q"):
    return {"question":question, "answer":"First claim (energy.pdf). Second claim (energy.pdf).",
            "claim_support":{"probability":0.9},
            "retrieved_chunks":[{"id":"c1", "source":"energy.pdf", "page_start":2,
                                 "page_end":3, "source_url":"https://example.com/report",
                                 "text":"Exact evidence, with a comma.\nAnd a newline.", "retrieval_score":0.8},
                                {"source":"other.pdf", "text":"Do not export uncited evidence"}]}


def test_selection_is_reproducible_blank_and_score_free():
    answers = [saved_answer(f"q{i}") for i in range(5)]
    rows = prepare_blind_rows(answers, count=8)
    assert rows == prepare_blind_rows(answers, count=8)
    assert len({r["unit_id"] for r in rows}) == 8
    for row in rows:
        assert set(row) == set(FIELDS)
        assert row["human_label"] == row["review_notes"] == ""
        assert "Exact evidence, with a comma.\nAnd a newline." in row["evidence"]
        assert "other.pdf" not in row["evidence"]
    groups = {r["question"]:{v["split"] for v in rows if v["question"] == r["question"]} for r in rows}
    assert all(len(splits) == 1 for splits in groups.values())
    assert {r["split"] for r in rows} == {"development", "test"}


def test_uncited_and_missing_evidence_units_are_retained():
    answers = [{"question":"q", "answer":"Claim (missing.pdf). Uncited assertion.", "retrieved_chunks":[]}]
    rows = prepare_blind_rows(answers, count=2)
    assert all(r["evidence"] == "" for r in rows)
    assert any(r["cited_sources"] == "missing.pdf" for r in rows)


def test_refusals_duplicates_and_insufficient_units():
    answer = saved_answer()
    answers = [answer, answer, {"question":"refusal", "answer":REFUSAL_MESSAGE}]
    assert len(prepare_blind_rows(answers, count=2)) == 2
    with pytest.raises(ValueError, match="only 2"):
        prepare_blind_rows(answers, count=3)
    with pytest.raises(ValueError, match="positive"):
        prepare_blind_rows([], count=0)


def test_csv_round_trip_and_formula_safety(tmp_path):
    rows = prepare_blind_rows([saved_answer("=1+1")], count=2)
    path = tmp_path / "labels.csv"
    write_blind_csv(rows, path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        decoded = list(csv.DictReader(file))
    assert decoded[0]["question"] == "'=1+1"
    assert decoded[0]["evidence"] == rows[0]["evidence"]
    assert decoded[0]["human_label"] == ""


def test_cli_export_does_not_load_credentials_or_score(tmp_path, monkeypatch):
    source = tmp_path / "answers.json"
    source.write_text(json.dumps([saved_answer()]))
    target = tmp_path / "labels.csv"
    monkeypatch.setattr(cli.AppConfig, "from_env", lambda: pytest.fail("No credential loading expected"))
    monkeypatch.setattr("sys.argv", ["rag-pipeline", "export-labels", "--answers", str(source),
                                  "--output", str(target), "--count", "2"])
    cli.main()
    assert target.exists()
