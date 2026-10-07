import json
from dataclasses import asdict
from types import SimpleNamespace

import chromadb
import pytest
from chromadb.config import Settings

from rag_pipeline.config import AppConfig
from rag_pipeline.embedding_context import CONTEXT_VERSION
from rag_pipeline.documents import SourceDocument
from rag_pipeline.retrieval_scope import source_scope
from rag_pipeline import cli, pipeline, store


WINTER = "entsoe-winter-outlook-2025-2026.pdf"
SUMMER = "entsoe-summer-outlook-2026.pdf"
CATALOG = [SourceDocument(WINTER, "ENTSO-E Winter Outlook 2025-2026", "https://example.com/w"),
           SourceDocument(SUMMER, "ENTSO-E Summer Outlook 2026", "https://example.com/s")]


@pytest.mark.parametrize("edition", ["2025-2026", "2025–2026", "2025—2026", "2025‑2026", "2025 / 2026", "2025/26"])
def test_explicit_winter_edition_resolves_filename(edition):
    assert source_scope(f"What does the winter outlook {edition} say?", CATALOG) == [WINTER]


def test_broad_questions_remain_unscoped():
    assert source_scope("What are the risks to European winter adequacy?", CATALOG) is None


def test_comparison_retains_both_requested_reports():
    assert source_scope("Compare Winter Outlook 2025-2026 with Summer Outlook 2026", CATALOG) == [WINTER, SUMMER]


def test_explicit_filename_can_scope_general_question():
    assert source_scope("Adequacy?", CATALOG, [WINTER, WINTER]) == [WINTER]


@pytest.mark.parametrize("question,sources", [("Winter Outlook 2024-2025?", None), ("Adequacy?", ["unknown.pdf"]), ("Adequacy?", [])])
def test_unknown_report_or_filename_cannot_fall_back(question, sources):
    with pytest.raises(ValueError):
        source_scope(question, CATALOG, sources)


@pytest.fixture
def retrieval_setup(tmp_path, monkeypatch):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([asdict(item) for item in CATALOG]))
    cfg = AppConfig(tmp_path, manifest, tmp_path / "raw", tmp_path / "chroma", "test", "embed", "gen", 220, 40)
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    collection = client.create_collection("scope-regression", metadata={"hnsw:space":"cosine", "embedding_context_version":CONTEXT_VERSION})
    collection.add(ids=["summer", "winter-1", "winter-2"],
                   embeddings=[[1.0,0.0],[0.1,1.0],[0.2,1.0]],
                   documents=["Winter adequacy discussion in Summer report", "Winter summary", "Winter risk"],
                   metadatas=[{"source":SUMMER}, {"source":WINTER}, {"source":WINTER}])
    monkeypatch.setattr(store, "get_collection", lambda cfg: collection)
    embedder = SimpleNamespace(embed=lambda texts: [[1.0,0.0] for text in texts])
    yield cfg, collection, embedder
    client.delete_collection("scope-regression")


def test_named_report_filter_precedes_semantic_ranking(retrieval_setup):
    cfg, _, embedder = retrieval_setup
    broad = store.retrieve(cfg, "European adequacy?", top_k=3, embedder=embedder)
    assert broad[0].source == SUMMER  # Deliberately more similar than either Winter chunk.
    scoped = store.retrieve(cfg, "Winter Outlook 2025-2026: European adequacy?", top_k=4, embedder=embedder)
    assert len(scoped) == 2
    assert {chunk.source for chunk in scoped} == {WINTER}


def test_explicit_source_filter_is_applied_in_chroma(retrieval_setup):
    cfg, _, embedder = retrieval_setup
    assert {c.source for c in store.retrieve(cfg, "Adequacy?", embedder=embedder, sources=[WINTER])} == {WINTER}


def test_missing_indexed_requested_report_returns_no_context(retrieval_setup):
    cfg, collection, embedder = retrieval_setup
    collection.delete(where={"source":WINTER})
    assert store.retrieve(cfg, "Winter Outlook 2025-2026?", embedder=embedder) == []


def test_unknown_edition_refuses_before_embedding(retrieval_setup):
    cfg, _, _ = retrieval_setup
    def fail_embed(texts):
        pytest.fail("Unrecognised editions must fail before embedding.")
    assert store.retrieve(cfg, "Winter Outlook 2024-2025?", embedder=SimpleNamespace(embed=fail_embed)) == []


def test_cli_and_pipeline_forward_source_selection(tmp_path, monkeypatch, capsys):
    cfg = AppConfig(tmp_path, tmp_path / "sources.json", tmp_path / "raw", tmp_path / "chroma",
                    "test", "embed", "gen", 220, 40)
    monkeypatch.setattr(cli.AppConfig, "from_env", lambda: cfg)
    monkeypatch.setattr(pipeline, "LocalEmbedder", lambda name: object())
    def retrieve(config, question, **kwargs):
        assert kwargs["sources"] == [WINTER]
        return []
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr("sys.argv", ["rag-pipeline", "ask", "Adequacy?", "--source", WINTER, "--json"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["answer"] == "Not found in the provided documents."


@pytest.mark.parametrize("command",["ask","retrieve"])
@pytest.mark.parametrize("question",["Compare Winter Outlook 2024-2025 and Winter Outlook 2025-2026",
                                    "What does the winter outlook for 2024-25 say?",
                                    "ENTSO-E Winter 2024/2025 Outlook: adequacy risks?"])
def test_missing_edition_cli_refuses_without_traceback(retrieval_setup,monkeypatch,capsys,command,question):
    cfg,_,_ = retrieval_setup
    monkeypatch.setattr(cli.AppConfig,"from_env",lambda:cfg)
    monkeypatch.setattr("sys.argv",["rag-pipeline",command,question])
    cli.main()
    captured=capsys.readouterr()
    assert "Not found in the provided documents." in captured.out
    assert "Traceback" not in captured.err


def test_old_embedding_index_requires_rebuild(retrieval_setup):
    cfg,collection,embedder=retrieval_setup
    collection.modify(metadata={"embedding_context_version":"passage-only"})
    with pytest.raises(RuntimeError,match="ingest --reset"):
        store.retrieve(cfg,"Adequacy?",embedder=embedder)
