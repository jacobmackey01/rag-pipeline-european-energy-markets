from types import SimpleNamespace

import pytest

from rag_pipeline.embedding_context import contextual_text, passage_token_budget
from rag_pipeline.hybrid import bm25, rank_records, terms
from rag_pipeline.documents import DocumentChunk
from rag_pipeline import store


def test_title_metadata_can_outweigh_misleading_passage_similarity():
    records = {
        "ids":["summer", "winter"],
        "documents":["Preparing for winter adequacy risks", "Overall adequacy is favourable"],
        "metadatas":[{"title":"ENTSO-E Summer Outlook 2026"}, {"title":"ENTSO-E Winter Outlook 2025-2026"}],
        # Similar seasonal passages, with a small semantic advantage for Summer.
        "embeddings":[[0.7,0.7141428], [0.65,0.7599342]],
    }
    ranked = rank_records("Winter 2025/2026 Outlook adequacy risks", [1.0,0.0], records)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]  # Cosine alone prefers the Summer passage.


def test_no_lexical_match_preserves_semantic_order():
    records = {"ids":["a","b"], "documents":["gas","electricity"],
               "metadatas":[{},{}], "embeddings":[[0.0,1.0],[1.0,0.0]]}
    assert rank_records("unmatched",[1.0,0.0],records)[0][0] == 1


def test_bm25_handles_empty_text_without_fake_matches():
    assert bm25("winter", ["", ""]) == [0.0,0.0]
    assert bm25("winter", ["summer outlook", "winter outlook"])[1] > 0


def test_short_edition_years_are_normalized_for_lexical_search():
    assert terms("Winter 2025/26") == terms("Winter 2025–2026")


def test_vector_dimension_mismatch_is_not_silently_ranked():
    records = {"ids":["a"],"documents":["winter"],"metadatas":[{}],"embeddings":[[1.0,0.0]]}
    with pytest.raises(RuntimeError,match="dimensions"):
        rank_records("winter",[1.0],records)


def test_context_header_is_only_for_embeddings(monkeypatch):
    collected = []
    stored = []
    collection = SimpleNamespace(count=lambda:0, upsert=lambda **kwargs:stored.append(kwargs))
    monkeypatch.setattr(store,"get_collection",lambda cfg:collection)
    def embed(texts):
        collected.extend(texts)
        return [[1.0,0.0] for text in texts]
    chunk = DocumentChunk("id","Prices were 1000 EUR/MWh.","energy.pdf","Winter Outlook 2025-2026",
                          "https://example.com",0,3,3)
    store.index_chunks(SimpleNamespace(),[chunk],embedder=SimpleNamespace(embed=embed))
    assert collected == [contextual_text(chunk.title,chunk.text)]
    assert stored[0]["documents"] == [chunk.text]
    assert stored[0]["metadatas"][0]["page_start"] == 3


def test_header_tokens_are_reserved_within_embedding_limit():
    tokenizer = SimpleNamespace(encode=lambda text,**kwargs:[0]*50)
    assert passage_token_budget("title",tokenizer,220) == 204
    tokenizer.encode = lambda text,**kwargs:[0]*300
    with pytest.raises(ValueError,match="too long"):
        passage_token_budget("title",tokenizer,220)
