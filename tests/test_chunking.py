import pytest

from rag_pipeline.chunking import chunk_text, tokenize


def test_chunk_text_uses_overlap():
    text = " ".join(f"token{i}" for i in range(12))

    chunks = chunk_text(text, chunk_tokens=5, overlap_tokens=2)

    assert len(chunks) == 4
    assert chunks[0].text == "token0 token1 token2 token3 token4"
    assert chunks[1].text.startswith("token3 token4")
    assert chunks[-1].text == "token9 token10 token11"


def test_chunk_text_rejects_overlap_at_or_above_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_tokens=10, overlap_tokens=10)


def test_tokenize_keeps_punctuation_as_tokens():
    assert tokenize("Bank Rate: 5.25%.") == ["Bank", "Rate", ":", "5", ".", "25", "%", "."]
