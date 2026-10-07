"""Document provenance used for embeddings; evidence text stays unchanged."""

CONTEXT_VERSION = "document-title-v1"


def contextual_text(title: str, passage: str) -> str:
    return f"Document: {title}\n\nPassage:\n{passage}"


def passage_token_budget(title: str, tokenizer, requested: int) -> int:
    # Reserve the header and special tokens within MiniLM's 256-token limit.
    header_tokens = len(tokenizer.encode(contextual_text(title, ""), add_special_tokens=True))
    budget = min(requested, 256 - header_tokens - 2)
    if budget < 1:
        raise ValueError("Document title is too long for contextual MiniLM embeddings.")
    return budget
