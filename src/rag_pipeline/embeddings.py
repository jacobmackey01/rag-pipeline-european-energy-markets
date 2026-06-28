"""Local embedding wrapper around sentence-transformers.

Embeddings are normalized so cosine distance in Chroma behaves as intended and
query/document vectors live in the same semantic space.
"""

from __future__ import annotations

from collections.abc import Sequence


class LocalEmbedder:
    """Lazy-loading wrapper for the local MiniLM embedding model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Load the model only when an embed call actually needs it."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized vectors for Chroma cosine retrieval."""
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return embeddings.tolist()
