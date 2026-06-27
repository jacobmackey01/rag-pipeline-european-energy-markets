from __future__ import annotations

from collections.abc import Sequence


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return embeddings.tolist()
