"""Create normalized local embeddings for document and query text."""

# Defer type-hint evaluation.
from __future__ import annotations

# Accept any ordered sequence of strings.
from collections.abc import Sequence


# Load the sentence-transformers model on first use and keep normalization
# settings in one place.
class LocalEmbedder:
    # Store the model name and defer loading until first use.
    def __init__(self, model_name: str) -> None:
        # Remember which model to load later.
        self.model_name = model_name
        # `_model` starts as None to mean "not loaded yet". The leading
        # underscore is a Python convention for "internal, don't touch directly".
        self._model = None

    # Load and cache the model on first access.
    @property
    def model(self):
        # If the model hasn't been loaded yet, load it now.
        if self._model is None:
            # Import inside the method so just importing this file stays cheap;
            # the heavy library only loads when embeddings are really needed.
            from sentence_transformers import SentenceTransformer

            # Download (first run) and load the model, then cache it on self so
            # later calls reuse the same in-memory model.
            self._model = SentenceTransformer(self.model_name)
        # Return the cached model.
        return self._model

    # Convert a batch of texts into a list of embedding vectors.
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # `model.encode` does the actual text -> vectors work.
        embeddings = self.model.encode(
            # Force to a list in case a tuple or other sequence was passed.
            list(texts),
            # Normalise every vector to length 1. This matters because with
            # unit-length vectors, cosine similarity becomes a simple dot
            # product, and query and document vectors stay on the same scale so
            # Chroma's cosine distance behaves as intended.
            normalize_embeddings=True,
            # Only show a progress bar for big batches (>32 items), to avoid
            # noisy output when embedding a single question at query-time.
            show_progress_bar=len(texts) > 32,
        )
        # `encode` returns a NumPy array; convert it to plain Python lists of
        # floats, which is the format Chroma expects to store.
        return embeddings.tolist()
