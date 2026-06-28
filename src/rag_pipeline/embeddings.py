# =============================================================================
# embeddings.py — Turn text into vectors (the heart of "semantic" search).
#
# An *embedding* is a list of numbers (a vector) that represents the MEANING of
# a piece of text as a point in space. Texts with similar meaning get vectors
# pointing in similar directions, so "renewable subsidy" lands near "green
# energy support" even though they share no words. This file wraps the local
# model that produces those vectors.
# =============================================================================

# Lazy type-hint evaluation (see config.py for the explanation).
from __future__ import annotations

# `Sequence` is a general type meaning "an ordered collection you can iterate"
# (a list, a tuple, etc.). Using it as a hint says "accept any sequence of str".
from collections.abc import Sequence


# A thin wrapper around the sentence-transformers model. Its main jobs are:
#   (1) load the model LAZILY (only when first used), and
#   (2) keep our embedding settings (normalisation) in one place.
class LocalEmbedder:
    # Constructor: store the model name but DON'T load the model yet. Loading
    # reads ~90MB and is slow, so we defer it until it's actually needed.
    def __init__(self, model_name: str) -> None:
        # Remember which model to load later.
        self.model_name = model_name
        # `_model` starts as None to mean "not loaded yet". The leading
        # underscore is a Python convention for "internal, don't touch directly".
        self._model = None

    # A @property lets us access `embedder.model` like a normal attribute while
    # actually running code behind the scenes — here, load-on-first-use.
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
        # Hand back the cached, ready model.
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
