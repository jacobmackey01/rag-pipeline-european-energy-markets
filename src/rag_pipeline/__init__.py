"""Grounded RAG pipeline over public European energy-market PDFs.

The package supports corpus download, local embedding, chunk retrieval, grounded
answer generation, and an explicit refusal path.
"""

# `__all__` lists the names that `from rag_pipeline import *` should export.
# Keeping it minimal avoids leaking internal names into other modules.
__all__ = ["__version__"]

# The package version string. Kept in sync with the version in pyproject.toml.
__version__ = "0.1.0"
