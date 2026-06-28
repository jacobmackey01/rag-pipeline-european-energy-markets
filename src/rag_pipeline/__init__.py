# =============================================================================
# __init__.py — Marks this folder as the `rag_pipeline` Python package.
#
# Any directory containing an __init__.py file is treated by Python as an
# importable "package", which is what makes `from rag_pipeline import ...` work.
# This file is almost empty on purpose: it just records the version number.
# =============================================================================

# Grounded RAG pipeline over public European energy-market PDF documents.
# The package exposes CLI-accessible pieces for downloading a corpus, indexing
# it with local embeddings, retrieving relevant chunks, and generating
# source-cited answers with an explicit refusal path.

# `__all__` lists the names that `from rag_pipeline import *` should export.
# Keeping it minimal avoids leaking internal names into other modules.
__all__ = ["__version__"]

# The package version string. Kept in sync with the version in pyproject.toml.
__version__ = "0.1.0"
