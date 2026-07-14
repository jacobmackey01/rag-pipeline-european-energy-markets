# =============================================================================
# config.py — One place that holds every runtime setting for the pipeline.
#
# Instead of scattering file paths, model names, and tuning numbers across the
# codebase, we gather them into a single immutable object (AppConfig). It also
# includes a tiny loader for ".env" files so secrets like the API key live
# outside the code.
# =============================================================================

# Makes every type hint in this file lazily evaluated (treated as a string).
# It lets us write modern hint syntax like `str | None` and avoids some
# import-ordering headaches. Common first line in modern Python modules.
from __future__ import annotations

# `os` gives access to environment variables: os.getenv to read them and
# os.environ (a dict) to read/write the live process environment.
import os
# `dataclass` auto-generates the boilerplate (__init__, __repr__, equality) for
# a class that mainly bundles data together — perfect for a settings object.
from dataclasses import dataclass
# `Path` is an object-oriented, cross-platform way to build filesystem paths
# (it handles Windows "\" vs Unix "/" for us).
from pathlib import Path


# The EXACT text the model must return when an answer isn't in the documents.
# Defining it once, as a single constant, means the prompt, the generator, and
# the validation test all refer to the same string — if they drifted apart, a
# refusal test could "pass" while the user sees slightly different wording.
REFUSAL_MESSAGE = "Not found in the provided documents."


# Read a ".env"-style file and copy its KEY=VALUE lines into the environment,
# so the rest of the program can read them with os.getenv.
def load_env_file(path: Path) -> None:
    # If the file isn't there, quietly do nothing (it's optional).
    if not path.exists():
        return
    # Read the whole file as text and iterate over it line by line.
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        # Trim whitespace, and strip a leading BOM (the invisible
        # character) that some Windows editors add to the first line of a file.
        line = raw_line.strip().lstrip("\ufeff")
        # Skip blank lines, comments (# ...), and anything that isn't KEY=VALUE.
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Split on the FIRST "=" only, so values that themselves contain "="
        # (like an API token) stay intact on the right-hand side.
        key, value = line.split("=", 1)
        # Clean the key the same way (whitespace + possible BOM).
        key = key.strip().lstrip("\ufeff")
        # Clean the value: trim spaces, then remove surrounding quotes so both
        # KEY="value" and KEY=value end up identical.
        value = value.strip().strip('"').strip("'")
        # Only set it if it has a name AND isn't already in the environment.
        # "Don't override" means a real exported env var always beats the file.
        if key and key not in os.environ:
            os.environ[key] = value


# A frozen (read-only) dataclass holding every setting in one inspectable place.
# `frozen=True` means once an AppConfig is created it can't be changed, so config
# can't be accidentally mutated halfway through a run.
@dataclass(frozen=True)
class AppConfig:
    # The project root; every other path is built relative to this.
    root_dir: Path
    # Path to the JSON manifest of source PDFs (filenames, URLs, checksums).
    sources_path: Path
    # Folder where downloaded PDFs are stored.
    raw_dir: Path
    # Folder where the Chroma vector database persists itself to disk.
    chroma_dir: Path
    # Name of the collection (Chroma's version of a table) holding our chunks.
    collection_name: str
    # Hugging Face id of the local embedding model.
    embedding_model: str
    # The OpenAI model used to generate answers.
    llm_model: str
    # Target chunk size, measured in tokens.
    chunk_tokens: int
    # How many tokens consecutive chunks share (overlap).
    overlap_tokens: int

    # A "factory" constructor: build an AppConfig from environment variables,
    # with sensible defaults baked in. `@classmethod` means it's called on the
    # class (AppConfig.from_env()) rather than on an instance.
    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "AppConfig":
        # Use the given root, or the current working directory if none is given,
        # then `.resolve()` to make it an absolute, symlink-free path.
        root = (root_dir or Path.cwd()).resolve()

        # Load env files in priority order. .env.local is loaded FIRST; because
        # load_env_file never overrides an existing variable, whatever it sets
        # wins over .env (the fallback) loaded second.
        load_env_file(root / ".env.local")
        load_env_file(root / ".env")

        # Construct the immutable settings object. Each os.getenv(name, default)
        # reads an override from the environment or uses the default 2nd argument.
        return cls(
            root_dir=root,
            # Build the standard data paths off the root directory.
            sources_path=root / "data" / "sources.json",
            raw_dir=root / "data" / "raw",
            chroma_dir=root / "data" / "chroma",
            # Chroma collection name; override with RAG_COLLECTION if you want.
            collection_name=os.getenv("RAG_COLLECTION", "grounded_pdf_chunks"),
            # The local embedding model — small, fast, free, runs on CPU.
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            # The generation model; defaults to GPT-5.6 Luna (override OPENAI_MODEL).
            llm_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            # Chunk size and overlap, converted from env strings to integers.
            chunk_tokens=int(os.getenv("CHUNK_TOKENS", "220")),
            overlap_tokens=int(os.getenv("OVERLAP_TOKENS", "40")),
        )
