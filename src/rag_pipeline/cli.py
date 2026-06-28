# =============================================================================
# cli.py — The command-line interface (the `rag-pipeline` command).
#
# This parses the sub-command you type (download / ingest / retrieve / ask /
# validate) and calls the matching pipeline function. It's the user-facing entry
# point wired up in pyproject.toml under [project.scripts].
# =============================================================================

from __future__ import annotations

# `argparse` is Python's standard library for building command-line interfaces.
import argparse
import json
# `asdict` converts a dataclass instance into a plain dict (for JSON output).
from dataclasses import asdict

from rag_pipeline.config import AppConfig
from rag_pipeline.documents import download_sources
from rag_pipeline.pipeline import ask_question, build_index
from rag_pipeline.store import retrieve
from rag_pipeline.validation import run_validation, write_validation_results


# Helper to print retrieved chunks compactly (used by `retrieve` and `ask --show-chunks`).
def _print_chunks(chunks) -> None:
    # Number the chunks from 1.
    for index, chunk in enumerate(chunks, start=1):
        # Header line: the citation label and the similarity distance (4 d.p.).
        print(f"\n[{index}] {chunk.citation_label} | distance={chunk.distance:.4f}")
        # Flatten newlines so each preview is one block, then truncate to 700 chars.
        preview = chunk.text.replace("\n", " ")
        print(preview[:700] + ("..." if len(preview) > 700 else ""))


# The entry point: define the commands, parse arguments, and dispatch.
def main() -> None:
    # The top-level parser, with a description shown in --help.
    parser = argparse.ArgumentParser(description="Grounded RAG over public PDFs.")
    # Sub-commands live under `dest="command"`; requiring one means a bare
    # `rag-pipeline` with no command shows an error instead of doing nothing.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `download` — fetch the PDFs and verify their checksums.
    download_parser = subparsers.add_parser("download", help="Download PDF corpus.")
    # Optional flag to re-download even if the files already exist.
    download_parser.add_argument("--overwrite", action="store_true")

    # `ingest` — extract, chunk, embed, and store the PDFs.
    ingest_parser = subparsers.add_parser("ingest", help="Build Chroma index.")
    # Optional flag to wipe and rebuild the collection from scratch.
    ingest_parser.add_argument("--reset", action="store_true")

    # `retrieve` — show the top-k chunks for a question WITHOUT calling the LLM
    # (great for debugging retrieval quality cheaply, with no API cost).
    retrieve_parser = subparsers.add_parser("retrieve", help="Show top-k retrieved chunks.")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--top-k", type=int, default=4)

    # `ask` — the full RAG flow: retrieve + generate + citation check.
    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=4)
    # Also print the retrieved chunks alongside the answer.
    ask_parser.add_argument("--show-chunks", action="store_true")
    # Emit machine-readable JSON instead of plain text.
    ask_parser.add_argument("--json", action="store_true")

    # `validate` — run the anti-hallucination test suite.
    validate_parser = subparsers.add_parser("validate", help="Run grounding/refusal validation.")
    validate_parser.add_argument("--top-k", type=int, default=4)
    # Where to write the detailed JSON results.
    validate_parser.add_argument(
        "--output",
        default="validation/results.json",
        help="Where to write detailed validation JSON.",
    )

    # Parse whatever the user typed into an `args` object.
    args = parser.parse_args()
    # Build the configuration (loads .env files, reads env vars).
    config = AppConfig.from_env()

    # --- Dispatch to the chosen command ---

    # download: fetch PDFs and print where each one landed.
    if args.command == "download":
        paths = download_sources(config, overwrite=args.overwrite)
        for path in paths:
            print(path)
        return

    # ingest: build the index and report how many chunks were stored.
    if args.command == "ingest":
        count = build_index(config, reset=args.reset)
        print(f"Indexed {count} chunks into {config.chroma_dir}")
        return

    # retrieve: print the top-k chunks for inspection.
    if args.command == "retrieve":
        chunks = retrieve(config, args.question, top_k=args.top_k)
        _print_chunks(chunks)
        return

    # ask: run full RAG and print the answer (and optionally the chunks / JSON).
    if args.command == "ask":
        result = ask_question(config, args.question, top_k=args.top_k)
        # JSON mode: serialise everything, converting chunk dataclasses to dicts.
        if args.json:
            serializable = {
                "answer": result["answer"],
                "citation_check": result["citation_check"],
                "citation_check_message": result["citation_check_message"],
                "retrieved_chunks": [asdict(chunk) for chunk in result["retrieved_chunks"]],
            }
            print(json.dumps(serializable, indent=2))
            return
        # Plain mode: print the answer, then the citation-check summary line.
        print(result["answer"])
        print(f"\nCitation check: {result['citation_check_message']}")
        # Optionally also print the supporting chunks.
        if args.show_chunks:
            _print_chunks(result["retrieved_chunks"])
        return

    # validate: run the test suite, save results, and print a pass/fail report.
    if args.command == "validate":
        results = run_validation(config, top_k=args.top_k)
        # Persist the detailed JSON to disk.
        write_validation_results(results, config.root_dir / args.output)
        # Print a per-case summary to the terminal.
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name}")
            # List each individual check and its result.
            for name, passed in result.checks.items():
                print(f"  - {name}: {'PASS' if passed else 'FAIL'}")
            # Print any notes attached to this case.
            if result.notes:
                for note in result.notes:
                    print(f"  - note: {note}")
        # If any case failed, exit with code 1 so CI/automation can detect it.
        failures = [result for result in results if not result.passed]
        if failures:
            raise SystemExit(1)
        return


# Standard Python idiom: only run main() when this file is executed directly
# (e.g. `python -m rag_pipeline.cli`), not when it's imported by another module.
if __name__ == "__main__":
    main()
