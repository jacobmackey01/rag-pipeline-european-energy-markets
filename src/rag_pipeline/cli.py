from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag_pipeline.config import AppConfig
from rag_pipeline.documents import download_sources
from rag_pipeline.pipeline import ask_question, build_index
from rag_pipeline.store import retrieve
from rag_pipeline.validation import run_validation, write_validation_results


def _print_chunks(chunks) -> None:
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n[{index}] {chunk.citation_label} | distance={chunk.distance:.4f}")
        preview = chunk.text.replace("\n", " ")
        print(preview[:700] + ("..." if len(preview) > 700 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded RAG over public PDFs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download PDF corpus.")
    download_parser.add_argument("--overwrite", action="store_true")

    ingest_parser = subparsers.add_parser("ingest", help="Build Chroma index.")
    ingest_parser.add_argument("--reset", action="store_true")

    retrieve_parser = subparsers.add_parser("retrieve", help="Show top-k retrieved chunks.")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--top-k", type=int, default=4)

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--show-chunks", action="store_true")
    ask_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Run grounding/refusal validation.")
    validate_parser.add_argument("--top-k", type=int, default=4)
    validate_parser.add_argument(
        "--output",
        default="validation/results.json",
        help="Where to write detailed validation JSON.",
    )

    args = parser.parse_args()
    config = AppConfig.from_env()

    if args.command == "download":
        paths = download_sources(config, overwrite=args.overwrite)
        for path in paths:
            print(path)
        return

    if args.command == "ingest":
        count = build_index(config, reset=args.reset)
        print(f"Indexed {count} chunks into {config.chroma_dir}")
        return

    if args.command == "retrieve":
        chunks = retrieve(config, args.question, top_k=args.top_k)
        _print_chunks(chunks)
        return

    if args.command == "ask":
        result = ask_question(config, args.question, top_k=args.top_k)
        if args.json:
            serializable = {
                "answer": result["answer"],
                "citation_check": result["citation_check"],
                "citation_check_message": result["citation_check_message"],
                "retrieved_chunks": [asdict(chunk) for chunk in result["retrieved_chunks"]],
            }
            print(json.dumps(serializable, indent=2))
            return
        print(result["answer"])
        print(f"\nCitation check: {result['citation_check_message']}")
        if args.show_chunks:
            _print_chunks(result["retrieved_chunks"])
        return

    if args.command == "validate":
        results = run_validation(config, top_k=args.top_k)
        write_validation_results(results, config.root_dir / args.output)
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name}")
            for name, passed in result.checks.items():
                print(f"  - {name}: {'PASS' if passed else 'FAIL'}")
            if result.notes:
                for note in result.notes:
                    print(f"  - note: {note}")
        failures = [result for result in results if not result.passed]
        if failures:
            raise SystemExit(1)
        return


if __name__ == "__main__":
    main()
