# =============================================================================
# validation.py — The anti-hallucination test suite (the headline feature).
#
# This file defines a handful of fixed test cases and runs them through the REAL
# pipeline to prove the system behaves: it answers and cites correctly when the
# answer exists, and refuses (instead of inventing) when it doesn't.
# =============================================================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_pipeline.config import AppConfig, REFUSAL_MESSAGE
from rag_pipeline.generation import answer_from_context
from rag_pipeline.store import RetrievedChunk, retrieve


# Describes ONE validation scenario and what we expect from it.
@dataclass(frozen=True)
class ValidationCase:
    # A short identifier for the case.
    name: str
    # The question to ask.
    question: str
    # PDF(s) we expect retrieval to surface (empty tuple = don't check sources).
    expected_sources: tuple[str, ...] = ()
    # Substrings the answer should contain (proves it found the real detail).
    expected_phrases: tuple[str, ...] = ()
    # True if this case must return the exact refusal string.
    must_refuse: bool = False
    # False to skip the (paid) LLM call and only check retrieval.
    generate_answer: bool = True


# The structured outcome of running one case (this is what gets saved to JSON).
@dataclass(frozen=True)
class ValidationResult:
    name: str
    question: str
    passed: bool
    answer: str
    retrieved_sources: list[str]
    retrieved_labels: list[str]
    # A dict mapping each check name -> pass/fail boolean.
    checks: dict[str, bool]
    # Any explanatory notes (e.g. why a citation check failed).
    notes: list[str]


# The actual test cases. Kept small and hand-picked so each result is explainable.
VALIDATION_CASES = (
    # Case 1: a grounding test — the answer IS in the corpus; expect that PDF and
    # a few specific phrases that prove the real detail was found.
    ValidationCase(
        name="grounding_acer_2026_market_developments",
        question="What does ACER say about key developments in European electricity and gas markets in 2026?",
        expected_sources=("acer-gas-electricity-key-developments-2026.pdf",),
        expected_phrases=("Russian gas imports", "LNG market", "network codes"),
    ),
    # Case 2: another grounding test on a different document and topic.
    ValidationCase(
        name="grounding_acer_see_cross_zonal_capacity",
        question="What does ACER say about cross-zonal capacity and flexibility in Southeast Europe?",
        expected_sources=("acer-see-cross-zonal-capacity-flexibility-2026.pdf",),
        expected_phrases=("price spikes", "Greece and Italy", "cross-zonal capacity"),
    ),
    # Case 3: the REFUSAL test — a plausible but ABSENT question. The whole point
    # of the anti-hallucination story: it must refuse, not invent a number.
    ValidationCase(
        name="refusal_plausible_absent_poland_peak_demand",
        question=(
            "According to the provided ENTSO-E and ACER documents, what is the "
            "projected Winter 2025-2026 peak electricity demand for Poland in GW?"
        ),
        must_refuse=True,
    ),
    # Case 4: a retrieval-only spot check (no LLM call) — does the right PDF rank
    # in the top-k for this question?
    ValidationCase(
        name="retrieval_quality_entsoe_winter_outlook",
        question="What does ENTSO-E say in the Winter Outlook 2025-2026 about European adequacy?",
        expected_sources=("entsoe-winter-outlook-2025-2026.pdf",),
        generate_answer=False,
    ),
)


# Check that all expected source PDFs appear among the retrieved chunks.
def _source_check(chunks: list[RetrievedChunk], expected_sources: tuple[str, ...]) -> bool:
    # No expectation set -> nothing to check, so pass.
    if not expected_sources:
        return True
    # The set of sources we actually retrieved.
    retrieved = {chunk.source for chunk in chunks}
    # Every expected source must be present.
    return all(source in retrieved for source in expected_sources)


# Check that the answer contains every expected phrase (case-insensitive).
def _phrase_check(answer: str, phrases: tuple[str, ...]) -> bool:
    # Lower-case the answer once for case-insensitive matching.
    lower_answer = answer.lower()
    # All phrases must be present as substrings.
    return all(phrase.lower() in lower_answer for phrase in phrases)


# Run every validation case through the real pipeline and collect the results.
def run_validation(config: AppConfig, top_k: int = 4) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    # Process each case in turn.
    for case in VALIDATION_CASES:
        # Always run retrieval, so source quality is visible even for refusal cases.
        chunks = retrieve(config, case.question, top_k=top_k)
        # The sources and human-readable labels of what we retrieved.
        retrieved_sources = [chunk.source for chunk in chunks]
        retrieved_labels = [chunk.citation_label for chunk in chunks]
        # Notes for anything noteworthy (e.g. a failed citation message).
        notes: list[str] = []

        # Check #1 (always): did the expected source(s) get retrieved?
        retrieval_ok = _source_check(chunks, case.expected_sources)
        checks: dict[str, bool] = {"retrieval_expected_source": retrieval_ok}

        # Default answer text.
        answer = ""
        # Only call the LLM if this case wants a generated answer.
        if case.generate_answer:
            # Generate the grounded answer and its citation-integrity result.
            generated = answer_from_context(config, case.question, chunks)
            answer = str(generated["answer"])
            # Check #2: did the citation-integrity check pass?
            checks["citation_integrity"] = bool(generated["citation_check"])
            # If it failed, record the reason as a note.
            if not generated["citation_check"]:
                notes.append(str(generated["citation_check_message"]))

            # Check #3 depends on the case type:
            if case.must_refuse:
                # Refusal case: the answer must be EXACTLY the refusal string.
                checks["refusal_exact"] = answer.strip() == REFUSAL_MESSAGE
            else:
                # Grounding case: the answer must contain all expected phrases.
                checks["expected_phrases"] = _phrase_check(answer, case.expected_phrases)
        else:
            # Retrieval-only case: note that generation was intentionally skipped.
            answer = "(generation skipped; retrieval spot check only)"

        # The case passes only if EVERY check passed.
        passed = all(checks.values())
        # Record the full structured result.
        results.append(
            ValidationResult(
                name=case.name,
                question=case.question,
                passed=passed,
                answer=answer,
                retrieved_sources=retrieved_sources,
                retrieved_labels=retrieved_labels,
                checks=checks,
                notes=notes,
            )
        )
    # Return all results.
    return results


# Write the validation results to a JSON file for later inspection/auditing.
def write_validation_results(results: list[ValidationResult], path: Path) -> None:
    # Make sure the output folder exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    # `asdict` turns each dataclass result into a plain dict; dump as pretty JSON.
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
