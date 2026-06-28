# Scripted validation checks for grounding, refusal, retrieval, and citations.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_pipeline.config import AppConfig, REFUSAL_MESSAGE
from rag_pipeline.generation import answer_from_context
from rag_pipeline.store import RetrievedChunk, retrieve


# One validation scenario and the assertions expected from it.
@dataclass(frozen=True)
class ValidationCase:
    name: str
    question: str
    expected_sources: tuple[str, ...] = ()
    expected_phrases: tuple[str, ...] = ()
    must_refuse: bool = False
    generate_answer: bool = True


# Serializable result for one validation case.
@dataclass(frozen=True)
class ValidationResult:
    name: str
    question: str
    passed: bool
    answer: str
    retrieved_sources: list[str]
    retrieved_labels: list[str]
    checks: dict[str, bool]
    notes: list[str]


# Cases are deliberately small and inspectable so failures can be explained.
VALIDATION_CASES = (
    ValidationCase(
        name="grounding_acer_2026_market_developments",
        question="What does ACER say about key developments in European electricity and gas markets in 2026?",
        expected_sources=("acer-gas-electricity-key-developments-2026.pdf",),
        expected_phrases=("Russian gas imports", "LNG market", "network codes"),
    ),
    ValidationCase(
        name="grounding_acer_see_cross_zonal_capacity",
        question="What does ACER say about cross-zonal capacity and flexibility in Southeast Europe?",
        expected_sources=("acer-see-cross-zonal-capacity-flexibility-2026.pdf",),
        expected_phrases=("price spikes", "Greece and Italy", "cross-zonal capacity"),
    ),
    ValidationCase(
        name="refusal_plausible_absent_poland_peak_demand",
        question=(
            "According to the provided ENTSO-E and ACER documents, what is the "
            "projected Winter 2025-2026 peak electricity demand for Poland in GW?"
        ),
        must_refuse=True,
    ),
    ValidationCase(
        name="retrieval_quality_entsoe_winter_outlook",
        question="What does ENTSO-E say in the Winter Outlook 2025-2026 about European adequacy?",
        expected_sources=("entsoe-winter-outlook-2025-2026.pdf",),
        generate_answer=False,
    ),
)


# Check whether each expected source PDF appeared in top-k retrieval.
def _source_check(chunks: list[RetrievedChunk], expected_sources: tuple[str, ...]) -> bool:
    if not expected_sources:
        return True
    retrieved = {chunk.source for chunk in chunks}
    return all(source in retrieved for source in expected_sources)


# Check for specific answer details, not just generic corpus words.
def _phrase_check(answer: str, phrases: tuple[str, ...]) -> bool:
    lower_answer = answer.lower()
    return all(phrase.lower() in lower_answer for phrase in phrases)


# Run all scripted validation cases and collect structured results.
def run_validation(config: AppConfig, top_k: int = 4) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for case in VALIDATION_CASES:
        # Retrieval runs for every case so source quality is always visible.
        chunks = retrieve(config, case.question, top_k=top_k)
        retrieved_sources = [chunk.source for chunk in chunks]
        retrieved_labels = [chunk.citation_label for chunk in chunks]
        notes: list[str] = []

        retrieval_ok = _source_check(chunks, case.expected_sources)
        checks: dict[str, bool] = {"retrieval_expected_source": retrieval_ok}

        answer = ""
        if case.generate_answer:
            generated = answer_from_context(config, case.question, chunks)
            answer = str(generated["answer"])
            checks["citation_integrity"] = bool(generated["citation_check"])
            if not generated["citation_check"]:
                notes.append(str(generated["citation_check_message"]))

            if case.must_refuse:
                # The refusal test is exact-match by design: no hedging or citations.
                checks["refusal_exact"] = answer.strip() == REFUSAL_MESSAGE
            else:
                checks["expected_phrases"] = _phrase_check(answer, case.expected_phrases)
        else:
            answer = "(generation skipped; retrieval spot check only)"

        passed = all(checks.values())
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
    return results


# Write validation output for auditing outside the terminal.
def write_validation_results(results: list[ValidationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
