from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_pipeline.config import AppConfig, REFUSAL_MESSAGE
from rag_pipeline.generation import answer_from_context
from rag_pipeline.store import RetrievedChunk, retrieve


@dataclass(frozen=True)
class ValidationCase:
    name: str
    question: str
    expected_sources: tuple[str, ...] = ()
    expected_phrases: tuple[str, ...] = ()
    must_refuse: bool = False
    generate_answer: bool = True


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


VALIDATION_CASES = (
    ValidationCase(
        name="grounding_acer_2026_market_developments",
        question="What does ACER say about key developments in European electricity and gas markets in 2026?",
        expected_sources=("acer-gas-electricity-key-developments-2026.pdf",),
        expected_phrases=("electricity", "gas"),
    ),
    ValidationCase(
        name="grounding_acer_see_cross_zonal_capacity",
        question="What does ACER say about cross-zonal capacity and flexibility in Southeast Europe?",
        expected_sources=("acer-see-cross-zonal-capacity-flexibility-2026.pdf",),
        expected_phrases=("cross-zonal", "Southeast Europe"),
    ),
    ValidationCase(
        name="refusal_absent_cobblestone_revenue",
        question="According to the provided documents, what was Cobblestone Energy's 2026 revenue forecast?",
        must_refuse=True,
    ),
    ValidationCase(
        name="retrieval_quality_entsoe_winter_outlook",
        question="What does ENTSO-E say in the Winter Outlook 2025-2026 about European adequacy?",
        expected_sources=("entsoe-winter-outlook-2025-2026.pdf",),
        generate_answer=False,
    ),
)


def _source_check(chunks: list[RetrievedChunk], expected_sources: tuple[str, ...]) -> bool:
    if not expected_sources:
        return True
    retrieved = {chunk.source for chunk in chunks}
    return all(source in retrieved for source in expected_sources)


def _phrase_check(answer: str, phrases: tuple[str, ...]) -> bool:
    lower_answer = answer.lower()
    return all(phrase.lower() in lower_answer for phrase in phrases)


def run_validation(config: AppConfig, top_k: int = 4) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for case in VALIDATION_CASES:
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


def write_validation_results(results: list[ValidationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
