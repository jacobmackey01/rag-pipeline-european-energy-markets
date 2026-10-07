"""Compare filename validation and optional support review on fixed examples."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from rag_pipeline.claim_support import assess_claim_support
from rag_pipeline.config import AppConfig
from rag_pipeline.grounding import citation_check
from rag_pipeline.store import RetrievedChunk


def evaluate_support(config: AppConfig, cases_path: Path) -> dict[str, object]:
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    results = []
    for case in fixture["cases"]:
        chunks = [RetrievedChunk(**fixture["evidence"][key]) for key in case["evidence_ids"]]
        started = perf_counter()
        baseline, _ = citation_check(case["answer"], chunks)
        baseline_latency = perf_counter() - started
        support = assess_claim_support(config, case["answer"], chunks)
        results.append({
            **case, "baseline_accepted": baseline,
            "baseline_latency_seconds": baseline_latency,
            "augmented_accepted": baseline and support["passed"] is True,
            "claim_support": support,
        })

    def counts(accepted_key: str) -> dict[str, int]:
        return {
            "supported_accepted": sum(r["expected_supported"] and r[accepted_key] for r in results),
            "unsupported_accepted": sum(not r["expected_supported"] and r[accepted_key] for r in results),
            "supported_flagged": sum(r["expected_supported"] and not r[accepted_key] for r in results),
            "unsupported_flagged": sum(not r["expected_supported"] and not r[accepted_key] for r in results),
        }

    unavailable = sum(r["claim_support"]["status"] == "unavailable" for r in results)
    return {
        "label_status": fixture["label_status"],
        "model": config.decisions_model,
        "thresholds": {"supported": config.support_threshold, "unsupported": config.unsupported_threshold},
        "summary": {
            "cases": len(results), "unavailable_cases": unavailable,
            "baseline": counts("baseline_accepted"), "augmented": counts("augmented_accepted"),
            "decisions_latency_seconds": round(sum(r["claim_support"].get("latency_seconds", 0) for r in results), 4),
            "decisions_input_tokens": sum(r["claim_support"].get("usage", {}).get("input_tokens", 0) for r in results),
        },
        "results": results,
    }
