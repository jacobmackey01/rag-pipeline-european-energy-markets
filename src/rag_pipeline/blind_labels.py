"""Prepare unscored, reproducible claim rows for human labelling."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from rag_pipeline.claim_support import split_claim_units
from rag_pipeline.config import REFUSAL_MESSAGE
from rag_pipeline.grounding import extract_cited_sources


FIELDS = ("unit_id", "human_label", "review_notes", "split", "question", "claim", "cited_sources", "evidence")


def prepare_blind_rows(answers: list[dict], count: int = 50, seed: int = 20261007) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("Unit count must be positive.")
    candidates = []
    seen = set()
    for answer in answers:
        if answer["answer"].strip() == REFUSAL_MESSAGE:
            continue
        group = hashlib.sha256(answer["question"].encode()).hexdigest()
        for unit in split_claim_units(answer["answer"]):
            cited = sorted(extract_cited_sources(unit))
            evidence = [c for c in answer["retrieved_chunks"] if c["source"] in cited]
            evidence_text = "\n\n".join(
                f"Source: {c['source']} | PDF pages {c['page_start']}-{c['page_end']} | chunk {c['id']}\n"
                f"URL: {c['source_url']}\n{c['text']}" for c in evidence
            )
            identity = json.dumps([answer["question"], unit, evidence_text], ensure_ascii=False)
            digest = hashlib.sha256(identity.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append({"unit_id":"unit_" + digest[:16], "human_label":"", "review_notes":"",
                               "question":answer["question"], "claim":unit,
                               "cited_sources":"; ".join(cited), "evidence":evidence_text, "group":group})
    if len(candidates) < count:
        raise ValueError(f"Need {count} claim units; only {len(candidates)} available.")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:count]
    groups = sorted({row["group"] for row in selected})
    rng.shuffle(groups)
    development = set(groups[:max(1, round(len(groups) * 0.6))])
    for row in selected:
        row["split"] = "development" if row.pop("group") in development else "test"
    return selected


def write_blind_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            # Keep imported source text from being interpreted as spreadsheet formulas.
            writer.writerow({field:("'" + value if value.startswith(("=", "+", "-", "@")) else value)
                             for field in FIELDS for value in [row[field]]})
