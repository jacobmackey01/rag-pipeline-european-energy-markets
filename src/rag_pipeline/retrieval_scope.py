"""Resolve explicitly named seasonal reports before semantic ranking."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_pipeline.documents import SourceDocument


OUTLOOK_REFERENCE = re.compile(
    r"\b(winter|summer)\s+(?:outlook(?:\s+for)?\s+(\d{4})(?:\s*[-–—‑/]\s*(\d{4}|\d{2}))?"
    r"|(\d{4})(?:\s*[-–—‑/]\s*(\d{4}|\d{2}))?\s+outlook)\b",
    re.IGNORECASE,
)


class MissingReportError(ValueError):
    """The question requests an edition absent from the corpus."""


def _report_key(match: re.Match[str]) -> tuple[str, int, int | None]:
    start = int(match.group(2) or match.group(4))
    end_text = match.group(3) if match.group(2) else match.group(5)
    end = int(end_text) if end_text else None
    if end is not None and end < 100:
        end += (start // 100) * 100
    return match.group(1).lower(), start, end


def source_scope(
    question: str,
    catalog: Sequence[SourceDocument],
    sources: Sequence[str] | None = None,
) -> list[str] | None:
    """Explicit filenames override inferred dated Outlook references.

    Broad questions have no scope. An unavailable named edition raises a typed
    error that retrieval converts to no context and the pipeline's refusal.
    """
    if sources is not None:
        if not sources:
            raise ValueError("At least one source filename is required.")
        unknown = set(sources) - {item.filename for item in catalog}
        if unknown:
            raise ValueError(f"Source(s) not in the manifest: {sorted(unknown)}")
        return list(dict.fromkeys(sources))

    requested = list(OUTLOOK_REFERENCE.finditer(question))
    if not requested:
        return None
    selected: list[str] = []
    for reference in requested:
        matching = [item.filename for item in catalog
                    if (title := OUTLOOK_REFERENCE.search(item.title))
                    and _report_key(title) == _report_key(reference)]
        if not matching:
            raise MissingReportError(f"Report not in the manifest: {reference.group(0)}")
        selected.extend(matching)
    return list(dict.fromkeys(selected))
