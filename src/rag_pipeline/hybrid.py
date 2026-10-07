"""Small-corpus hybrid ranking over stored embeddings and source text."""

from __future__ import annotations

import math
import re
from collections import Counter


def terms(text: str) -> list[str]:
    text = re.sub(r"\b(\d{4})\s*[-–—‑/]\s*(\d{2})\b",
                  lambda m: m.group(1) + " " + m.group(1)[:2] + m.group(2), text)
    return re.findall(r"[a-z0-9]+", text.casefold())


def bm25(query: str, documents: list[str]) -> list[float]:
    counts = [Counter(terms(doc)) for doc in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) if lengths else 0
    scores = [0.0] * len(documents)
    if not average:
        return scores
    for term in set(terms(query)):
        frequency = sum(term in count for count in counts)
        if not frequency:
            continue
        idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        for i, count in enumerate(counts):
            tf = count[term]
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[i] / average))
    return scores


def rank_records(query: str, query_vector, records: dict) -> list[tuple[int, float, float]]:
    """Return record index, cosine distance, and combined retrieval score.

    Title BM25 has double the body weight, making source identity available to
    lexical search as well as contextual embeddings. Semantic and normalized
    lexical scores each receive half the final weight. No label-based tuning.
    """
    titles = [(meta or {}).get("title", "") for meta in records["metadatas"]]
    bodies = [doc or "" for doc in records["documents"]]
    title_scores, body_scores = bm25(query, titles), bm25(query, bodies)
    lexical = [2 * title + body for title, body in zip(title_scores, body_scores)]
    maximum = max(lexical, default=0)
    query_norm = math.sqrt(sum(float(x) ** 2 for x in query_vector))
    ranked = []
    for i, vector in enumerate(records["embeddings"]):
        if len(vector) != len(query_vector):
            raise RuntimeError("Index embedding dimensions differ from the query model. Rebuild the index.")
        norm = math.sqrt(sum(float(x) ** 2 for x in vector))
        cosine = sum(float(a) * float(b) for a, b in zip(query_vector, vector)) / (query_norm * norm) if norm and query_norm else 0
        cosine = max(-1.0, min(1.0, cosine))
        semantic = (cosine + 1) / 2
        score = 0.5 * semantic + 0.5 * lexical[i] / maximum if maximum else semantic
        ranked.append((i, 1 - cosine, score))
    return sorted(ranked, key=lambda item: (-item[2], item[1], records["ids"][item[0]]))
