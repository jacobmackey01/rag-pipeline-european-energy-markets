from __future__ import annotations

import os

from openai import OpenAI

from rag_pipeline.config import AppConfig, REFUSAL_MESSAGE
from rag_pipeline.grounding import GROUNDING_INSTRUCTION, build_grounded_prompt, citation_check
from rag_pipeline.store import RetrievedChunk


def answer_from_context(
    config: AppConfig,
    question: str,
    chunks: list[RetrievedChunk],
) -> dict[str, object]:
    if not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "citation_check": True,
            "citation_check_message": "No context was retrieved.",
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.local or the environment.")

    client = OpenAI()
    response = client.responses.create(
        model=config.llm_model,
        instructions=GROUNDING_INSTRUCTION,
        input=build_grounded_prompt(question, chunks),
        temperature=config.llm_temperature,
        max_output_tokens=600,
    )
    answer = response.output_text.strip()
    passed, message = citation_check(answer, chunks)
    return {
        "answer": answer,
        "citation_check": passed,
        "citation_check_message": message,
    }
