# =============================================================================
# generation.py — Call the LLM to produce a grounded, source-cited answer.
#
# This is the "G" in RAG (Generation). It takes the retrieved chunks, wraps them
# in the grounding prompt, sends them to OpenAI, and then double-checks the
# model's citations before returning.
# =============================================================================

from __future__ import annotations

import os

# The official OpenAI Python client.
from openai import OpenAI

from rag_pipeline.config import AppConfig, REFUSAL_MESSAGE
from rag_pipeline.grounding import GROUNDING_INSTRUCTION, build_grounded_prompt, citation_check
from rag_pipeline.store import RetrievedChunk


# Generate an answer for `question` using ONLY `chunks` as evidence, then run a
# citation-integrity check. Returns a dict with the answer and the check result.
def answer_from_context(
    config: AppConfig,
    question: str,
    chunks: list[RetrievedChunk],
) -> dict[str, object]:
    # If retrieval found nothing, there's no evidence to answer from — refuse
    # immediately and DON'T spend money/time on an LLM call.
    if not chunks:
        return {
            "answer": REFUSAL_MESSAGE,
            "citation_check": True,
            "citation_check_message": "No context was retrieved.",
        }

    # Fail clearly if the API key isn't configured, instead of getting a cryptic
    # error from deep inside the OpenAI client.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.local or the environment.")

    # Create the OpenAI client (it reads OPENAI_API_KEY from the environment).
    client = OpenAI()
    # Call the Responses API with:
    response = client.responses.create(
        # which model to use (from config / env),
        model=config.llm_model,
        # the system instruction (our anti-hallucination contract),
        instructions=GROUNDING_INSTRUCTION,
        # the user prompt (retrieved context + the question),
        input=build_grounded_prompt(question, chunks),
        # a hard cap on answer length to control cost and runaway output.
        max_output_tokens=600,
    )
    # Pull the plain-text answer out of the response and trim whitespace.
    answer = response.output_text.strip()

    # Verify the model only cited filenames it was actually given.
    passed, message = citation_check(answer, chunks)
    # Return the answer plus the outcome of that integrity check.
    return {
        "answer": answer,
        "citation_check": passed,
        "citation_check_message": message,
    }
