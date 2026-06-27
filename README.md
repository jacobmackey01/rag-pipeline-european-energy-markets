# RAG Pipeline - Grounded QA

This project is a small, defensible retrieval-augmented generation pipeline over public European energy-market PDFs. It retrieves relevant document chunks, asks an LLM to answer only from those chunks, cites source filenames, and refuses with `Not found in the provided documents.` when the answer is absent.

## Corpus

Option A is used, but narrowed to the Cobblestone Energy story: public European power and gas market documents from ENTSO-E and ACER.

- ENTSO-E Summer Outlook 2026
- ENTSO-E Winter Outlook 2025-2026
- ACER Key Developments in European Electricity and Gas Markets 2026
- ACER Increasing Cross-Zonal Capacity and System Flexibility in Southeast Europe 2026
- ACER Key Developments in European Gas Wholesale Markets Winter 2025-2026

The PDFs are downloaded from the source URLs in `data/sources.json`. They are not committed to Git because they are public binary artifacts.

## Design Choices

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally. Embeddings map semantically similar text close together, so a query can match related wording even without exact keyword overlap.
- **Chunking:** default 220 token-like units with 40 overlap. The original target was roughly 500 tokens, but MiniLM has a short input window; keeping chunks below that limit avoids silently embedding truncated text. The overlap preserves meaning across boundaries.
- **Vector store:** ChromaDB with cosine distance and persistent local storage in `data/chroma`.
- **Generation:** OpenAI Responses API. The default model is `gpt-4.1-mini`, a lower-cost model suitable for this short grounded-answer task. Override with `OPENAI_MODEL`.
- **Anti-hallucination:** the prompt says to answer only from retrieved context, cite source filenames, and return exactly `Not found in the provided documents.` when the context does not contain the answer.
- **Validation:** the CLI includes grounding, refusal, retrieval-quality, and citation-source checks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Create `.env.local` with:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
```

## Run

Download the corpus:

```powershell
rag-pipeline download
```

Build the vector index:

```powershell
rag-pipeline ingest --reset
```

Ask a question:

```powershell
rag-pipeline ask "What does ACER say about electricity and gas market developments in 2026?" --show-chunks
```

Inspect retrieval without generation:

```powershell
rag-pipeline retrieve "What does ACER say about cross-zonal capacity in Southeast Europe?"
```

Run validation:

```powershell
rag-pipeline validate
```

## Validation Layer

The validation command checks:

- **Grounding:** known in-corpus questions should answer and cite the expected PDF.
- **Refusal:** an out-of-corpus question should return exactly `Not found in the provided documents.`
- **Retrieval quality:** expected source PDFs should appear in the top-k retrieved chunks.
- **Citation integrity:** any cited PDF filename must come from the retrieved chunk set.

Latest local validation run:

- `grounding_acer_2026_market_developments`: passed. Retrieved `acer-gas-electricity-key-developments-2026.pdf`, answered with electricity/gas market context, and cited only retrieved PDFs.
- `grounding_acer_see_cross_zonal_capacity`: passed. Retrieved `acer-see-cross-zonal-capacity-flexibility-2026.pdf` and answered with a source filename citation.
- `refusal_absent_cobblestone_revenue`: passed. The model returned exactly `Not found in the provided documents.`
- `retrieval_quality_entsoe_winter_outlook`: passed because the expected Winter Outlook PDF appeared in the top-k chunks. The top result was the Summer Outlook, which is an honest retrieval nuance: that report also discusses preparation for winter 2025-2026, so the query is semantically ambiguous.

Detailed local validation output is written to `validation/results.json`, which is ignored by Git.

## Cobblestone 150-Word Answer Draft

Problem: answering questions from European power and gas market reports by hand is slow and error-prone. I built a RAG pipeline over five public ENTSO-E and ACER PDFs covering seasonal adequacy, cross-zonal capacity, and 2026 EU electricity/gas market developments. The pipeline chunks each PDF, embeds the chunks locally with MiniLM, stores them in ChromaDB, retrieves the top-k relevant passages for a query, and asks the OpenAI model to answer only from those passages while citing source filenames. To validate against hallucination, I added a refusal test using an absent Cobblestone revenue question; the model returned exactly `Not found in the provided documents.` I also return retrieved chunks with each answer, check that cited filenames were actually retrieved, and spot-check retrieval quality against known report-specific questions.

## Example CV Line

Built a retrieval-augmented generation pipeline over public European energy-market PDFs from ENTSO-E and ACER using local MiniLM embeddings, ChromaDB, and source-cited OpenAI generation, with retrieval-quality checks and a refusal validation layer that returns "not found" when an answer is absent.
