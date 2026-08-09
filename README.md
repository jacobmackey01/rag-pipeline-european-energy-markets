# RAG Pipeline - Grounded QA

This project is a small, defensible retrieval-augmented generation pipeline over public European energy-market PDFs. It retrieves relevant document chunks, asks an LLM to answer only from those chunks, cites source filenames, and refuses with `Not found in the provided documents.` when the answer is absent.

## Corpus

Option A is used, but narrowed to the Cobblestone Energy story: public European power and gas market documents from ENTSO-E and ACER.

- ENTSO-E Summer Outlook 2026
- ENTSO-E Winter Outlook 2025-2026
- ACER Key Developments in European Electricity and Gas Markets 2026
- ACER Increasing Cross-Zonal Capacity and System Flexibility in Southeast Europe 2026
- ACER Key Developments in European Gas Wholesale Markets Winter 2025-2026

The PDFs are downloaded from the source URLs in `data/sources.json`. Each source includes a SHA256 checksum, and `rag-pipeline download` verifies the local file after download or cache reuse. PDFs are not committed to Git because they are public binary artifacts.

## Design Choices

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally. Embeddings map semantically similar text close together, so a query can match related wording even without exact keyword overlap.
- **Chunking:** default 220 MiniLM tokenizer tokens with 40-token overlap. Chunks are built from the tokenizer's offset mappings, so they preserve the original PDF text instead of rebuilding it; figures such as `5.25%` and `2025-2026` are not mangled. Chunking runs over the whole PDF text, not page-by-page, and metadata records the page range each chunk spans.
- **Vector store:** ChromaDB with cosine distance and persistent local storage in `data/chroma`.
- **Generation:** OpenAI Responses API. The default model is `gpt-5.6-luna`, with reasoning effort explicitly set to `low` for this short grounded-answer task. Override the model with `OPENAI_MODEL`. The request omits `temperature` because GPT-5.6 models do not support that parameter. Fixed validation cases check behavior for regressions; they do not make generation deterministic.
- **Grounding controls:** the prompt says to answer only from retrieved context, cite source filenames, and return exactly `Not found in the provided documents.` when the context does not contain the answer.
- **Validation:** the CLI runs corpus-checksum, retrieval, refusal, expected-phrase, and citation-source smoke checks.

### Why Low Reasoning Effort?

Retrieval has already narrowed the evidence before generation begins, so the model's job is to produce a short answer with source citations rather than perform open-ended research or multi-step tool use. Explicit `low` effort avoids GPT-5.6's default `medium` reasoning overhead for this latency-sensitive step. It is not a determinism, correctness, or claim-level entailment guarantee: the prompt, refusal behavior, citation-source check, and fixed validation cases are regression controls. The setting should be raised to `medium` only if representative validation questions show a material gain in answer completeness and source use. This follows [OpenAI's guidance](https://developers.openai.com/api/docs/guides/latest-model) to choose reasoning effort from workload evidence rather than assuming higher is always better.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Create `.env.local` with:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-luna
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

## Validation Smoke Suite

The validation command checks:

- **Answer behavior:** known in-corpus questions should contain the expected phrases and cite a retrieved PDF.
- **Refusal:** a plausible but absent energy-market question should return exactly `Not found in the provided documents.`
- **Retrieval quality:** expected source PDFs should appear in the top-k retrieved chunks.
- **Citation-source integrity:** any cited PDF filename must come from the retrieved chunk set.
- **Corpus integrity:** downloaded PDFs must match the SHA256 checksums in `data/sources.json`.

Corpus integrity runs first. If a manifest PDF is missing, unpinned, or has the wrong checksum, validation stops before retrieval or paid generation. The citation-source check is deliberately narrower than claim verification: it catches invented or unretrieved filenames, but it does not prove that every generated sentence is entailed by the cited PDF. The expected-phrase cases are explainable regression smoke tests, not a comprehensive RAG benchmark.

Validation cases:

- `grounding_acer_2026_market_developments`: retrieves `acer-gas-electricity-key-developments-2026.pdf` and checks for specific market-monitoring details such as LNG, Russian gas imports, and network-code work.
- `grounding_acer_see_cross_zonal_capacity`: retrieves `acer-see-cross-zonal-capacity-flexibility-2026.pdf` and checks for specific detail on price spikes, Greece-Italy HVDC capacity, and cross-zonal capacity.
- `refusal_plausible_absent_poland_peak_demand`: asks an on-topic but absent question about Poland's projected Winter 2025-2026 peak electricity demand in GW and requires the exact refusal string.
- `retrieval_quality_entsoe_winter_outlook`: checks that the expected Winter Outlook PDF appears in the top-k chunks. A related Summer Outlook chunk can rank highly because that report also discusses preparation for winter 2025-2026, which is a useful retrieval-quality nuance to know.

Detailed local validation output is written to `validation/results.json`, which is ignored by Git.

Unit tests run in GitHub Actions for pushes and pull requests on Python 3.11 and 3.12. The end-to-end validation command remains local because it requires the downloaded corpus, a built Chroma index, an OpenAI API key, and paid model calls.
