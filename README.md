# RAG Pipeline - Grounded QA

This project is a small, defensible retrieval-augmented generation pipeline over public European energy-market PDFs. It retrieves relevant document chunks, asks an LLM to answer only from those chunks, cites source filenames, and refuses with `Not found in the provided documents.` when the answer is absent.

![Pipeline diagram: ingestion from a pinned PDF manifest through checksum verification, chunking, local embedding and a Chroma vector store; the query path from question through top-k retrieval to either a grounded answer or an exact refusal; and the four validation gates](docs/assets/rag_pipeline_architecture.svg)

The diagram marks where each control sits: checksum verification on download and ingestion, the grounding instruction ahead of generation, and the four gates that `rag-pipeline validate` runs against the result.

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
- **Generation:** OpenAI Responses API. The default model is `gpt-5.6-luna`, with reasoning effort explicitly set to `low` for this short grounded-answer task. Override the model with `OPENAI_MODEL`. The request omits `temperature` because GPT-5.6 models do not support that parameter; repeatability is measured through the validation suite instead.
- **Anti-hallucination:** the prompt says to answer only from retrieved context, cite source filenames, and return exactly `Not found in the provided documents.` when the context does not contain the answer.
- **Validation:** the CLI includes grounding, refusal, retrieval-quality, and citation-filename checks. SHA-256 verification of pinned source documents runs separately during download and ingestion.
- **Optional claim-support review:** the OpenAI Decisions API (`gpt-6-luna`) assesses answer units against their cited retrieved passages. It adds a review signal alongside the existing filename check; generation continues to use Responses.

### Why Low Reasoning Effort?

Retrieval has already narrowed the evidence before generation begins, so the model's job is to produce a short grounded answer with source citations rather than perform open-ended research or multi-step tool use. Explicit `low` effort avoids GPT-5.6's default `medium` reasoning overhead for this latency-sensitive step. It is not a determinism or correctness guarantee: the grounding prompt, refusal behavior, citation-filename check, and validation cases remain the controls that must be measured. The setting should be raised to `medium` only if representative validation questions show a material gain in answer completeness or grounding. This follows [OpenAI's guidance](https://developers.openai.com/api/docs/guides/latest-model) to choose reasoning effort from workload evidence rather than assuming higher is always better.

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

Assess the generated answer's claims against their cited evidence:

```powershell
rag-pipeline ask "What does ACER say about cross-zonal capacity in Southeast Europe?" --check-support --json
rag-pipeline validate --check-support
```

Use `RAG_CHECK_SUPPORT=true` in `.env.local` to enable the review step by default, or `--no-check-support` to disable it for a particular run. The feature is off by default while its thresholds and usefulness are being evaluated.

## Validation Layer

The validation command checks:

- **Grounding:** known in-corpus questions should answer and cite the expected PDF.
- **Refusal:** a plausible but absent energy-market question should return exactly `Not found in the provided documents.`
- **Retrieval quality:** expected source PDFs should appear in the top-k retrieved chunks.
- **Citation filename check:** any cited PDF filename must come from the retrieved chunk set.

Corpus integrity is enforced separately from `rag-pipeline validate`: both download and ingestion verify each PDF against the pinned SHA-256 checksum in `data/sources.json`.

Validation cases:

- `grounding_acer_2026_market_developments`: retrieves `acer-gas-electricity-key-developments-2026.pdf` and checks for specific market-monitoring details such as LNG, Russian gas imports, and network-code work.
- `grounding_acer_see_cross_zonal_capacity`: retrieves `acer-see-cross-zonal-capacity-flexibility-2026.pdf` and checks for specific detail on price spikes, Greece-Italy HVDC capacity, and cross-zonal capacity.
- `refusal_plausible_absent_poland_peak_demand`: asks an on-topic but absent question about Poland's projected Winter 2025-2026 peak electricity demand in GW and requires the exact refusal string.
- `retrieval_quality_entsoe_winter_outlook`: checks that the expected Winter Outlook PDF appears in the top-k chunks. A related Summer Outlook chunk can rank highly because that report also discusses preparation for winter 2025-2026, which is a useful retrieval-quality nuance to know.

Detailed local validation output is written to `validation/results.json`, which is ignored by Git.

## Decisions Claim-Support Review

A valid filename citation does not establish that the retrieved text supports an answer. For example, a passage about prices reaching 1000 EUR/MWh in summer 2024 does not support a claim about 2000 EUR/MWh in summer 2025. The optional Decisions check targets this gap.

The checker groups text into conservative spans ending at a citation. Several sentences or bullets can share that citation; each span is assessed in full, including any additional clauses. It submits the span and the retrieved passages from its cited filenames to Decisions as a `predicate` question. The instructions require support for numbers, units, dates, regions, and causal claims, and distinguish proposals or forecasts from realised outcomes. Evidence from uncited documents is excluded from that span's record.

The JSON result includes each unit's text, cited sources, evidence chunk IDs and pages, support probability, review status, request latency, and token usage. The default thresholds are provisional:

| Probability | Status | Review required |
| --- | --- | --- |
| At least 0.8 | `supported` | No |
| At most 0.2 | `unsupported` | Yes |
| Between the thresholds | `uncertain` | Yes |

Configure these with `RAG_SUPPORT_THRESHOLD` and `RAG_UNSUPPORTED_THRESHOLD`; use `OPENAI_DECISIONS_MODEL` to override the Decisions model independently of generation. Python SDK 3.26.0 or later is required. The request format and threshold guidance follow the [official Decisions documentation](https://developers.openai.com/api/docs/guides/decisions).

Missing or unknown citations, empty passages, provider refusals, malformed responses, and API failures require review. An API failure is reported as `unavailable`, with no support score. Requests have a 20-second timeout and no automatic retries. Exact pipeline refusals skip Decisions. Up to 32 answer units can be submitted; any additional units are retained as `not_checked` and require review.

This step preserves the generated answer and the existing citation result. It does not repair claims or establish factual truth beyond the supplied evidence. Citation grouping is deliberately simple: a span with several assertions receives one score, and an ambiguous citation scope can need manual review. Trailing uncited prose requires review. The returned probabilities are model estimates, not calibrated correctness guarantees. With `validate --check-support`, a non-refusal answer passes the added check only if every span is marked supported.

### Fixed Energy Examples

```powershell
rag-pipeline evaluate-support
```

This command calls Decisions on fixed examples in `data/claim_support_cases.json`; it does not run retrieval or regenerate answers. The examples use excerpts from pages 9, 18, and 26 of the checksum-verified ACER Southeast Europe report. They cover supported statements, incorrect prices, dates, units and regions, unsupported additional clauses, proposals presented as completed outcomes, and missing citations. The same answers and evidence are used for the filename-only baseline and the augmented review.

The report at `validation/claim-support-results.json` records false acceptances, supported claims flagged for review, API availability, latency, and token usage. It exits unsuccessfully if an API check is unavailable or the augmented check disagrees with a label. Labels are provisional author judgements, not independent labels or a held-out benchmark. Do not tune thresholds on these cases and report the same cases as an independent evaluation. Before using results as application evidence, have a reviewer label additional energy examples without seeing the API scores, choose thresholds on a separate development set, and measure errors and review workload on held-out cases.

The [7 October 2026 diagnostic](docs/decisions-evaluation.md) records the initial comparison and the remaining failures in the existing full-pipeline validation suite.
