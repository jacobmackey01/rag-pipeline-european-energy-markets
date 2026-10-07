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

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally. Every passage is embedded with a header containing its manifest document title, so report identity accompanies the passage's meaning. Stored evidence text and PDF page ranges are preserved.
- **Chunking:** default 220 MiniLM tokenizer tokens with 40-token overlap. Chunks are built from the tokenizer's offset mappings, so they preserve the original PDF text instead of rebuilding it; figures such as `5.25%` and `2025-2026` are not mangled. Chunking runs over the whole PDF text, not page-by-page, and metadata records the page range each chunk spans.
- **Retrieval:** ChromaDB stores contextual embeddings and evidence in `data/chroma`. Hybrid ranking combines cosine similarity with BM25 over document titles and passage text; titles receive twice the passage weight in the lexical score. Normalized lexical and semantic scores receive equal final weights. Ranking currently reads all eligible records locally, which suits this small corpus.
- **Document scope:** queries that name a dated seasonal report, such as `Winter Outlook 2025-2026`, search only matching manifest PDFs before ranking passages. General questions retain cross-document search. Use `--source` to select a manifest filename explicitly.
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

Rebuild an existing passage-only index with this command before querying the updated pipeline. The index records its embedding format and rejects stale passage-only vectors. Chunking reserves space for the title header within MiniLM's 256-token limit; the verified corpus remains at 392 passages, with a maximum contextual length of 243 tokens. See [Sentence Transformers' sequence-length guidance](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html).

Ask a question:

```powershell
rag-pipeline ask "What does ACER say about electricity and gas market developments in 2026?" --show-chunks
```

Inspect retrieval without generation:

```powershell
rag-pipeline retrieve "What does ACER say about cross-zonal capacity in Southeast Europe?"
```

Search a specific report when the question does not name its dated title:

```powershell
rag-pipeline ask "What are the adequacy risks?" --source entsoe-winter-outlook-2025-2026.pdf --show-chunks
```

`ask` and `retrieve` accept repeated `--source` options for comparisons. Explicit filenames override automatic title matching. Dated seasonal titles accept hyphens, en/em dashes, nonbreaking hyphens, slash-separated years, `outlook for YYYY-YY`, and `Winter YYYY/YYYY Outlook`. A question requesting an absent edition returns the normal refusal, including comparisons that require an unavailable edition. An invalid `--source` filename gets a CLI argument error; a matching report with no indexed chunks returns no context. The explicit filter uses Chroma's [metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering). Questions such as “latest winter outlook” use hybrid ranking across the available corpus; they do not establish which report is globally newest.

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

- `grounding_acer_2026_market_developments` and `grounding_acer_see_cross_zonal_capacity`: retain the original broad questions and phrase expectations. They are known failures at top-k four: the required passages rank fifth. The specific monitoring and HVDC questions remain as additional cases, not replacements.
- `retrieval_see_price_spike_top4`: checks whether the unprompted summer-price-spike question retrieves the Greece-Italy HVDC evidence. It is a known recall failure; the relevant passage ranks tenth. A separate `top12_probe` case checks a larger context without changing the default retrieval depth.
- `refusal_plausible_absent_poland_peak_demand`: asks an on-topic but absent question about Poland's projected Winter 2025-2026 peak electricity demand in GW and requires the exact refusal string.
- `retrieval_quality_entsoe_winter_outlook` and `retrieval_winter_paraphrase_1`–`3`: test the original query and three rewordings, with automatic document filtering disabled. Each requires nonempty retrieval with every returned passage from the Winter report. These exercise contextual hybrid ranking directly.

Validation prints `PASS`, `XFAIL`, `XPASS` or `FAIL`, including each check and the known-gap reason. `XFAIL` only covers the explicitly named content/phrase checks; it preserves `passed: false` in JSON. Citation and support failures remain `FAIL`. Unexpected passes (`XPASS`) also exit unsuccessfully so the marker must be reviewed. Known cases and the larger-context probe record their explicit retrieval depth; `--top-k` applies to cases without an override. Detailed output is written to `validation/results.json`, which is ignored by Git.

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

The [contextual retrieval and source audit](docs/retrieval-and-claim-audit.md) records the unfiltered paraphrase comparison, restored recall gaps, and audit of a real uncertain claim. The [blinded labelling protocol](docs/blinded-label-protocol.md) describes the fresh 50-unit batch and the offline `export-labels` command. Human labels must be frozen before scoring or the NLI comparison. The [initial Decisions diagnostic](docs/decisions-evaluation.md) retains the earlier results and their limitations.
