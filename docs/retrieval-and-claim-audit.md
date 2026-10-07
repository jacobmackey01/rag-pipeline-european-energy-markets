# Contextual retrieval and source audit — 7 October 2026

The updated retrieval returns four Winter passages for each of the four supplied questions, with document filtering disabled. Contextual headers alone did not reliably solve the failure; hybrid keyword and vector ranking was needed. These are regression results on known cases, not a held-out retrieval benchmark. The earlier 71-test, seven-case result replaced two broad questions with specific ones and therefore obscured remaining recall gaps. The broad questions are now restored alongside the specific cases.

| Query form | Passage-only cosine | Contextual cosine | Contextual hybrid |
| --- | ---: | ---: | ---: |
| Original dated title | 0/4 | 2/4 | 4/4 |
| Outlook for 2025-26 | 1/4 | 1/4 | 4/4 |
| Winter 2025/2026 Outlook | 3/4 | 1/4 | 4/4 |
| Latest winter outlook | 2/4 | 2/4 | 4/4 |

The question texts, source PDFs, embedding model, 220-token passage setting, and top-k of four were held fixed for this comparison. All stages used the same 392 passage IDs. Titles were added only to embedding inputs; raw passages and page provenance remain unchanged. The longest contextual embedding input was 243 tokens, below the model’s 256-token limit. The index carries an embedding-format marker and requires rebuilding old passage-only vectors.

Hybrid ranking uses BM25 title and passage scores, with title scores weighted twice as much as passage scores. Normalized lexical and semantic scores each receive half the final weight. These are fixed implementation choices, not independently calibrated weights. The implementation ranks all eligible records locally, so the evidence here applies to the current small corpus. `--source` remains available for explicit selection. “Latest” means retrieval from available documents, not proof of the latest publication outside the corpus.

Queries asking for a missing edition now yield no context and the exact pipeline refusal, without a stack trace or an API request. This includes the supplied Winter Outlook 2024-2025 comparison and the `outlook for` and reordered-date forms. Invalid `--source` filenames are reported as argument errors.

## Audit of the uncertain grid-cost claim

The frozen generated claim was: “ACER also reports that new grid costs are projected to rise by 66% between 2022 and 2050.” Decisions scored it 0.72, sending it to review. Inspection of the original ACER PDF on page 48 shows a stacked household-grid-cost chart: the 66% annotation applies to the combined total, comprising legacy and new grid costs. The claim assigns the change to the new-cost component alone. I judge that claim unsupported as written; a supported formulation would attribute the projection to total household grid costs, following the chart annotation.

The retrieved plain-text excerpt contained chart values and both legend labels but had lost their spatial relationships and the full chart heading at the chunk boundary. This is a specific chart-extraction limitation. Adding document context to embeddings does not reconstruct chart semantics. The audit occurred after seeing the score and is not an independent or blinded label. It illustrates a useful review flag on one selected example; it does not establish a false-flag rate or general judge accuracy.

Source: [ACER Key Developments in European Electricity and Gas Markets 2026](https://www.acer.europa.eu/sites/default/files/documents/Publications/2026-ACER-Gas-Electricity-Key-Developments.pdf), page 48. The PDF was verified against the pinned manifest checksum before inspection.

## Why the Greece/Italy check failed

The frozen retrieved excerpts did not contain the required “Greece and Italy” wording or the complete HVDC discussion. The relevant passage exists on page 18 of the Southeast Europe report; the earlier broad question did not reliably retrieve it. The generator was not shown the fact that the test demanded. The broad question also did not explicitly request that particular link, making the exact-phrase expectation a poor completeness criterion.

The previous update substituted a specific HVDC question and a specific monitoring question for the original broad cases. Putting expected facts in a keyword query makes them easier to retrieve and is not evidence that broad-query recall improved. Both original questions and their phrase expectations are restored as visible known failures. The specific questions remain as additional cases. Broad exact-phrase expectations are brittle, but rewriting the questions was not an adequate way to resolve that limitation.

## Recall diagnostic with unchanged ranking

The same contextual hybrid index was queried without document filtering. No weights or query rewrites were changed for this diagnostic.

| Question | Required passage rank | Present at k=4 | k=6 | k=8 | k=12 |
| --- | ---: | --- | --- | --- | --- |
| What does ACER say about key developments in European electricity and gas markets in 2026? | 5 (network codes) | No | Yes | Yes | Yes |
| What does ACER say about cross-zonal capacity and flexibility in Southeast Europe? | 5 (page-18 HVDC) | No | Yes | Yes | Yes |
| Why did Southeast European prices spike in summer 2024? | 10 (page-18 HVDC) | No | No | No | Yes |

Increasing top-k to six or eight includes the missing passages for both broad questions, but does not recover the HVDC detail for the price-spike question. A retrieval-only top-12 probe records that distinction. Passing this probe establishes passage presence, not that a generated answer will include or correctly interpret the detail. Production defaults remain four chunks; the recall gap is still open. A general reranker has not been installed or evaluated in this round.

Known failures only cover content and phrase expectations. The JSON retains the actual failed checks and `passed: false`, with `status: XFAIL` and a reason. API support failures and citation failures cannot be masked by a recall marker. Unexpected passes require review of the marker. Model-generated answers and scores can vary between runs.

The restored suite has 82 passing offline tests. A live run with `validate --no-check-support` reports eight `PASS` and three `XFAIL`, with no `FAIL` or `XPASS`. The known failures are the two original broad questions and the default-depth price-spike recall case. The specific grounding questions, exact Poland refusal, four unfiltered Winter cases and top-12 recall probe pass. This run deliberately did not reassess Decisions; it verifies retrieval, generated phrase coverage, citation integrity and refusal behavior. It is not an all-green retrieval result.

The PR remains a draft. A fresh [50-unit blinded batch](blinded-label-protocol.md) has been prepared with blank human labels and no model scores computed. Jacob must label it before scoring, threshold selection or comparison with local NLI. The 11 easy diagnostic examples remain provisional author-labelled cases. Chart extraction is deferred.
