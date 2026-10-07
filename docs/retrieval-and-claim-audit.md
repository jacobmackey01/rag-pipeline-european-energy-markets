# Contextual retrieval and source audit — 7 October 2026

The updated retrieval returns four Winter passages for each of the four supplied questions, with document filtering disabled. Contextual headers alone did not reliably solve the failure; hybrid keyword and vector ranking was needed. All 71 offline tests pass, and the latest seven-case live validation run passes. These are regression results on known cases, not a held-out retrieval benchmark.

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

The validation question now explicitly asks about the HVDC interconnection and summer 2024 price spikes. Retrieval must contain Greece, Italy, and HVDC before the answer is assessed; the answer must include the link and the original price-spike/cross-zonal-capacity detail. Country names are checked separately to accept either ordering. The monitoring case was likewise clarified to ask for the planned activities its existing assertions required, after hybrid retrieval correctly produced a general market summary for the old broad wording. Relevant-content checks were added to both cases; no case was deleted or support threshold lowered.

The latest live run passes both grounding cases, their context checks, citation checks, and Decisions review; it also preserves the exact absent-Poland-demand refusal and passes all four unfiltered Winter regressions. A separate live “latest winter outlook” answer retrieves four Winter passages and passes citation integrity and support review. Model-generated answers and scores can vary between runs.

The PR remains a draft. Independent labels on real generated claims, held-out threshold selection, and comparison with a local NLI model remain outstanding. The 11 easy diagnostic examples are still described as provisional author-labelled cases.
