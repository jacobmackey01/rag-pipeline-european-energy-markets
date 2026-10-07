# Decisions claim-support diagnostic — 7 October 2026

This document records earlier diagnostic snapshots. The subsequent [contextual retrieval and source audit](retrieval-and-claim-audit.md) reports the current implementation, paraphrase regressions, corrected validation questions, and chart review.

The optional support checker flagged all seven unsupported examples that passed the existing filename check, and accepted all three supported examples in this run. These are provisional author-labelled cases based on excerpts from a checksum-verified ACER report, not an independent benchmark. The result supports keeping an optional review experiment; it does not establish production accuracy or a reduction in analyst workload.

| Outcome on the same 11 answers | Filename check | Filename plus support review |
| --- | ---: | ---: |
| Supported examples accepted | 3 | 3 |
| Unsupported examples accepted | 7 | 0 |
| Supported examples flagged | 0 | 0 |
| Unsupported examples flagged | 1 | 8 |

The uncited example was already flagged by the baseline. The seven additional flags included a partially supported compound statement sent to review as uncertain. The supported and unsupported thresholds were fixed at 0.8 and 0.2 before the run. Ten Decisions requests completed without API failures, used 4,193 input tokens, and took 2.8511 seconds in aggregate as measured by the checker. This is added checking time on fixed answers, not end-to-end RAG latency or a comparison with Responses.

Validation used OpenAI Python SDK 3.26.0, Decisions model `gpt-6-luna`, the existing Responses generation model `gpt-5.6-luna`, and the default local MiniLM embedding model. All 38 offline tests passed. The five pinned source PDFs passed checksum verification and produced 392 indexed chunks.

The initial live `rag-pipeline validate --check-support` run completed retrieval, Responses generation, and Decisions review. Both generated grounding answers passed claim support and citation integrity, and the absent Poland demand question returned the exact refusal without a Decisions request. The full suite remains unsuccessful: the Southeast Europe answer omitted the existing expected phrase `Greece and Italy`, and the Winter Outlook retrieval case returned Summer Outlook chunks in all four positions. Those checks concern existing generation/retrieval behavior and remain visible; they were not weakened to obtain a pass.

The cases and label reasons are in `data/claim_support_cases.json`. Reproduce the diagnostic with `rag-pipeline evaluate-support`. Detailed live outputs are written under the ignored `validation/` directory. Before citing the result in an application, independently label additional energy examples, select thresholds on a separate development set, and measure held-out errors, review workload, latency, and cost.

## Retrieval follow-up — 7 October 2026

The wrong-report retrieval failure was reproduced without changing the question, embeddings, corpus, or top-k setting. The Winter report had 60 indexed chunks. Unrestricted similarity search returned four Summer report chunks, including passages discussing winter preparation. The issue was document selection before passage ranking, rather than a missing Winter document.

Queries naming a dated seasonal Outlook now resolve the edition against the source manifest and apply a Chroma source-metadata filter before similarity ranking. Broad questions still search across documents. `ask` and `retrieve` also accept explicit, repeatable `--source` filenames. Unknown editions and filenames are rejected; missing indexed content returns no passages rather than falling back to a different report. Comparisons naming both dated reports retain both as eligible sources.

For the unchanged Winter Outlook 2025-2026 adequacy question, all four returned chunks now belong to the Winter report, on PDF pages 21, 3, 12, and 31. A live Responses answer described the favourable overall adequacy outlook and conditional regional risks, citing that report. The checker initially split its `p. 3` annotation into uncited prose; preserving `p.` and `pp.` annotations fixed that formatting error. Rechecking the same frozen answer and passages with Decisions returned a support probability of 0.98 and no review flag. This is a diagnostic on a real answer, not an independent correctness label.

All 57 offline tests pass. A Chroma regression deliberately gives a Summer passage greater similarity than the Winter passages and verifies that named-report filtering still returns only Winter passages. Tests also cover Unicode/slash date forms, explicit filenames, comparisons, unavailable reports, and CLI propagation.

The latest full live validation passes the Winter retrieval case, including a new check requiring every returned chunk to come from the requested report. The exact Poland-demand refusal still passes. The full suite remains unsuccessful: the Southeast Europe answer again omitted `Greece and Italy`, and a newly generated ACER grid-cost claim was marked uncertain at 0.72. Its correctness has not been independently labelled; the threshold remains 0.8. The PR remains a draft, and neither the NLI comparison nor an independent false-flag evaluation has been completed.
