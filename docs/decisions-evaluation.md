# Decisions claim-support diagnostic — 7 October 2026

The optional support checker flagged all seven unsupported examples that passed the existing filename check, and accepted all three supported examples in this run. These are provisional author-labelled cases based on excerpts from a checksum-verified ACER report, not an independent benchmark. The result supports keeping an optional review experiment; it does not establish production accuracy or a reduction in analyst workload.

| Outcome on the same 11 answers | Filename check | Filename plus support review |
| --- | ---: | ---: |
| Supported examples accepted | 3 | 3 |
| Unsupported examples accepted | 7 | 0 |
| Supported examples flagged | 0 | 0 |
| Unsupported examples flagged | 1 | 8 |

The uncited example was already flagged by the baseline. The seven additional flags included a partially supported compound statement sent to review as uncertain. The supported and unsupported thresholds were fixed at 0.8 and 0.2 before the run. Ten Decisions requests completed without API failures, used 4,193 input tokens, and took 2.8511 seconds in aggregate as measured by the checker. This is added checking time on fixed answers, not end-to-end RAG latency or a comparison with Responses.

Validation used OpenAI Python SDK 3.26.0, Decisions model `gpt-6-luna`, the existing Responses generation model `gpt-5.6-luna`, and the default local MiniLM embedding model. All 38 offline tests passed. The five pinned source PDFs passed checksum verification and produced 392 indexed chunks.

The final live `rag-pipeline validate --check-support` run completed retrieval, Responses generation, and Decisions review. Both generated grounding answers passed claim support and citation integrity, and the absent Poland demand question returned the exact refusal without a Decisions request. The full suite remains unsuccessful: the Southeast Europe answer omitted the existing expected phrase `Greece and Italy`, and the Winter Outlook retrieval case returned Summer Outlook chunks in all four positions. Those checks concern existing generation/retrieval behavior and remain visible; they were not weakened to obtain a pass.

The cases and label reasons are in `data/claim_support_cases.json`. Reproduce the diagnostic with `rag-pipeline evaluate-support`. Detailed live outputs are written under the ignored `validation/` directory. Before citing the result in an application, independently label additional energy examples, select thresholds on a separate development set, and measure held-out errors, review workload, latency, and cost.
