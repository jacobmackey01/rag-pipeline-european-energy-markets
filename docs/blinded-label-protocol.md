# Blinded claim labels

Label real generated answers before running any support judge on them. The initial 11 author-labelled edits remain diagnostics; they do not measure accuracy or the false-flag rate on real answers.

The 7 October batch contains 50 unique citation-delimited units selected from 25 fresh pipeline answers to energy questions. The selected rows represent 24 answers. Generation used `gpt-5.6-luna`, the contextual hybrid index and the default four retrieved passages. Decisions was disabled throughout capture. No Decisions or NLI scores have been computed for this batch, and no human labels have been filled by Codex.

The export uses seed `20261007`. Selection and the development/test split occur before labels: 27 units are development, 23 are test. All units from one question stay in one split. Both splits draw from the same small corpus, so this is an answer-group split, not a document holdout. The questions were written for this batch and the sample is small; the result will not establish performance on all energy questions.

## Filling the CSV

Fill only `human_label` and, optionally, `review_notes`. Keep unit IDs, split assignments, questions, claims and evidence unchanged. Judge the entire unit against its supplied evidence, including every number, qualifier, date, region and causal assertion. A unit can contain several sentences sharing one citation; it is not necessarily one atomic claim.

| Label | Meaning |
| --- | --- |
| `supported` | The supplied evidence supports every factual assertion in the unit. |
| `unsupported` | At least one assertion contradicts the evidence or is not established by it. Missing evidence does not establish a claim. |
| `unclear` | The claim or evidence is ambiguous enough that you cannot make a defensible judgement. Explain the ambiguity in notes. |

Use the exact retrieved text shown in `evidence`, which includes filename, physical PDF page range, chunk ID and source URL. Evidence is restricted to retrieved passages from the unit's cited filenames, matching the support check's evidence scope. Do not supplement it with outside knowledge or another PDF page: this measures support from the supplied passages. If chart layout or extraction prevents judgement, use `unclear` and explain why. Empty evidence and uncited units are retained rather than removed to make the sample look better.

Save and return the labelled CSV before viewing any model scores. Freeze its bytes and hash before scoring. Codex should validate the allowed labels and unchanged source fields, without filling or revising your judgements. Keep the labels and capture files private; `validation/labels/` is ignored by Git.

## After labels are frozen

Only then run the filename baseline, a local NLI model and Decisions on the identical claim/evidence units. Select thresholds using development labels alone, freeze them, and report results separately on the test units. Do not tune thresholds against the test labels. Record unsupported units accepted, supported units flagged, review workload, availability, latency and cost. Report `unclear` labels separately rather than silently dropping them or counting them as successes. Include the NLI model and evidence-aggregation rule so the comparison can be reproduced.

The batch is too small for precise error-rate claims. Counts, denominators and uncertainty belong alongside any percentages. Chart extraction and changing the production retrieval depth are outside this labelling step.

## Reproducing an export

Save a JSON list of pipeline results with `question`, `answer` and `retrieved_chunks`. Each chunk needs `id`, `source`, `source_url`, `page_start`, `page_end` and `text`. Use fresh answers with support checking disabled for a blinded capture. The exporter does not call an API or load credentials; it always leaves labels blank and copies only the documented fields, even if the input snapshot contains scores.

```sh
rag-pipeline export-labels \
  --answers validation/labels/unscored-answers.json \
  --output validation/labels/energy-claims-blind.csv \
  --count 50 --seed 20261007
```

It retains exact evidence text with CSV quoting for commas and embedded newlines. Leading formula characters are escaped for spreadsheet safety. Re-exporting after labelling would reset the blank label columns; use a separate path.
