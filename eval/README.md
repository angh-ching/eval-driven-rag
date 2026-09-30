# `golden.jsonl` — the labeled set

30 questions, hand-labeled against the 14-page `socceraction` docs corpus.
This file is the only ground truth in the project. Every number in
`results/ablations.md` and `results/threshold.md` is scored against it, so
changing a single line here moves the headline results.

## Schema

One JSON object per line, 30 lines.

| Field | Type | Meaning |
|---|---|---|
| `id` | int | 1–30, stable. Keys results and traces — do not renumber. |
| `question` | str | What a user would type. |
| `answer_doc_ids` | list[str] | Corpus filenames that contain the answer. `[]` when `answerable` is false. |
| `hops` | int | 0 = unanswerable. 1 = one document answers it. 2 = the answer spans documents. |
| `answerable` | bool | Whether the corpus can answer it at all. |

```json
{"id": 1, "question": "How do I get started with the library for the first time?", "answer_doc_ids": ["intro"], "hops": 1, "answerable": true}
```

## The two slices

`answerable` splits the set, and the two halves are **never** folded into one
denominator:

- **24 answerable** — scored on `hit@5`, `all@5`, MRR. `all@5` requires
  *every* gold document in the top 5, so a 2-hop question counts as solved
  only when both hops land.
- **6 unanswerable** — scored on `abstain@5` alone. The corpus has no answer,
  so returning nothing is the correct behavior. Folding these into the
  answerable denominator rewards a retriever for failing.

That separation is not bookkeeping. Unanswerable id 3 was previously counted
answerable, which turned a labeling mistake into an apparent retrieval
failure.

## Labeling protocol

1. Write the question from a user's intent, not from a document's wording.
2. Label **every** document that answers it, not just the first. A multi-hop
   question gets a longer list; that is what `all@5` tests.
3. If no document in the corpus answers it, set `answerable: false` and
   `answer_doc_ids: []`. Do not stretch a weak match to keep the item.
4. Domain familiarity is the point of this corpus: a bad label should be
   catchable by reading, without asking a second annotator.

## The audit

The labels have been audited once — all 30 items re-read against the corpus
after a low question/gold word-overlap shortlist surfaced suspicious ones.
Three labels were wrong (one pointed at a document that could not answer its
question, two pointed at the wrong document) and one item moved from the
answerable to the unanswerable slice.

Full write-up, including the `grep` checks that settled each case:
[`../results/label_audit.md`](../results/label_audit.md).

Two lessons worth keeping if you edit this file:

- **Overlap is a triage signal, not a verdict.** A correctly-labeled item
  scored 0.14 because the question paraphrases the document. Only reading
  settles it.
- **A correction that raises your score is not automatically a
  validation.** Both corrections here raised the numbers while changing
  nothing about the system. Re-run `python3 src/metrics.py` after any edit —
  the tables regenerate from this file.
