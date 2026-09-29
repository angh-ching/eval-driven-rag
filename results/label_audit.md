# Label audit — 2026-09-30

Method: for each answerable item, read the gold document and ask whether it
actually contains the answer. A low question/gold word-overlap shortlist picked
candidates (ids 3, 4, 11, 16, 19); each of the five was then read by hand.
Two of the five turned out to be correct labels that merely paraphrase the
question. **All 30 items were re-checked against the corpus; the audit changed
four labels.**

The shortlist is only a triage heuristic — overlap 0.14 on a correctly-labeled
item just means the question paraphrases. Only reading settles it.

## Changed

### id 3 — gold document cannot answer the question

    "Why is my data not loading, and where should I report a bug?"  -> ["faq"]

`corpus/faq.txt` has exactly three entries: what is socceraction, where to get
event data, what license it uses. No troubleshooting, no bug reporting.

```
$ grep -rli "bug\|issue tracker\|github issue" corpus/
(no matches)
```

Those words appear in **no document in the corpus**. The item is not
answerable, so it belonged in the unanswerable slice. Moved there: 25 → 24
answerable, 5 → 6 unanswerable.

This was previously reported as a retrieval failure. It never was one.

### id 11 — wrong document

    "How do I break a match into periods or phases?"  -> ["spadl-atomic_spadl"]

`corpus/spadl-atomic_spadl.txt` is about the *atomic* variant of SPADL. It
defines a `period_id` attribute but never explains how a match is divided.

The actual answer is in `corpus/valuing_actions-xT.txt`:

> The expected threat or xT model is a possession-based model. That is, it
> divides matches into possessions, which are periods of the game where the
> same team has control of the ball.

Corrected to `["valuing_actions-xT"]`.

### id 19 — one gold document is inert

    "How do I turn raw provider data into the unified action representation?"
    -> ["data-statsbomb", "spadl-spadl"]

`corpus/spadl-spadl.txt` is correct: it documents `convert_to_actions()`.

`corpus/data-statsbomb.txt` never mentions SPADL or conversion:

```
$ grep -in "spadl\|convert" corpus/data-statsbomb.txt
(no matches)
```

The document describing provider → unified representation is
`corpus/data-index.txt`: *"a unified data model… The schema defines the minimal
set of columns"* and *"the main differences between kloppy and"*. That doc
also mentions kloppy, which is why a question phrased around "regardless of
provider" is the right way to reach it.

Corrected to `["data-index", "spadl-spadl"]`.

### id 24 — verified correct, recorded because it looks wrong

    "How do I convert actions into a pandas dataframe?"  -> ["intro", "spadl-spadl"]

Checked because it was a known miss. Both gold docs genuinely answer it —
`corpus/intro.txt` has a "Pandas DataFrames" section, and
`corpus/spadl-spadl.txt` line 159: *"The obtained dataframe represents the body
part, result, action type, players…"*. The label is right; the retrieval is
what fails (the fusion regression noted in the README).

## Unchanged after reading

**id 4** — *"common way to pull in event data regardless of provider"* →
`data-index`. Correct: *"Socceraction provides API clients for various popular
event stream data sources… using a unified data model. Alternatively, you can
also use kloppy"*. The low overlap came from the word "regardless".

**id 16** — *"how many previous actions does the model look at"* →
`valuing_actions-vaep`. Correct: the gamestate is \(S_i = \{a_{i-2}, a_{i-1},
a_i\}\), i.e. two previous actions. The question asks for a count; the answer
is a formula, which is why the words barely match.

## Consequence

The unanswerable slice becomes 6 items, the answerable slice 24. Both headline
numbers move. Re-run `python3 src/metrics.py` — the ablation and threshold
tables regenerate from this file.
