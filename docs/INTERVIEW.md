# Talking about this project

Material for a recruiter screen and an AI Engineer interview. Every number here
is transcribed from `results/ablations.md` or `results/threshold.md` — nothing
is asserted that isn't in a generated file.

---

## 30-second verbal pitch

> I built an evaluation harness for a RAG system over football-analytics
> documentation. The point wasn't to build a chatbot — it was to find out
> whether swapping the retriever actually made answers better, or whether it
> just moved a number. So I hand-labeled 30 questions, tagging which documents
> contain each answer, and measured a ladder of retrieval strategies against
> them. Hybrid retrieval with a cross-encoder reranker took all@5 from 0.75 to
> 0.92 over 24 answerable questions — 22 of 24. Twice I caught my own eval
> being wrong
> rather than the system being right — I'd been folding unanswerable questions
> into the denominator, and then reading the corpus document by document
> turned up three labels that pointed at documents which couldn't answer their
> question. Both corrections raised the score, and neither was a retrieval
> improvement. Splitting the slices also exposed a real bug: hybrid fusion
> loses a question that dense alone ranks 2nd. And the system still can't tell
> when it doesn't know — the relevance floor only rejects 1 of 6 unanswerable
> questions, which is why I published the whole trade-off curve instead of one
> tuned number. The writeup says so.

Read that aloud and time it. If it's over 40 seconds, cut everything after
"Both corrections raised the score" — the audit is the point, the abstention
curve is the bonus.

**The football corpus is not a gimmick.** It's why the labels were graded
carefully: I know the domain well enough to know a label is wrong without
asking anyone. If asked "why football?" — "so I'd catch a bad label myself
instead of trusting my own annotation blindly."

---

## Resume bullets

Three lines. Numbers first.

- Built an eval harness for a RAG pipeline over a 14-page football-analytics
  corpus: hand-labeled 30 questions (24 answerable, 6 unanswerable) with
  document-level multi-hop labels, scored by hit@5 / all@5 / MRR, and swept
  chunk size, hybrid RRF fusion, and cross-encoder reranking against them.
- Raised all@5 from 0.750 to 0.917 (18% to 22% of 24) via hybrid retrieval,
  and MRR from 0.889 to 0.910 via cross-encoder reranking; then audited all 30
  labels against the corpus and found 3 that pointed at documents which can't
  answer their question — published the audit, and stated that hit@5 1.000
  means the metric is saturated, not that the system is perfect.
- Built a Streamlit trace viewer exposing per-query retrieved chunks, scores,
  and gold-label agreement, so every eval number is inspectable per query
  instead of taken on faith.

**Do not put the serving or load-test work on the resume.** The harness is
built but no valid latency table exists — the dev proxy in this environment
strips `cache_control` and rate-limits concurrency past 1. It comes up in
conversation, where the blocker is a good answer. A resume bullet whose number
you cannot produce on request is a liability.

---

## Interview deep-dive

**"Why fuse on rank instead of adding the scores?"**

BM25 output is unbounded — it grows with term frequency and document length.
Cosine similarity lives in [-1, 1]. Adding them is meaningless; the dense score
would be a rounding error next to BM25. So `rrf()` in `src/retrieve.py` ranks
each retriever independently and sums `1/(k + rank)`. Rank position is
comparable across methods; raw scores are not.

**"Your hit@5 is 1.000 on a 33-chunk corpus. That's saturated. Is the eval even
doing anything?"**

Fair challenge, and I lead with it in the README. Top-5 covers a sixth of the
corpus, so hit@5 can't move much — the honest reading is that the metric is
*saturated*, not that the system is perfect. The number I quote is **all@5
0.917, 22 of 24**, because the two remaining misses are the ones that decide
answer quality on multi-hop questions. all@5 moved 0.750 → 0.917 across the
ladder, and it moves for a legible reason: hybrid retrieval is what finds the
*second* required document, not just the first. A saturated metric can't
demonstrate improvement, which is the strongest argument I have for a bigger
corpus — so I scoped that rather than hiding it.

**"Hybrid made your score better, so why does the README list fusion as a bug?"**

Because I traced the two remaining misses through the stages and they lose the
same document the same way. For one of them the correct chunk ranks **2nd by
dense, 13th by BM25, 6th after RRF** — one place outside the cut. Dense gets
it right, BM25 drops it because it rewards the literal word "convert" and
misses the paraphrased intent, and RRF averages the two rankings into a loss.
The hybrid that lifts all@5 overall costs this question individually. That's a
real, characterizable failure, not a shrug — the obvious fix is weighted RRF
that trusts dense more on lexically-poor queries, and `rrf()` already takes the
constant.

**"How do you know the numbers are real and not overfitting to your own eval?"**

Four things, and the strongest one is the label audit. Labels attach to
documents, not chunks, so the same 30 items score every chunk size — the chunk
sweep isn't seven independent chances to get lucky. The relevance floor ships
as a full trade-off curve, not one tuned number, because tuning a threshold on
the set you report it on would make the number meaningless. And twice I found
an error in my own eval rather than in the system: the denominator once
included unanswerable questions no top-k retriever could ever win, and then
reading all 30 gold documents one by one turned up **three labels pointing at
documents that cannot answer their question** — one asked where to report a bug
and the words "bug" and "issue tracker" appear in *no document in the corpus*.
Both fixes raised the score. Neither was a retrieval improvement, and I'd
rather have found them than have you find them.

**"How do you know you didn't just relabel things until the numbers looked
better?"** — the sharpest version of the above, and worth having ready.

The audit method is the defense: for each item I read the gold document and
asked whether it actually contains the answer, then went looking for the
document that does. Nothing was changed because a score moved — the diff note
in `results/label_audit.md` quotes the `grep` that settles each case. One
candidate on the shortlist turned out to be correctly labeled and just
paraphrased, and I recorded that too, because an audit that only ever confirms
changes isn't an audit.

**"Why not fine-tune the embedder?"**

30 labels can't train one, and I'd rather spend the time on knowing whether the
current one works. Fine-tuning is a second experiment; being able to run the
first one is the prerequisite. The ablation harness is what makes a tuning
experiment a ten-minute job later rather than a rebuild.

**"Abstention is 1 out of 6. Is that a failed feature?"**

It's a measured limitation and the mechanism is visible. The cross-encoder
scores a topically-adjacent chunk highly for a confidently-worded unanswerable
question, so the two score distributions overlap — only 1 of 24 answerable
questions ranks below *every* unanswerable one under dense, and zero under the
reranker. No threshold separates classes that overlap. Pushing the floor higher
does buy rejection, but only by silencing questions the corpus can answer: the
table shows it declining 4 of 6 unanswerables at a floor that also silently
refuses to answer "how do I get started?" Getting to a clean 6/6 costs 42% of
answerable recall. So the selection rule is *maximize rejection subject to
silencing nothing already correct* — and with 6 items I can't fit or validate
a floor without overfitting, which is why the deliverable is the curve. The
fix is more unanswerable items, then a proper calibration set.

**"What would you do first with another week?"**

Grow the corpus, not the retriever. At 33 chunks the task has no headroom, and
every retrieval change I make is measured against a saturated metric. The eval
harness doesn't need changes — labels are doc-level, so a bigger corpus drops
straight into the existing tables.

**"You said the load test is written but unpublished."**

Yes. The environment's dev proxy silently strips `cache_control`, so the
prompt-cache A/B would have measured nothing, and the single shared credential
enters a ~100s cooldown after a handful of requests, so concurrency above 1
can't be sustained long enough to produce a real p99. I built the harness to
report p50/p95/p99, throughput, and cost including the cache write, and every
result row carries a `cache_engaged` flag so a provider that ignored
`cache_control` couldn't be misread as a caching win. But I didn't publish the
partial numbers, because the cached arm only looked cheaper *because the
directive was dropped*. With a direct Anthropic key it's one command.

---

## What not to say

- **"Hit@5 of 1.000."** Correct, and still the wrong thing to lead with —
  it reads as a solved eval, and the honest reading is a saturated metric on a
  corpus where top-5 is a sixth of everything. Quote **all@5 0.917, 22 of 24**.
- **"Improved retrieval by 22%."** 0.750 → 0.917 is a 0.17 absolute gain. Quote
  the absolute numbers; a percentage makes a small lift sound large.
- **Anything about load testing or serving.** No valid latency table exists.
- **"Production-grade."** It's a 14-page corpus and a Streamlit viewer.
