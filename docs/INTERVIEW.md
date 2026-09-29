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
> them. Hybrid retrieval with a cross-encoder reranker took all@5 from 0.72 to
> 0.88 over 25 answerable questions. I also caught that my own metric was wrong
> — I'd been folding five unanswerable questions into the denominator, where
> they were unwinnable, and it was understating the result. Splitting the two
> slices revealed a second problem: the system couldn't tell when it didn't
> know. Adding a relevance floor fixed the capability, and publishing the whole
> trade-off curve instead of one tuned number showed the fix only works on 1 of
> 5 cases. The writeup says so.

Read that aloud and time it. If it's over 40 seconds, cut the metric-correction
sentence — everything after it is a bonus, not the point.

**The football corpus is not a gimmick.** It's why the labels were graded
carefully: I know the domain well enough to know a label is wrong without
asking anyone. If asked "why football?" — "so I'd catch a bad label myself
instead of trusting my own annotation blindly."

---

## Resume bullets

Three lines. Numbers first.

- Built an eval harness for a RAG pipeline over a 14-page football-analytics
  corpus: hand-labeled 30 questions (25 answerable, 5 unanswerable) with
  document-level multi-hop labels, scored by hit@5 / all@5 / MRR, and swept
  chunk size, hybrid RRF fusion, and cross-encoder reranking against them.
- Raised all@5 from 0.720 to 0.880 (18% to 22% of 25) via hybrid retrieval,
  and MRR from 0.803 to 0.821 via cross-encoder reranking; the writeup states
  the corpus is small enough that hit@5 has little headroom, rather than
  overclaiming the delta.
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

**"Your hit@5 is 0.92 on a 33-chunk corpus. That's near-ceiling. Is the eval
even doing anything?"**

Fair challenge, and the answer is in the README. Top-5 covers a sixth of the
corpus, so hit@5 has little headroom — which is exactly why `all@5` is the
metric I lead with. It moved 0.720 → 0.880, and it moves on multi-hop
questions, which are the ones that actually decide answer quality. Hit@5 barely
moving is a property of the corpus, not evidence the ladder does nothing. The
fix is a bigger corpus, which I've scoped rather than hidden.

**"How do you know 0.92 is real and not overfitting to your own eval?"**

Three things. Labels are attached to documents, not chunks, so the same 30
items score every chunk size — the chunk sweep isn't seven independent chances
to get lucky. The relevance floor is published as a full trade-off curve, not
one tuned number, precisely because tuning a threshold on the set you report it
on would make the number meaningless. And I audited the metric: when I first
reported all@5 0.733, I found the denominator included 5 unanswerable questions
that no top-k retriever could ever win. That inflated nothing — it understated
the score by five unwinnable items. I'd rather catch my own metric error than
have an interviewer find it.

**"Why not fine-tune the embedder?"**

30 labels can't train one, and I'd rather spend the time on knowing whether the
current one works. Fine-tuning is a second experiment; being able to run the
first one is the prerequisite. The ablation harness is what makes a tuning
experiment a ten-minute job later rather than a rebuild.

**"Abstention is 1 out of 5. Is that a failed feature?"**

It's a measured limitation and the mechanism is visible. The cross-encoder
scores a topically-adjacent chunk highly for a confidently-worded unanswerable
question, so the two score distributions overlap — only 11 of 25 answerable
questions rank below *every* unanswerable one. No threshold separates classes
that overlap. With 5 items I also can't fit or validate a floor without
overfitting, which is why the deliverable is the curve and not a number. The
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

- **"Hit@5 of 0.92"** with no denominator. It reads as a smaller eval than it
  is, in the direction that flatters you. This is the number to be most careful
  with — quote it as "23 of 25".
- **"Improved retrieval by 27%."** 0.720 → 0.880 is a 0.16 absolute gain. Quote
  the absolute numbers; a percentage makes a small lift sound large.
- **"Production-grade."** It's a 14-page corpus and a Streamlit viewer.
