# Eval-driven RAG over football-analytics docs

A retrieval-augmented Q&A system over the `socceraction` documentation, built
around one question: **does a change to the retriever actually improve
answers, or does it just look better on a metric?** Everything here is
measured on a hand-labeled set and reproducible from a script.

## Why this shape

Most RAG writeups show one number and no way to check it. This one ships the
eval harness, the traces behind the number, and the cost/latency measurements
for the same pipeline — so the claims are auditable rather than asserted.

## How it works

```mermaid
flowchart LR
    Q([question]) --> C[corpus<br/>14 pages]
    C --> K[chunker<br/>512 tokens]
    K --> D[dense<br/>MiniLM embeddings]
    K --> B[BM25<br/>word overlap]
    D --> R[RRF fusion<br/>fuse on rank]
    B --> R
    R --> X[cross-encoder<br/>rerank top 20]
    X --> F{top score<br/>≥ floor?}
    F -->|no| N["abstain<br/>I don't know"]
    F -->|yes| L[LLM answer<br/>+ citations]
    L --> V([streamlit trace viewer])
    N --> V
```

Every box is a real function: chunker in `src/chunk.py`, the three retrievers
and `rrf()` in `src/retrieve.py`, the floor branch in `retrieve()`, and
generation in `src/answer.py`.

Dense retrieval matches on meaning; BM25 matches on words. They fail
differently — the embedding handles "how do I start?" → "installation", BM25
handles the exact product name the embedding has never seen — so fusing them
beats either alone. The cross-encoder is a second model that reads the
question and each candidate chunk *together* and reorders them. It is the
slowest stage and cannot rescue a document the first stage missed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # only needed for generation and the load test
python3 scripts/fetch_corpus.py   # downloads + cleans corpus/ (optional; already committed)
```

Retrieval and the ablation ladder need no API key at all — they run entirely
on local models. Only `app.py`'s Generate button and `bench/loadtest.py` call a
generation API.

## Reproduce

```bash
python3 src/chunk.py 512           # corpus -> data/chunks_512.jsonl
python3 src/metrics.py             # ablation ladder -> results/ablations.md
python3 bench/loadtest.py --concurrency 1,4,8 --n 20   # results/latency.json
streamlit run app.py               # trace viewer
```

If you are reviewing this as a candidate: [`docs/INTERVIEW.md`](docs/INTERVIEW.md)
has a 30-second pitch, resume bullets, and the questions an AI Engineer
interviewer is likely to ask, with answers.

## Results

### Retrieval (`results/ablations.md`)

Two slices, two denominators, never folded together:

- **25 answerable** questions — `hit@5`, `all@5`, MRR
- **5 unanswerable** questions — `abstain@5`, the share where the retriever
  returned nothing. 0.000 on every row below because these runs use no
  relevance floor; see the next section.

| chunk | method | hit@5 | all@5 | MRR | abstain@5 |
|---|---|---|---|---|---|
| 256 | dense | 0.880 | 0.720 | 0.775 | 0.000 |
| 256 | hybrid | 0.920 | 0.760 | 0.757 | 0.000 |
| 512 | dense | 0.920 | 0.760 | 0.795 | 0.000 |
| 512 | hybrid | 0.920 | 0.880 | 0.803 | 0.000 |
| 512 | rerank | 0.920 | 0.880 | 0.821 | 0.000 |
| 1024 | dense | 0.920 | 0.800 | 0.756 | 0.000 |
| 1024 | hybrid | 0.920 | 0.880 | 0.760 | 0.000 |

**Correcting an earlier number.** This table previously reported `hit@5` 0.767
and `all@5` 0.733 for the best row. Those were computed over all 30 questions,
including the 5 with no gold documents — items a top-5 retriever can never get
right, because returning `top_k` results is what it is built to do. Leaving
them in the denominator understated the headline by five unwinnable items. The
real best row is **hit@5 0.920 (23/25), all@5 0.880 (22/25)**.

**Read this honestly.** The deltas are small — 0.880 → 0.920 on `hit@5` —
and that is a property of the corpus, not a failure of the methods. At 512
tokens the corpus is only ~33 chunks, so top-5 covers a sixth of it and the
task has little headroom. A metric that cannot move is not evidence the
ladder works. The claims that do hold:

- `all@5` moves more than `hit@5` (0.720 → 0.880), and `all@5` is the metric
  that predicts answer quality on multi-hop questions. Hybrid retrieval is
  what helps — it finds the *second* required document, not just the first.
- The cross-encoder reranker lifts MRR (0.803 → 0.821) without changing
  `hit@5` or `all@5`. It reorders, it does not rescue — exactly the expected
  behavior of a second-stage precision tool, and worth naming as such.
- The eval set is the asset. 30 hand-labeled items with doc-level labels,
  reusable across chunk sizes because labels attach to documents, not chunks.

The fix for the headroom problem is a larger corpus, not a better retriever.
That is scoped as the next iteration rather than quietly omitted.

### Abstention (`results/threshold.md`)

A retriever that always returns 5 chunks cannot answer "I don't know." The
only way out is a relevance floor: if the best score is under it, return
nothing. The floor is a threshold, so reporting one number would mean tuning
it on the same 30 questions the score is reported on. Instead, the whole
trade-off curve is published:

| floor | answerable all@5 kept | unanswerable abstain@5 |
|---|---|---|
| -10.249 | 1.000 | 0.200 | ← selected |
| -7.574 | 1.000 | 0.200 |
| -6.334 | 0.920 | 0.200 |
| -6.248 | 0.920 | 0.600 |
| -5.685 | 0.880 | 0.600 |
| -3.330 | 0.800 | 0.800 |
| -0.271 | 0.600 | 0.800 |
| 0.022 | 0.560 | 1.000 |
| 7.214 | 0.040 | 1.000 |

**This result is unimpressive and the table shows why.** The zero-regression
point rejects **1 of 5** unanswerable questions. Pushing higher buys rejection
only by silencing questions the corpus can answer: at -6.248 the system
declines 3 of 5 unanswerables, but it also silently refuses to answer id 1
("How do I get started with the library for the first time?") and id 23.
Getting to a perfect 5/5 costs 44% of answerable recall.

The selection rule in `sweep_threshold()` is therefore *maximize rejection
subject to silencing nothing already correct* — not a retention-percentage
bound. An earlier version used "keep ≥90% of answerable," which is only 2
questions at n=25, and it picked a floor that broke a real query while a
strictly better zero-regression point sat one row away.

Score overlap is the underlying cause. At 512/rerank, only 11 of 25 answerable
questions sit entirely below every unanswerable one, versus 21 of 25 for
dense. Reranking is the method worth building a floor for — and even it cannot
separate the classes, because a confidently-worded unanswerable question
retrieves a topically-adjacent chunk and the cross-encoder scores it high.
The honest conclusion is that 5 unanswerable items is too few to fit or
validate a floor on; the curve is the deliverable, the operating point is not
a claim.

### Generation and serving — harness written, not yet measured

`bench/loadtest.py` is complete: it sweeps concurrency, arms the prompt cache
on and off, and reports p50/p95/p99, throughput, and cost per question
including the one-off cache write. Every result row carries
`cache_engaged`, so a run where the provider silently ignored
`cache_control` cannot be mistaken for a caching win.

**No latency table is published, because no valid run completed.** Two
independent blockers, both environmental:

- The dev proxy in this environment exposes exactly one usable model
  (`ling-3.0-flash-sante:free`) and strips `cache_control` from requests, so
  the caching A/B cannot be measured — both arms would be the same run.
- That model's single shared credential enters a ~100 s cooldown after a
  handful of requests, so concurrency > 1 cannot be sustained long enough to
  produce a p99.

Partial serial data from the one arm that did complete, before cooldowns began:

```
c=1  cache=False  p50=3.87s  p99=10.85s  qps=0.19  $0.380/20q
c=1  cache=True   p50=4.45s  p99=19.68s  qps=0.17  $0.264/20q
```

Those two rows are **not** a result. The cache arm was cheaper only because
the provider dropped the cache directives, and the latency difference is
single-sample noise. Quoting them would be exactly the overclaiming this
project exists to avoid.

To produce the real table, run against a direct Anthropic key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_BASE_URL=https://api.anthropic.com   # bypass the local proxy
unset ANTHROPIC_DEFAULT_HAIKU_MODEL
export RAG_MODEL=claude-sonnet-5
python3 bench/loadtest.py --concurrency 1,4,8 --n 40
```

## Design notes worth knowing

**Fusion is on rank, not score.** BM25 output is unbounded and cosine
similarity lives in [-1, 1]. Adding them is meaningless, so `rrf()` sums
`1/(k + rank)` instead. This is why `src/retrieve.py:rrf` takes rankings.

**Reranking is a precision tool.** It re-scores the top 20 candidates with a
cross-encoder. It cannot surface a document the first-stage retriever missed,
which is exactly what the `hit@5` vs MRR split above shows.

**Unanswerable questions are scored as a separate slice.** For an item with no
gold documents, retrieving anything confident is counted wrong, over its own
denominator of 5. Folding it into `all@5` made the metric unwinnable on those
items and understated the rest. A relevance floor is the only way to score
non-zero there, and the floor ships as a curve, not a tuned constant.

**The corpus is small and that is stated, not hidden.** 14 pages, ~12.5k
tokens. Two dead sources were dropped mid-build (`pitchcontrol` docs, which no
longer exist, and a football news site that was methodology-free). A fresh
clone that wants a bigger fight should extend `SOURCES` in
`scripts/fetch_corpus.py`; the eval harness does not need changes, because
labels are doc-level.

## Layout

```
corpus/            cleaned text, one file per doc
eval/golden.jsonl  30 hand-labeled items
src/chunk.py       fixed-size token chunker
src/retrieve.py    dense / BM25 / RRF hybrid / cross-encoder rerank
src/metrics.py     hit@k, all@k, MRR + ablation runner
src/answer.py      generation with prompt-prefix caching
app.py             Streamlit trace viewer
bench/loadtest.py  concurrency sweep, latency percentiles, cost
results/           generated tables
```
