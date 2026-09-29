"""Retrieval strategies over a chunk set.

Rungs of the ablation ladder, each building on the last:
  dense      - embedding cosine similarity
  hybrid     - dense + BM25 fused by reciprocal rank
  rerank     - hybrid candidates rescored by a cross-encoder
"""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANKER = None  # loaded lazily; only the rerank rung needs it


def load_chunks(size=512, strategy="fixed"):
    suffix = "" if strategy == "fixed" else f"_{strategy}"
    path = ROOT / f"data/chunks_{size}{suffix}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]


_CACHE = {}


def indexes(size, strategy="fixed"):
    """Build (chunks, dense, bm25) once per chunk size and strategy.

    Embedding the chunks is fast; doing it per query is what made the viewer
    feel broken. Keyed by (size, strategy) so the chunk-size switch still works
    and the sentence-aware build is a separate axis from the size.
    """
    key = (size, strategy)
    if key not in _CACHE:
        chunks = load_chunks(size, strategy)
        _CACHE[key] = (chunks, DenseRetriever(chunks), BM25Retriever(chunks))
    return _CACHE[key]


def _embed(texts, model=None):
    return (model or MODEL).encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False)


class DenseRetriever:
    def __init__(self, chunks, model=None):
        self.chunks = chunks
        self.model = model or MODEL
        self.vecs = _embed([c["text"] for c in chunks], self.model)

    def scores(self, query):
        return self.vecs @ _embed([query], self.model)[0]


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        corpus = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def scores(self, query):
        return np.array(self.bm25.get_scores(re.findall(r"\w+", query.lower())))


def rrf(dense, sparse, k=60, weights=(1.0, 1.0)):
    """Reciprocal rank fusion. Combines rankings, not raw scores.

    Scores from different retrievers are not comparable (BM25 is unbounded,
    cosine is in [-1,1]), so we fuse on rank position instead.

    `weights` scales each retriever's contribution. Equal weights assume the
    two are equally trustworthy per query, which the fusion-regression case
    shows is false: a chunk dense ranks 2nd and BM25 ranks 13th lands 6th
    after equal fusion, just outside the cut.
    """
    n = len(dense)
    out = np.zeros(n)
    for weight, scores in zip(weights, (dense, sparse)):
        if weight == 0:
            continue
        order = np.argsort(-scores)
        for rank, idx in enumerate(order):
            out[idx] += weight / (k + rank + 1)
    return out


def retrieve(query, size=512, method="hybrid", top_k=5, rerank_n=20,
              threshold=None, dense_weight=0.5, strategy="fixed"):
    """Return up to top_k (chunk, score) pairs, best first.

    `threshold` is a relevance floor: if the best score falls under it,
    return nothing rather than the top_k regardless. This is the only way
    the system can decline a question it has no evidence for.

    The floor is per-method and not comparable across methods — cosine sits
    in [-1,1], RRF sums 1/(k+rank), the cross-encoder emits logits.
    """
    chunks, dense_r, bm25_r = indexes(size, strategy)
    dense = dense_r.scores(query)
    scores = dense

    if method in ("hybrid", "rerank"):
        # Scaled so the pair sums to 2, matching rrf()'s equal-weight default:
        # dense_weight=0.5 is the original unweighted RRF, exactly.
        scores = rrf(dense, bm25_r.scores(query),
                     weights=(2 * dense_weight, 2 * (1 - dense_weight)))

    order = np.argsort(-scores)[:rerank_n]

    if method == "rerank":
        global RERANKER
        if RERANKER is None:
            from sentence_transformers import CrossEncoder
            RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, chunks[i]["text"]) for i in order]
        ce = RERANKER.predict(pairs)
        final = sorted(zip(order, ce), key=lambda x: -x[1])[:top_k]
        out = [(chunks[i], float(s)) for i, s in final]
    else:
        out = [(chunks[i], float(scores[i])) for i in order[:top_k]]

    if threshold is not None and (not out or out[0][1] < threshold):
        return []
    return out


def _selfcheck():
    """The default fusion path must be bit-identical to unweighted RRF, and the
    weight endpoints must be the single retrievers. A weight pair that does not
    sum to 2 silently rescales every fused score, which changes the relevance
    floor's meaning without changing any ranking."""
    import numpy as np
    d = np.array([0.9, 0.8, 0.7, 0.1])
    b = np.array([0.1, 0.2, 0.3, 0.9])

    def unweighted():
        out = np.zeros(len(d))
        for s in (d, b):
            for rank, idx in enumerate(np.argsort(-s)):
                out[idx] += 1.0 / (60 + rank + 1)
        return out

    assert np.allclose(rrf(d, b), unweighted()), "default weights changed"
    assert np.allclose(rrf(d, b, weights=(1.0, 1.0)), unweighted())
    assert list(np.argsort(-rrf(d, b, weights=(2.0, 0.0)))) == \
        list(np.argsort(-d)), "dense_weight=1.0 is not dense-only"
    assert list(np.argsort(-rrf(d, b, weights=(0.0, 2.0)))) == \
        list(np.argsort(-b)), "dense_weight=0.0 is not BM25-only"
    print("rrf selfcheck ok: default = unweighted, endpoints = single retrievers")


if __name__ == "__main__":
    _selfcheck()
