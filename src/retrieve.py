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


def load_chunks(size=512):
    path = ROOT / f"data/chunks_{size}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]


_CACHE = {}


def indexes(size):
    """Build (chunks, dense, bm25) once per chunk size and keep them.

    Embedding the chunks is fast; doing it per query is what made the viewer
    feel broken. Keyed by size so the chunk-size switch still works.
    """
    if size not in _CACHE:
        chunks = load_chunks(size)
        _CACHE[size] = (chunks, DenseRetriever(chunks), BM25Retriever(chunks))
    return _CACHE[size]


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


def rrf(dense, sparse, k=60):
    """Reciprocal rank fusion. Combines rankings, not raw scores.

    Scores from different retrievers are not comparable (BM25 is unbounded,
    cosine is in [-1,1]), so we fuse on rank position instead.
    """
    n = len(dense)
    out = np.zeros(n)
    for scores in (dense, sparse):
        order = np.argsort(-scores)
        for rank, idx in enumerate(order):
            out[idx] += 1.0 / (k + rank + 1)
    return out


def retrieve(query, size=512, method="hybrid", top_k=5, rerank_n=20,
              threshold=None):
    """Return up to top_k (chunk, score) pairs, best first.

    `threshold` is a relevance floor: if the best score falls under it,
    return nothing rather than the top_k regardless. This is the only way
    the system can decline a question it has no evidence for.

    The floor is per-method and not comparable across methods — cosine sits
    in [-1,1], RRF sums 1/(k+rank), the cross-encoder emits logits.
    """
    chunks, dense_r, bm25_r = indexes(size)
    dense = dense_r.scores(query)
    scores = dense

    if method in ("hybrid", "rerank"):
        scores = rrf(dense, bm25_r.scores(query))

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
