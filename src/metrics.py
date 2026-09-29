"""Retrieval metrics and the ablation ladder runner.

hit@k  - top-k contained at least one required doc
all@k  - top-k contained every required doc
MRR    - mean reciprocal rank of the first correct doc
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = [json.loads(l)
          for l in (ROOT / "eval/golden.jsonl").read_text().splitlines()]


def evaluate(size=512, method="dense", k=5, threshold=None, dense_weight=0.5,
             strategy="fixed"):
    """Score the two slices separately.

    Answerable and unanswerable items measure different things and have
    different denominators. Folding them into one number hid the fact that
    the unanswerable slice was unwinnable at the retrieval stage — a
    retriever returns top_k by construction and cannot return zero without
    an explicit threshold.
    """
    hits, alls, rr, ids = [], [], [], []
    abstained = []

    for item in GOLDEN:
        results = retrieve(item["question"], size=size, method=method,
                           top_k=k, threshold=threshold,
                           dense_weight=dense_weight, strategy=strategy)
        got = [c["doc_id"] for c, _ in results]
        need = set(item["answer_doc_ids"])

        if need:
            # A threshold that silences an answerable question is a false
            # abstention: count it as a miss on both metrics, not a skip.
            hits.append(bool(got) and any(d in need for d in got))
            alls.append(bool(got) and need.issubset(got))
            ids.append(item["id"])
            rr.append(next((1 / (i + 1) for i, d in enumerate(got)
                            if d in need), 0.0))
        else:
            abstained.append(not got)

    return {
        "hit@k": float(np.mean(hits)) if hits else 0.0,
        "all@k": float(np.mean(alls)) if alls else 0.0,
        "mrr": float(np.mean(rr)) if rr else 0.0,
        "abstain@k": float(np.mean(abstained)) if abstained else 0.0,
        "n_answerable": len(hits),
        "n_unanswerable": len(abstained),
        "missed_ids": [i for i, ok in zip(ids, alls) if not ok],
    }


def run_ablation():
    rows = []
    for size in (256, 512, 1024):
        for method in ("dense", "hybrid", "rerank"):
            if size != 512 and method == "rerank":
                continue  # reranker is the expensive rung; run it at 512 only
            r = evaluate(size=size, method=method, k=5)
            rows.append({"chunk_size": size, "method": method, **r})
            print(f"{size:>5}  {method:<8} "
                  f"hit@5={r['hit@k']:.3f} ({r['n_answerable']})  "
                  f"all@5={r['all@k']:.3f}  mrr={r['mrr']:.3f}  "
                  f"abstain@5={r['abstain@k']:.3f} ({r['n_unanswerable']})")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/ablation.json").write_text(json.dumps(rows, indent=2))

    md = ["Answerable slice and unanswerable slice are scored separately.",
          f"hit@5 / all@5 / MRR are over the {rows[0]['n_answerable']} answerable questions;",
          f"abstain@5 is over the {rows[0]['n_unanswerable']} unanswerable ones "
          "(no threshold: always 0).",
          "",
          "| chunk | method | hit@5 | all@5 | MRR | abstain@5 |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['chunk_size']} | {r['method']} | {r['hit@k']:.3f} | "
           f"{r['all@k']:.3f} | {r['mrr']:.3f} | {r['abstain@k']:.3f} |"
           for r in rows]
    (ROOT / "results/ablations.md").write_text("\n".join(md))
    print("\nwrote results/ablation.json and results/ablations.md")


def sweep_threshold(method="rerank", size=512, k=5):
    """Trade-off curve: how much answerable recall does a floor cost per
    unit of unanswerable rejection?

    Reported as a curve rather than a single chosen threshold. Picking one
    number on this set and reporting it as the abstention score would be
    fitting the threshold to the eval, so the reader gets the whole curve
    and can see where (or whether) a usable operating point exists.
    """
    top1 = []
    for item in GOLDEN:
        hits = retrieve(item["question"], size=size, method=method, top_k=k)
        if not hits:
            continue
        got = {c["doc_id"] for c, _ in hits}
        need = set(item["answer_doc_ids"])
        top1.append({"answerable": bool(need),
                     # Only questions that are already right can be regressed.
                     # Protecting the already-wrong ones is a no-op constraint.
                     "correct": bool(need) and need.issubset(got),
                     "score": hits[0][1]})

    pos = [r["score"] for r in top1 if r["answerable"]]
    neg = [r["score"] for r in top1 if not r["answerable"]]
    keep = [r["score"] for r in top1 if r["correct"]]
    grid = sorted({round(r["score"], 3) for r in top1})

    rows = []
    for t in grid:
        kept = sum(1 for s in pos if s >= t) / len(pos)
        rejected = sum(1 for s in neg if s < t) / len(neg)
        rows.append({"threshold": t, "answerable_all@k": kept,
                     "unanswerable_abstain@k": rejected})

    # Best point: max rejections subject to silencing nothing already
    # correct. A retention-% bound hides the cost — at n=25, "keep 90%"
    # still means losing 2 questions, and a strictly better zero-regression
    # point exists at the same rejection level.
    viable = [r for r in rows if all(s >= r["threshold"] for s in keep)]
    best = max(viable, key=lambda r: r["unanswerable_abstain@k"]) if viable else None

    print(f"\nthreshold sweep — {method} @ {size}, k={k}")
    print(f"{'floor':>9}  {'answerable kept':>16}  {'unanswerable rejected':>21}")
    for r in rows:
        mark = "  <- best viable" if best and r is best else ""
        print(f"{r['threshold']:>9.3f}  {r['answerable_all@k']:>16.3f}  "
              f"{r['unanswerable_abstain@k']:>21.3f}{mark}")

    (ROOT / "results/threshold.json").write_text(
        json.dumps({"method": method, "size": size, "k": k,
                    "curve": rows, "best": best}, indent=2))

    md = [f"Relevance floor sweep — {method} @ {size} tokens, k={k}.",
          "",
          "| floor | answerable all@5 kept | unanswerable abstain@5 |",
          "|---|---|---|"]
    md += [f"| {r['threshold']:.3f} | {r['answerable_all@k']:.3f} | "
           f"{r['unanswerable_abstain@k']:.3f} |" for r in rows]
    (ROOT / "results/threshold.md").write_text("\n".join(md))
    print("\nwrote results/threshold.json and results/threshold.md")
    return best


def sweep_dense_weight(k=5, size=512):
    """How much should fusion trust dense over BM25, per query?

    Equal weights (0.5) are the assumption the fusion-regression case breaks:
    a chunk dense ranks 2nd and BM25 ranks 13th lands 6th after equal fusion.
    Trusting dense more should recover it, and the cost is that a genuinely
    lexical query loses. Publishing the whole curve rather than one value
    keeps a weight picked on these 24 questions from reading as a tuned
    constant.

    dense_weight 0.0 is BM25 alone and 1.0 is dense alone, so the curve also
    shows what fusion buys at each end.
    """
    rows = []
    for w in [round(0.05 * i, 2) for i in range(21)]:
        r = evaluate(size=size, method="hybrid", k=k, dense_weight=w)
        rows.append({"dense_weight": w, **r})
        print(f"w={w:>4.2f}  hit@5={r['hit@k']:.3f}  all@5={r['all@k']:.3f}  "
              f"mrr={r['mrr']:.3f}  all@5 misses: {r['missed_ids'] or 'none'}")

    (ROOT / "results/dense_weight.json").write_text(json.dumps(rows, indent=2))
    md = ["Dense-vs-BM25 fusion weight sweep. dense_weight=0 is BM25 alone,",
          "1.0 is dense alone, 0.5 is the equal-weight RRF the system ships.",
          "Reported as a curve: any single value here was read off these 24",
          "answerable questions, so none of them is a tuned constant.",
          "",
          "| dense_weight | hit@5 | all@5 | MRR | all@5 misses |",
          "|---|---|---|---|---|"]
    md += [f"| {r['dense_weight']:.2f} | {r['hit@k']:.3f} | {r['all@k']:.3f} | "
           f"{r['mrr']:.3f} | {', '.join(map(str, r['missed_ids'])) or 'none'} |"
           for r in rows]
    (ROOT / "results/dense_weight.md").write_text("\n".join(md))
    print("\nwrote results/dense_weight.json and results/dense_weight.md")
    return rows


def sweep_chunking(k=5):
    """Is the fixed-size cut actually costing anything measurable?

    The sentence-aware chunker cuts mid-sentence endings from 78.8% to 60.0%
    of chunks. The question this answers is whether that shows up in retrieval,
    or whether it is a cosmetic improvement the metrics cannot see — which
    would be the more useful thing to know.
    """
    rows = []
    for strategy in ("fixed", "para"):
        for size in (256, 512, 1024):
            r = evaluate(size=size, method="hybrid", k=k, strategy=strategy)
            rows.append({"strategy": strategy, "chunk_size": size, **r})
            print(f"{strategy:>5} {size:>5}  hit@5={r['hit@k']:.3f}  "
                  f"all@5={r['all@k']:.3f}  mrr={r['mrr']:.3f}  "
                  f"misses: {r['missed_ids'] or 'none'}")

    (ROOT / "results/chunking.json").write_text(json.dumps(rows, indent=2))
    md = ["Fixed-size vs sentence-aware chunking, same retriever (hybrid, equal",
          "weights). `para` packs whole paragraph blocks and only splits inside a",
          "paragraph larger than the chunk size; the fixed build cuts on a token",
          "count and leaves 78.8% of chunks ending mid-sentence against 60.0%.",
          "",
          "| chunker | size | hit@5 | all@5 | MRR | all@5 misses |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['strategy']} | {r['chunk_size']} | {r['hit@k']:.3f} | "
           f"{r['all@k']:.3f} | {r['mrr']:.3f} | "
           f"{', '.join(map(str, r['missed_ids'])) or 'none'} |" for r in rows]
    (ROOT / "results/chunking.md").write_text("\n".join(md))
    print("\nwrote results/chunking.json and results/chunking.md")
    return rows


if __name__ == "__main__":
    run_ablation()
    sweep_dense_weight()
    sweep_threshold()
