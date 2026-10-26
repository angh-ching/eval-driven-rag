"""Load test for the keyless answer path.

Sweeps concurrency and reports p50/p95/p99 latency and throughput for the
full request: retrieval plus answer extraction. There is no API client, no
token cost, and no cache arm, because the default path makes no paid call.

  python3 bench/loadtest.py --concurrency 1,4,8 --n 20

Set RAG_PROVIDER=ollama to benchmark a local Ollama model instead. That is
still keyless, but the numbers are model- and machine-specific.
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from answer import answer, provider_name  # noqa: E402
from retrieve import indexes  # noqa: E402

GOLDEN = [json.loads(l) for l in
          (ROOT / "eval/golden.jsonl").read_text().splitlines()]
ANSWERABLE = [g["question"] for g in GOLDEN if g["answer_doc_ids"]]


def one_config(questions, concurrency, method, size, k):
    # Build the retrieval indexes and warm the reranker before timing. The
    # warm request is excluded from the percentiles; it measures setup, not
    # request latency.
    indexes(size)
    answer(questions[0], size=size, method=method, top_k=k)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(
            lambda q: answer(q, size=size, method=method, top_k=k),
            questions,
        ))
    wall = time.perf_counter() - t0

    lat = [r["latency_s"] for r in rows]
    return {
        "provider": provider_name(),
        "method": method,
        "chunk_size": size,
        "k": k,
        "concurrency": concurrency,
        "n": len(rows),
        "wall_s": wall,
        "qps": len(rows) / wall,
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "mean": statistics.fmean(lat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--method", default="rerank",
                    choices=["dense", "hybrid", "rerank"])
    ap.add_argument("--size", type=int, default=512,
                    choices=[256, 512, 1024])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results/latency.json"))
    args = ap.parse_args()

    qs = [ANSWERABLE[i % len(ANSWERABLE)] for i in range(args.n)]
    rows = []
    print(f"provider={provider_name()} method={args.method} "
          f"size={args.size} k={args.k} n={args.n}")
    for c in [int(x) for x in args.concurrency.split(",")]:
        r = one_config(qs, c, args.method, args.size, args.k)
        rows.append(r)
        print(f"c={c:<2}  p50={r['p50']:.3f}s  p95={r['p95']:.3f}s  "
              f"p99={r['p99']:.3f}s  qps={r['qps']:.2f}  "
              f"wall={r['wall_s']:.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    md = [f"Keyless {rows[0]['provider']} answer path — {args.method} "
          f"retrieval, {args.size}-token chunks, k={args.k}. "
          f"n={args.n} per concurrency level.",
          "",
          "| concurrency | p50 | p95 | p99 | qps | wall |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['concurrency']} | {r['p50']:.3f}s | {r['p95']:.3f}s | "
           f"{r['p99']:.3f}s | {r['qps']:.2f} | {r['wall_s']:.2f}s |"
           for r in rows]
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md))
    print(f"\nwrote {out} and {md_path}")


if __name__ == "__main__":
    main()
