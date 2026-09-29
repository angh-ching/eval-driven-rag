"""Load test: sweep concurrency, report p50/p99 latency and token cost.

Compares the same run with and without prompt caching. The first call in
each config is discarded from the timing: it pays the cache write, and
including it would flatter the uncached arm on cost. Its tokens are added
back separately so the write is still reported.

  python3 bench/loadtest.py --concurrency 1,4,8 --n 20
"""
import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from answer import answer  # noqa: E402
from retrieve import indexes  # noqa: E402

GOLDEN = [json.loads(l) for l in
          (ROOT / "eval/golden.jsonl").read_text().splitlines()]
ANSWERABLE = [g["question"] for g in GOLDEN if g["answer_doc_ids"]]

# Published Sonnet list prices, USD per million tokens. Pinned in the file
# so the cost column is reproducible without re-checking pricing.
IN_PRICE, OUT_PRICE = 3.0 / 1e6, 15.0 / 1e6
CACHE_READ_PRICE = IN_PRICE * 0.1
CACHE_WRITE_PRICE = IN_PRICE * 1.25


def price(r):
    return (r["input_tokens"] * IN_PRICE
            + r["output_tokens"] * OUT_PRICE
            + r["cache_read_tokens"] * CACHE_READ_PRICE
            + r["cache_write_tokens"] * CACHE_WRITE_PRICE)


def call(q, client, cache_prompt, retries=6):
    """One request, honouring the server's own cooldown.

    Free-tier providers cool down under concurrency, and each rejected
    attempt re-arms the timer — so exponential backoff starting small just
    burns the retry budget. Sleep the `reset_seconds` the 429 reports, plus
    margin. Retrying keeps the sample full instead of dropping slow requests,
    which would bias p99 downward.
    """
    for attempt in range(retries):
        try:
            return answer(q, client=client, cache_prompt=cache_prompt)
        except Exception as e:
            if "429" not in str(e) and "RateLimit" not in type(e).__name__:
                raise
            if attempt == retries - 1:
                raise
            m = re.search(r"reset_seconds['\"]?\s*[:=]\s*(\d+)", str(e))
            wait = int(m.group(1)) + 10 if m else 2 ** attempt * 30
            print(f"    429, sleeping {wait}s (attempt {attempt + 1})",
                  flush=True)
            time.sleep(wait)


def one_config(questions, concurrency, cache_prompt, client):
    # Warm the prefix cache and the retrieval index before timing. The warm
    # call pays the cache write; we keep its cost but drop its latency.
    indexes(512)
    warm = call(questions[0], client, cache_prompt)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda q: call(q, client, cache_prompt), questions))
    wall = time.perf_counter() - t0

    lat = [r["latency_s"] for r in rows]
    cache_reads = sum(r["cache_read_tokens"] for r in rows)
    return {
        "concurrency": concurrency,
        "cache_prompt": cache_prompt,
        # If the provider silently ignores cache_control, reads stay 0 and
        # this is False — the two arms are then the same run twice and the
        # cost difference between them is noise. Check it before citing.
        "cache_engaged": cache_reads > 0,
        "n": len(rows),
        "wall_s": wall,
        "qps": len(rows) / wall,
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "mean": statistics.fmean(lat),
        "cost_usd": sum(price(r) for r in rows) + price(warm),
        "cache_read_tokens": cache_reads,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results/latency.json"))
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    qs = [ANSWERABLE[i % len(ANSWERABLE)] for i in range(args.n)]

    rows = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        for cache_prompt in (False, True):
            r = one_config(qs, c, cache_prompt, client)
            rows.append(r)
            flag = "" if (not cache_prompt or r["cache_engaged"]) else \
                "  [cache_engaged=false: provider ignored cache_control]"
            print(f"c={c:<2} cache={str(cache_prompt):<5}  "
                  f"p50={r['p50']:.2f}s  p99={r['p99']:.2f}s  "
                  f"qps={r['qps']:.2f}  ${r['cost_usd']:.3f}/{r['n']}q{flag}")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
