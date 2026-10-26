Keyless extractive answer path — rerank retrieval, 512-token chunks, k=5. n=20 per concurrency level.

| concurrency | p50 | p95 | p99 | qps | wall |
|---|---|---|---|---|---|
| 1 | 0.357s | 0.389s | 0.390s | 2.77 | 7.23s |
| 4 | 1.419s | 1.603s | 1.605s | 2.76 | 7.25s |
| 8 | 2.865s | 2.960s | 2.971s | 2.74 | 7.31s |