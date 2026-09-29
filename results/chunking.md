Fixed-size vs sentence-aware chunking, same retriever (hybrid, equal
weights). `para` packs whole paragraph blocks and only splits inside a
paragraph larger than the chunk size; the fixed build cuts on a token
count and leaves 78.8% of chunks ending mid-sentence against 60.0%.

| chunker | size | hit@5 | all@5 | MRR | all@5 misses |
|---|---|---|---|---|---|
| fixed | 256 | 1.000 | 0.875 | 0.812 | 9, 18, 24 |
| fixed | 512 | 1.000 | 0.917 | 0.889 | 19, 24 |
| fixed | 1024 | 1.000 | 0.917 | 0.840 | 19, 24 |
| para | 256 | 0.958 | 0.792 | 0.792 | 9, 16, 18, 19, 24 |
| para | 512 | 0.958 | 0.917 | 0.833 | 9, 16 |
| para | 1024 | 0.958 | 0.875 | 0.847 | 9, 16, 19 |