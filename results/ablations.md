Answerable slice and unanswerable slice are scored separately.
hit@5 / all@5 / MRR are over the 25 answerable questions;
abstain@5 is over the 5 unanswerable ones (no threshold: always 0).

| chunk | method | hit@5 | all@5 | MRR | abstain@5 |
|---|---|---|---|---|---|
| 256 | dense | 0.880 | 0.720 | 0.775 | 0.000 |
| 256 | hybrid | 0.920 | 0.760 | 0.757 | 0.000 |
| 512 | dense | 0.920 | 0.760 | 0.795 | 0.000 |
| 512 | hybrid | 0.920 | 0.880 | 0.803 | 0.000 |
| 512 | rerank | 0.920 | 0.880 | 0.821 | 0.000 |
| 1024 | dense | 0.920 | 0.800 | 0.756 | 0.000 |
| 1024 | hybrid | 0.920 | 0.880 | 0.760 | 0.000 |