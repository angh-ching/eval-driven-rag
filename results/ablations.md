Answerable slice and unanswerable slice are scored separately.
hit@5 / all@5 / MRR are over the 24 answerable questions;
abstain@5 is over the 6 unanswerable ones (no threshold: always 0).

| chunk | method | hit@5 | all@5 | MRR | abstain@5 |
|---|---|---|---|---|---|
| 256 | dense | 0.917 | 0.750 | 0.807 | 0.000 |
| 256 | hybrid | 1.000 | 0.875 | 0.812 | 0.000 |
| 512 | dense | 1.000 | 0.875 | 0.838 | 0.000 |
| 512 | hybrid | 1.000 | 0.917 | 0.889 | 0.000 |
| 512 | rerank | 1.000 | 0.917 | 0.910 | 0.000 |
| 1024 | dense | 1.000 | 0.875 | 0.808 | 0.000 |
| 1024 | hybrid | 1.000 | 0.917 | 0.840 | 0.000 |