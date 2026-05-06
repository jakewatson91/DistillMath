# Benchmark Summary

| Metric | Teacher 7B | Base 1.5B | Distilled 1.5B |
|---|---|---|---|
| GSM8K accuracy | 85.4% | 66.3% | 80.5% |
| Peak VRAM | OOM (>8 GB) | 3.31 GB | 3.29 GB |
| Latency p50 | — | 17.2 ms/tok | 17.1 ms/tok |
| Throughput @ bs=1 | — | 58 tok/s | 59 tok/s |
| Throughput @ bs=8 | — | 394 tok/s | 395 tok/s |
| Max throughput | — | 790 tok/s | 794 tok/s |
| Energy / correct | — | 197 J | 158 J |

**Notes**

- Teacher 7B does not load under the 8 GB cap; speed and memory metrics are N/A. Accuracy is cited from prior measurement.
- All measurements use greedy decode (`do_sample=False`) for reproducibility.
- Energy = total joules sampled at 10 Hz over the GSM8K eval, divided by # correct answers.
