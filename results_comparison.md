# Qwen3-ASR-1.7B: Error Corrector Evaluation (Waihu Dataset)

**Dataset:** Waihu

---

## Accuracy

| Metric | Without EC | With EC | Delta |
|--------|-----------|---------|-------|
| CER    | 14.08%    | 13.40%  | **−0.68%** |
| MER    | 14.14%    | 13.46%  | **−0.68%** |

The error corrector reduces CER/MER by ~0.68% points (~4.8% relative improvement).

---

## Latency (ms)

### First Token Latency (FTL)

| Stat   | Without EC | With EC  | Delta     |
|--------|-----------|---------|-----------|
| Min    | 23.8      | 111.9   | +88.1     |
| Max    | 56,791.8  | 44,330.3 | −12,461.5 |
| Median | 70.8      | 342.9   | **+272.1** |

### Last Token Latency (LTL)

| Stat   | Without EC | With EC  | Delta     |
|--------|-----------|---------|-----------|
| Min    | 0.0       | 0.0     | —         |
| Max    | 56,686.1  | 9,729.3  | −46,956.8 |
| Median | 41.7      | 164.5   | **+122.8** |

---

## Summary

- **Accuracy:** The error corrector improves CER by ~0.68 pp (14.08% → 13.40%).
- **Median latency:** Adding the EC increases median FTL (71 ms → 343 ms) and median LTL (42 ms → 165 ms), which is the expected cost of running the corrector model on each chunk.
- **Trade-off:** The EC offers a modest accuracy gain at the cost of ~270 ms additional median first-token latency — worthwhile for accuracy-sensitive use cases, potentially too slow for strict real-time requirements.
