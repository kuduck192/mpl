| method | 40 labels | 100 labels |
|---|---|---|
| Supervised | 56.88 ± 1.23 (n=3) | 69.50 ± 0.70 (n=3) |
| Pseudo-Label | 64.18 ± 1.29 (n=3) | 74.71 ± 1.32 (n=3) |
| Mean Teacher | 58.16 ± 0.59 (n=3) | 67.45 ± 1.08 (n=3) |
| FixMatch | 70.96 ± 3.69 (n=5) | 79.68 ± 3.76 (n=5) |
| FlexMatch | 71.11 ± 2.32 (n=3) | 77.88 ± 1.55 (n=3) |
| MPL + FixMatch (optimistic) | 71.06 ± 4.11 (n=5) | 81.65 ± 3.35 (n=5) |
| **EBPL + FixMatch (expected)** | 73.00 ± 3.49 (n=3) | 81.88 ± 1.54 (n=3) |
| **PBPL + FixMatch (pessimistic)** | 72.98 ± 3.38 (n=5) | 81.34 ± 4.08 (n=5) |

Paired difference against FixMatch on common seeds (percentage points):

| method | labels | seeds | mean diff | wins | paired t p-value |
|---|---|---|---|---|---|
| Supervised | 40 | 1,2,3 | -13.77 ± 3.58 | 0/3 | 0.022 |
| Supervised | 100 | 1,2,3 | -8.02 ± 2.57 | 0/3 | 0.033 |
| Pseudo-Label | 40 | 1,2,3 | -6.47 ± 5.14 | 0/3 | 0.161 |
| Pseudo-Label | 100 | 1,2,3 | -2.82 ± 4.27 | 1/3 | 0.371 |
| Mean Teacher | 40 | 1,2,3 | -12.49 ± 3.95 | 0/3 | 0.032 |
| Mean Teacher | 100 | 1,2,3 | -10.07 ± 3.69 | 0/3 | 0.042 |
| FlexMatch | 40 | 1,2,3 | +0.46 ± 3.54 | 1/3 | 0.842 |
| FlexMatch | 100 | 1,2,3 | +0.36 ± 2.72 | 1/3 | 0.842 |
| MPL + FixMatch (optimistic) | 40 | 1,2,3,4,5 | +0.10 ± 2.62 | 2/5 | 0.937 |
| MPL + FixMatch (optimistic) | 100 | 1,2,3,4,5 | +1.97 ± 4.16 | 3/5 | 0.349 |
| **EBPL + FixMatch (expected)** | 40 | 1,2,3 | +2.36 ± 1.32 | 3/3 | 0.091 |
| **EBPL + FixMatch (expected)** | 100 | 1,2,3 | +4.36 ± 4.80 | 2/3 | 0.256 |
| **PBPL + FixMatch (pessimistic)** | 40 | 1,2,3,4,5 | +2.02 ± 1.18 | 5/5 | 0.019 |
| **PBPL + FixMatch (pessimistic)** | 100 | 1,2,3,4,5 | +1.66 ± 3.99 | 2/5 | 0.405 |

Cost and end-of-training diagnostics (mean over runs; 40 + 100 labels pooled):

| method | minutes / run (T4) | pseudo-label acc. above threshold | mask rate | student test acc. |
|---|---|---|---|---|
| Supervised | 2.3 | — | — | — |
| Pseudo-Label | 5.2 | 73.0 | 91.2 | — |
| Mean Teacher | 10.4 | 63.5 | 100.0 | — |
| FixMatch | 11.3 | 78.7 | 89.8 | — |
| FlexMatch | 11.3 | 76.9 | 91.6 | — |
| MPL + FixMatch (optimistic) | 27.9 | 80.5 | 87.6 | 76.3 |
| **EBPL + FixMatch (expected)** | 37.9 | 82.1 | 87.1 | 77.5 |
| **PBPL + FixMatch (pessimistic)** | 40.6 | 81.3 | 87.4 | 77.2 |
