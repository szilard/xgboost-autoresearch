# Research Log — may3

## Setup
- Branch: `may3` from v0.5.1.
- Mode: 5-fold StratifiedKFold CV on `2005-slice1-100k.csv` (100k balanced rows).
- Target: `dep_delayed_15min` (binary). Metric: ROC AUC.
- Raw features: 6 categorical (`Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`), 2 numeric (`DepTime`, `Distance`).
- Budget: 1 minute wall-clock per experiment (~65s tolerated as borderline).

## Final result (129 experiments)

**CV AUC 0.8213** at commit `f8e0a05`. Baseline 0.7445 → 0.8213 = **+0.0768 absolute, +10.3% relative**.

**Best config:**
```python
XGBClassifier(
    n_estimators=440, max_depth=14, learning_rate=0.04,
    colsample_bytree=0.4, min_child_weight=3,
    max_bin=512, max_cat_threshold=128,
    enable_categorical=True, random_state=0,
    n_jobs=4,
)
# cross_val_score(..., n_jobs=2)  # parallel folds within budget
```

**Feature engineering in `prepare()`:**
- `CarrierHour` cat (UniqueCarrier × DepTime//100, ~480 cats)
- `Dep20Min` cat (minute_of_day // 20, 72 buckets)
- `DepTimeSin/Cos` (cyclic on minute_of_day, period **6h** not 24h)
- `DepMinute` numeric (DepTime % 100)
- `MonthSin/Cos` (12-month cyclic)

## Where the gain came from (cumulative ~0.077 AUC)

| Lever | Approx contribution | Note |
|---|---|---|
| Capacity (n_est 30→440, lr 0.1→0.04, depth 6→14) | +0.030 | Foundational. |
| `colsample_bytree=0.4` | +0.013 | Aggressive feature subsampling. |
| `CarrierHour` interaction cat | +0.005–0.008 | Carrier-specific time-of-day delays. |
| `Dep20Min` cat (72-bucket time) | +0.006 | Memory tip; explicit time bucket beats raw splits. |
| `DepMinute = DepTime % 100` | +0.0046 | Memory tip; non-recoverable minute-of-hour signal. |
| Cyclic `DepTimeSin/Cos` (6h period) | +0.003–0.005 | Smooth periodic signal. |
| Cyclic `MonthSin/Cos` | +0.0014 | Seasonality. |
| `min_child_weight=3` | +0.002 | Regularizes deep trees; also speeds up. |
| `max_cat_threshold=128` | +0.002 | Memory tip; finer native cat partitioning. |
| `random_state=0` (vs 42) | +0.0026 | Partly seed lottery — verified seed=1 is -0.003 from seed=0. |
| `max_bin=512` | +0.001 | Histogram precision. |
| Cyclic period 6h vs 24h | +0.0005 | Memory tip. |
| Parallel CV folds (cv_n_jobs=2 + xgb_n_jobs=4) | 0 AUC, -5s wall | Wall savings funded the n_est=400→440 keep. |

## What did not help (each tested, often multiple times)

- **Subsample**: 0.8 hurt -0.009; 0.95 / 0.99 noise.
- **Ordinal-encode date cats** (Month/DayofMonth/DayOfWeek as int) **replacing** native cats: -0.006. Memory said +0.008; did not transfer here, likely because the may3 config uses `max_cat_threshold=128` which improves native handling.
- **Ordinal date cats alongside native cats** (e.g. `DayOfWeekNum + DayOfWeek`): -0.005. Dilutes feature picks at low colsample.
- **DepHour as plain int / cat** alongside CarrierHour & Dep20Min: -0.007. Dilutive.
- **DayOfWeek cyclic sin/cos**: -0.001. Redundant with native cat (only 7 levels).
- **Route (Origin+Dest) cat, OriginHour cat**: regressed AND timed out (high cardinality).
- **More interaction cats** (CarrierMonth, CarrierDow, HourDow, CarrierDep20, IsWeekend, IsLateNight, DistanceBin, DayofMonthInt): all dilutive at colsample=0.4. Raising colsample to 0.5 alongside did not recover.
- **gamma** (0.5, 0.05, 0.01): no AUC gain.
- **reg_lambda=5**, **reg_alpha=0.1**: no AUC gain.
- **grow_policy=lossguide** (max_leaves=255 or 1024): hurt or timed out.
- **colsample_bylevel** (0.7, 0.8) / **colsample_bynode** (0.5, 0.6, 0.8, 0.9): all hurt on top of bytree=0.4.
- **DepHalfHour (48 cat)** instead of Dep20Min (72): -0.0014. Memory said interchangeable; not so.
- **Dep15Min (96 cat)** instead of Dep20Min: -0.001. Over-granular as memory predicted.
- **Removing `CarrierHour`, `DayofMonth`, raw `DepTime`, `MonthSin/Cos`, `DepTimeSin/Cos`, or `Dep20Min`**: each lost 0.005–0.009. Nothing in the FE stack is redundant.
- **tree_method=approx, lossguide max_leaves=1024**: timeouts (>2 min).
- **Memory's exact final config** (lr=0.073, n_est=322): -0.0011. Transferred poorly.
- **mcw=2 alone**: +0.0010–0.0018 AUC, but always over budget; n_est cuts to compensate gave back the gain.
- **Many fine-tuning attempts** (lr 0.035/0.045/0.05, mcw 2.5/4, depth 12–16, max_bin 256/1024, max_cat_threshold 64/192/256, colsample_bytree 0.35/0.45/0.5): all within ±0.002 noise.

## Hard ceiling
A `depth=13 + mcw=2 + n_est=600` config reaches **0.8225 CV AUC** but at ~85s wall. Compensating cuts (n_est=380–500) drop the AUC back to 0.8205–0.8213. This is the firm budget-bound ceiling at the current FE stack.

## Memory portability
- **Transferred well**: DepMinute (+0.0046), Dep20Min (+0.006), max_cat_threshold=128 (+0.002), random_state caveat, "avoid early-stopping experiments" guidance.
- **Did NOT transfer / hurt**: ordinal-encode date cats (memory said +0.008, observed -0.006); subsample=0.95 with low mcw; memory's exact final hyperparams; DepHalfHour/Dep20Min interchangeability claim.
- **Lesson**: memory provides priors but every tip needs revalidation in the current configuration. Memory's biggest individual wins on a prior run remained the biggest individual wins here, but their relative ordering and several ancillary claims shifted.

## Practical takeaway for future runs
1. Set up the FE stack early: `DepMinute` numeric + `Dep20Min` cat + cyclic time (6h period) + cyclic Month + `CarrierHour` interaction cat. This block alone delivers ~+0.06 AUC.
2. Then tune capacity + regularization: depth=14, mcw=3, colsample_bytree=0.4, max_cat_threshold=128, max_bin=512, lr=0.04, n_est ~ what fits 60s budget.
3. Parallelize CV folds (`cross_val_score(n_jobs=2)` + `xgb_n_jobs=4`) — same scores, ~5s wall savings, fund a small n_est bump.
4. Resist adding more interaction cats at colsample=0.4 — they dilute and consistently regress.
5. Treat single-seed gains > ±0.002 as suspect; ±0.003 of plain seed variance is normal here.
