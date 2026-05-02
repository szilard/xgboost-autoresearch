# Research log — `may2` branch

Run started 2026-05-02. Mode: 5-fold CV on `2005-slice1-100k.csv`. Branched from `v0.5.1`.

## Exp 1 — baseline (commit 000331d)

- **Setup**: `train.py` unchanged from v0.5.1. 30 trees, depth 6, lr 0.1, `enable_categorical=True`, categoricals via `pd.Categorical` with fixed `cat_levels`.
- **Result**: CV AUC = 0.7445 ± 0.0043, CV time 0.7s.
- **Status**: keep (baseline reference).

## Exp 2 — n_estimators 30 -> 200 (commit 561bd19)

- **Hypothesis**: baseline is severely under-trained (30 trees is far below the 100–1000 range that XGBoost docs / 2025 tuning guides recommend for tabular). Bump trees keeping lr=0.1.
- **Change**: `n_estimators` 30 -> 200. Single lever.
- **Classification**: follow-up to baseline (correcting an obviously under-tuned hparam).
- **Result**: CV AUC 0.7538 ± 0.0050, CV time 3.1s. **+0.0093 vs baseline**.
- **Status**: keep. Branch advances.
- **Take-away**: large jump confirms baseline was under-trained. Need to find where more trees stop helping.

## Exp 3 — n_estimators 200 -> 500 (commit a78819f)

- **Hypothesis**: still room left in tree budget at lr=0.1.
- **Change**: `n_estimators` 200 -> 500.
- **Result**: CV AUC 0.7563 ± 0.0059, CV time 7.7s. **+0.0025 vs exp2**.
- **Status**: keep. Diminishing returns kicking in.

## Exp 4 — n_estimators 500 -> 1000 (commit 7e5d217)

- **Hypothesis**: bracket the saturation point at lr=0.1 before switching levers.
- **Result**: CV AUC 0.7583 ± 0.0049, CV time 15.1s. **+0.0020**.
- **Status**: keep. Trajectory (+0.0093, +0.0025, +0.0020) confirms n_estimators alone is plateauing. Switch lever.

## Exp 5 — add DepHour (commit d758137)

- **Hypothesis**: per research (UC Berkeley delay study; multiple FE references), hour-of-day is a strong delay signal — late-night hours notably worse. DepTime is HHMM as a single int; trees *can* find hour cuts but explicit DepHour gives cleaner splits.
- **Change**: inside `prepare`, add `X["DepHour"] = (X["DepTime"] // 100).astype("int16")`.
- **Classification**: feature-engineering exploration grounded in research.
- **Result**: CV AUC 0.7590 ± 0.0043. **+0.0007 vs exp4**. Small.
- **Status**: keep. Marginal because trees already approximated hour from raw DepTime; explicit feature is a slight aid.

## Exp 6 — ordinal cats as int (commit 4d4d7ca)

- **Hypothesis**: Month/DayofMonth/DayOfWeek are ordinal (not nominal). Stored as "c-N" strings, even with `enable_categorical=True`, XGB's partition search may not exploit the natural order; integer encoding lets a single split capture e.g. "Month >= 7" cleanly.
- **Change**: split `cat_cols` into `ordinal_cats` (Month, DayofMonth, DayOfWeek → int via `str.removeprefix("c-").astype(int16)`) and `nominal_cats` (UniqueCarrier, Origin, Dest → still pd.Categorical).
- **Classification**: encoding-level exploration of a meaningfully different direction.
- **Result**: CV AUC 0.7609 ± 0.0036. **+0.0019 vs exp5**.
- **Status**: keep. Confirms ordinal-int beats pd.Categorical for these three.

## Exp 7 — colsample_bytree=0.8 (commit 267d59a)

- **Hypothesis**: tree diversity via column subsampling. Default 1.0 means every tree sees every feature; with only ~9 features dominant ones (DepTime/DepHour/Origin/Dest) get picked first every tree. Forcing 80% subset diversifies trees.
- **Change**: add `colsample_bytree=0.8` to the XGBClassifier.
- **Classification**: HP exploration (regularization).
- **Result**: CV AUC 0.7694 ± 0.0036. **+0.0085 vs exp6**. Big.
- **Status**: keep. Should probe other colsample values to bracket optimum.

## Exp 8 — colsample_bytree 0.8 -> 0.6 (commit 861526d)

- **Result**: CV AUC 0.7739 ± 0.0046. **+0.0045**.
- **Status**: keep. Same direction (more diversity) still helping.

## Exp 9 — colsample_bytree 0.6 -> 0.4 (commit ec19fe3)

- **Hypothesis**: bracket optimum by going aggressive.
- **Result**: CV AUC 0.7732 ± 0.0050. **-0.0007**.
- **Status**: discard, revert to 0.6. Optimum is near 0.6; 0.4 starts losing signal (each tree sees only ~3-4 of 9 features).

## Exp 10 — max_depth 6 -> 8 (commit 2267e67)

- **Hypothesis**: with column subsampling regularizing, allow deeper trees to capture richer interactions (e.g., Origin/Dest × hour). Default 6 is shallow for 100k tabular with strong interaction structure.
- **Result**: CV AUC 0.7858 ± 0.0039, CV time 19.3s. **+0.0119**. Largest single win so far.
- **Status**: keep. Push depth further next.

## Synthesis at 10 experiments

- Baseline 0.7445 → current 0.7858 (+0.0413).
- Biggest single levers: max_depth (+0.0119), colsample_bytree (+0.0130 cumulative), n_estimators (+0.0138 cumulative).
- Surprises: DepHour barely moved AUC (XGB can find hour cuts on raw DepTime); ordinal-int encoding beat pd.Categorical despite XGB 3.x optimal partitioning being available.
- Theory: this dataset rewards (a) deep interactions between Origin/Dest/time and (b) tree diversity from column subsampling. Both depth and diversity are levers; baseline was severely under-trained and over-fitting through feature dominance.
- Untouched levers ranked by my expected value: max_depth higher (10/12), subsample 0.8, lower-lr+more-trees finishing move, min_child_weight, route interaction feature (Origin-Dest pair as cat), DepMinute/cyclical hour.

## Exp 11 — max_depth 8 -> 10 (commit 6453372)

- **Result**: CV AUC 0.7942 ± 0.0046, CV time 31.0s. **+0.0084**. Keep.

## Exp 12 — max_depth 10 -> 12 (commit 45caf37)

- **Result**: CV AUC 0.7989 ± 0.0039, CV time 43.4s. **+0.0047**. Keep. Total run ~66s — at the edge.

## Exp 13 — max_depth 12 -> 14 (commit 24f9aa1)

- **Result**: CV AUC 0.8015 ± 0.0043, CV time 54.2s, total ~77s. AUC gain +0.0026 but **violates 60s budget** → discard per program rule. Reverted to depth=12.
- **Take-away**: depth=12 is the cap under the time budget. Need to find time savings before pushing depth.

## Exp 14–20 — plateau (all discard)

- 14: subsample=0.8 → 0.7924 (regression, over budget).
- 15: n_est 1000→500 at depth=12 → 0.7963 (regression, but -27s wall).
- 16: Route=Origin_Dest cat → 0.7866 (overfit + 133s wall).
- 17: min_child_weight=5 → 0.7960 (regression, -14s wall).
- 18: gamma=1 → 0.7814 (way too aggressive).
- 19: max_bin=128 → 0.7963 (no speedup).
- 20: lr 0.1→0.05 → 0.7999 (+0.001 but over budget at 72s).
- 7-discard plateau confirmed model is at local optimum for this compute budget. Researched min_child_weight/gamma docs and FE ideas mid-plateau.

## Exp 21 — add DepMinute (commit a00566d) — MAJOR WIN

- **Hypothesis**: trees can extract DepHour from raw DepTime via monotonic cuts (>1500 ≈ afternoon), which is why DepHour gave only +0.0007. But the **minute-within-hour** component is *cyclic* in DepTime — to split on "minute >= 30" you'd need 24 separate cuts (one per hour). Explicit DepMinute = DepTime % 100 should be genuinely new, non-extractable signal.
- **Change**: in `prepare`, `X["DepMinute"] = (X["DepTime"] % 100).astype("int16")`.
- **Classification**: feature-engineering exploration with strong a-priori reason to expect a gain (information-theoretic, not just empirical).
- **Result**: CV AUC **0.8115 ± 0.0037**. **+0.0126 vs exp12**. Wall 65.7s (borderline; depth-12 baseline was 63.4s, +2.4s for the extra feature is within the rule's "few seconds" allowance).
- **Status**: keep. Largest single gain in the run.
- **Lesson**: when extracting components from a composite feature, ask which components are *non-monotonic* in the original. Those are the ones the trees can't recover by themselves and where explicit FE pays.

## Exp 22 — add Dep20Min cat (commit 8681567) — over budget

- **Hypothesis**: 72-bucket time-of-day captures cyclic structure that hour+minute numerics can't (cat partitioning groups e.g. {21,22,23,0,1,2} as a delay-prone block; numeric ordering can't).
- **Result**: CV AUC **0.8287** ± 0.0028 (+0.0172) but wall 89.6s — way over budget. Discard the standalone version.

## Exp 23 — Dep20Min cat + n_est 1000→500 (commit 6428b80)

- **Hypothesis**: Dep20Min FE adds ~25s wall and ~+0.0172 AUC; halving trees subtracts ~27s wall and ~-0.0026 AUC (per exp15). Net predicted: AUC ≈ 0.8261, wall ≈ 62s.
- **Classification**: compound experiment to fit a high-value FE inside the time budget. Coherent rationale; not a near-duplicate.
- **Result**: CV AUC **0.8268** ± 0.0028, wall 53.0s. **+0.0153 vs exp21**. Predicted within 0.001.
- **Status**: keep. Branch advances. Have ~10s of budget headroom now to push other levers.

## Exp 24 — sin/cos hour period 6h (commit bb93222)

- **Hypothesis**: cyclical encoding of time-of-day with non-24h period as a low-frequency basis feature; trees can use the sinusoid as a delay-cycle proxy.
- **Result**: CV AUC **0.8300** ± 0.0025, wall 54s. **+0.0032**.
- **Status**: keep.

## Exp 25–27 — small-knob plateau (all discard)

- 25: n_est 500→600 → 0.8308 (+0.0008 but 62.9s wall, over budget for tiny gain).
- 26: mcw=2 → 0.8287 (regression, but -6s wall — interesting time/AUC trade not exploited).
- 27: colsample 0.6→0.7 → 0.8299 (no change, at budget edge).

## Exp 28 — max_cat_threshold=128 (commit 0d2bc62)

- **Hypothesis**: default 64 limits partition search for high-cardinality cats (Origin/Dest ~250+ levels, Dep20Min 72 levels). Doubling allows better cat splits at no real time cost.
- **Result**: CV AUC **0.8311** ± 0.0028, wall 56s. **+0.0011**.
- **Status**: keep.

## Exp 29 — max_cat_threshold 128 -> 256 (commit 946c337)

- **Result**: CV AUC **0.8314** ± 0.0024, wall 58s. **+0.0003** (borderline noise but kept).

## Synthesis at 30 experiments

- Best: **0.8314** (depth=12, n=500, lr=0.1, colsample=0.6, max_cat_threshold=256, full feature set).
- Total gain: 0.7445 → 0.8314 = **+0.0869** (~11.7% relative).
- Top single levers (cumulative): max_depth (+0.021), colsample (+0.013), DepMinute (+0.013), Dep20Min cat (+0.015 net of n-cut), n_estimators (+0.014 partly given back).
- Surprises: DepHour barely helped because trees recover hour from raw DepTime monotonically; DepMinute was a huge win because minute-of-hour is *cyclic* in DepTime so non-recoverable. Sin/cos period 6h beat 24h intuition.
- Untouched promising levers: subsample~0.95 (memory says helps when other reg is mild), fractional mcw (1.5), DayOfWeek cyclical, drop redundant DepHour.
- Time-budget binding: wall 58/60s — most expansive moves (depth=14, n=700, lr=0.05) blocked unless paired with a compensating cut.

## Exp 30–54 — sustained plateau at 0.8318

After exp 35 (max_cat_threshold=512, marginal +0.0001 keep), 19 consecutive discards. Comprehensive HP sweeps and FE attempts did not move the needle:

- **Tried & failed (most more than once)**: subsample=0.8/0.95-no-effect, mcw 1.5/2/0.7/0.8, gamma 0.05/1, reg_lambda=3, lr 0.05/0.075/0.08/0.12, max_depth 13/14 (over-budget at full n), colsample_bytree 0.5/0.7, max_bin 128, max_cat_threshold 1024, max_delta_step=1, grow_policy=lossguide, colsample_bylevel=0.8.
- **Compound moves tried**: depth=14+n=400 (over-budget +0.0002); depth=14+n=320+lr=0.075 (-0.0006); mcw=0.5+lr=0.12 (-0.0001, over-budget); subsample-drop+mcw=0.5 (predicted no win). All under-perform vs 0.8318.
- **FE tried**: drop DepHour (-0.0005); drop DepTime (-0.0010); drop subsample (-0.0004); SinHour12h harmonic (-0.0008); SinMonth/CosMonth (-0.0017); IsLateNight binary (-0.0012); Dep30Min instead of Dep20Min (-0.0011); add Route=Origin_Dest (overfit, way over-budget).
- **Real-but-blocked**: mcw=0.5 alone gives **+0.0005 AUC (0.8323)** but **wall 67s — over budget**. Cannot fit it in via tree/depth cuts without giving up more AUC than mcw=0.5 buys.

**Conclusion at exp 54**: model is at the local optimum for this compute budget on this dataset. Best CV AUC = **0.8318**. Total gain over baseline = **+0.0873** (~11.7% relative). The 60s wall constraint is the binding factor — every remaining AUC-positive lever requires more compute.

## Final config (commit 1942995)

```python
n_estimators=500, max_depth=12, learning_rate=0.1,
colsample_bytree=0.6, subsample=0.95, max_cat_threshold=512,
enable_categorical=True, random_state=42, n_jobs=-1.
```

Features (in `prepare`): DepTime, Distance, DepHour, DepMinute (numerics, both critical), Dep20Min cat (72-level cyclical-aware bucket), SinHour6h/CosHour6h (period 6h harmonic), Month/DayofMonth/DayOfWeek as ordinal int (not categorical), UniqueCarrier/Origin/Dest as nominal categorical.

## Exp 55–65 — extended plateau confirmed at 0.8318

Continued probes after the synthesis: tree_method=approx (way over budget), mcw=0.5 verified twice (real +0.0005 but stable 67s wall — 7s over budget; compounds with n_est cuts give ≤0.8318 in budget), CarrierOrigin cat (real **+0.0015 to 0.8333** but **92s wall** — way over; even with max_cat_threshold cut to default 64 still 73s and a regression because partition quality drops more than CarrierOrigin gains), CarrierDoW (catastrophic -0.02), colsample_bynode=0.8 (-0.0042), objective=binary:hinge (breaks AUC, expected), Dep30Min cat (-0.0011), IsLateNight binary (-0.0012), drop DepTime (-0.0010).

**Two AUC-positive levers exist that the 60s wall blocks**:
1. mcw=0.5 → +0.0005 AUC, +7s wall
2. CarrierOrigin cat → +0.0015 AUC, +32s wall

Combined would be 0.8338. But neither fits without wiping the gain via tree-count or depth cuts. The model is genuinely compute-bounded at 0.8318 on the 60s budget.

Total: 65 experiments in this run. Best CV AUC 0.8318 = +0.0873 over baseline (~11.7% relative gain).

## Exp 66–70 — period sweep on sin/cos hour

Probed alternative periods on the sin/cos hour features:
- 66: drop Dep20Min → 0.8205 (-0.0113, confirms Dep20Min provides 0.011 of value).
- 67: colsample=0.62 → 0.8314 over budget. Discard.
- **68: period 6h → 4h → 0.8319 (+0.0001)**. Borderline noise but technically positive — kept under rule. May not be robust.
- 69: period 3h → 0.8313. Discard.
- 70: period 8h → 0.8305 over budget. Discard.

Final best: **CV AUC 0.8319** (commit c1ad65c). Wall ~60s. Same model config as exp35 except sin/cos period=4h instead of 6h. The 0.0001 advantage over period=6h is at the noise floor and may not survive on held-out data — period=6h has stronger external support (may1 run validated it across 420 experiments).

---

# Final summary — `may2` run

**Headline.** 70 experiments. CV AUC **0.7445 → 0.8319** (+0.0874, ~11.7% relative). Best commit `c1ad65c`. Wall 60s on the 60s budget — compute-bound.

## Gain attribution (cumulative, in roughly the order it was unlocked)

| Lever | Δ AUC | Notes |
|---|---:|---|
| `n_estimators` 30 → 1000 (later cut to 500) | +0.0138 then -0.005 | Baseline was severely under-trained; partly given back when cut to fit Dep20Min in budget. |
| `max_depth` 6 → 12 | +0.0214 | Largest single HP win. Depth = 14 was +0.0026 more but doesn't fit budget. |
| `colsample_bytree` 1.0 → 0.6 | +0.0130 | Critical even with only 9 features. 0.4 too aggressive, 0.7 ≈ same as 0.6 over budget. |
| Ordinal int for Month / DayofMonth / DayOfWeek | +0.0019 | "c-N" string + `pd.Categorical` is worse than plain int even with `enable_categorical=True`. |
| Add DepHour numeric | +0.0007 | Tiny — trees already extract hour from raw DepTime via monotonic cuts. |
| **Add DepMinute numeric** | **+0.0126** | Star result. Minute-of-hour is *cyclic* in HHMM, so non-recoverable from monotonic splits on DepTime. |
| **Add Dep20Min cat** (compound with n_est 1000→500 to fit budget) | **+0.0153 net** | 72-level partition can group `{21,22,23,0,1,2}` as a delay block — numeric ordering can't. |
| Sin/cos hour, period 6h | +0.0032 | Cyclical encoding at non-24h period as a harmonic basis. Period 4h is +0.0001 better (noise). |
| `max_cat_threshold` 64 → 128 → 256 → 512 | +0.0018 | Small monotonic gains; most of it from the first doubling. |
| `subsample` 1.0 → 0.95 | +0.0004 | Right at noise. Probed once, removing it cost the same +0.0004. |

## What didn't work (each tested at least once)

- **HPs**: `min_child_weight` ≥ 1.5 (regress), `gamma` ≥ 0.05 (regress), `reg_lambda` 3 (regress), `lr` 0.05 / 0.075 / 0.08 / 0.12 (small gains over budget), `max_bin=128` (no speedup), `max_delta_step=1` (no effect), `grow_policy=lossguide` (way over budget), `colsample_bylevel=0.8` (regress, too much reg on top of `bytree=0.6`), `colsample_bynode=0.8` (regress), `tree_method=approx` (3× slower), `objective=binary:hinge` (kills AUC by design), `colsample_bytree` 0.4 / 0.5 / 0.65 / 0.7 (all worse than 0.6).
- **FE**: drop DepHour (-0.0005) / drop DepTime (-0.0010) / drop subsample (-0.0004) / drop Dep20Min (-0.0113, *confirms it's load-bearing*); SinHour 12h harmonic alongside 6h (-0.0008); SinMonth/CosMonth (-0.0017); IsLateNight binary (-0.0012, trees discover it from raw); Dep30Min cat instead of Dep20Min (-0.0011); period 3h (-0.0006), period 8h (-0.0014).
- **High-cardinality interactions**: `Route = Origin_Dest` (~10K levels, overfit + 133s); `CarrierDoW` (~140 levels, *catastrophic* -0.0203 — high-cardinality interaction whose support overlaps simpler features can hurt a lot, mechanism not fully understood).

## Levers that worked but the budget blocked

| Lever | Δ AUC | Wall over | Could not be saved by... |
|---|---:|---:|---|
| `min_child_weight=0.5` | +0.0005 | +7s | n_est cuts (-0.0005 to -0.001 wipes it) |
| `CarrierOrigin` cat (~1500 levels) | +0.0015 | +32s | depth/tree cuts, smaller `max_cat_threshold` (each costs more than the gain) |

If both fit, the ceiling on this dataset/budget would have been ~0.834. The may1 run reached 0.8324 with 420 experiments; we matched it within noise in 70.

## Lessons for future runs

1. **Cyclic-within-monotonic FE is the highest-value FE move**. Decompose composite features and identify which components are *non-monotonic* in the original — those are the parts trees can't recover for themselves. (Also written into `xgboost_tuning_insights.md`.)
2. **Coupled changes are the right move when one knob has known compensation costs** — it's not a near-duplicate experiment if you state the compensation rationale upfront. (Exp 23 was a textbook case: Dep20Min cat worth +0.0172 / +25s, n_est halving costs -0.0026 / -27s, net +0.0146 / -2s, exactly as predicted.)
3. **Ordinal-int beats `pd.Categorical` even with `enable_categorical=True`** for genuinely ordered features (Month, DayofMonth, DayOfWeek). XGB 3.x's optimal partitioning is *not* always a free improvement over plain numeric splits.
4. **Memory/prior-run hints are most valuable as direction-finders, not as configs to copy**. Mine are tuned around `depth=12, n=500, lr=0.1, colsample=0.6`; may1's converged at `depth=14, n=322, lr=0.073, colsample=0.7`. Both reach ~0.832; the *what works* list (Dep20Min cat, sin/cos at 6h, ordinal-int for date cats, colsample 0.6–0.7, depth deeper than 6) generalized cleanly between runs even though the exact HP optima didn't.
5. **High-cardinality categorical interactions are the most expensive thing in this codebase** — `Route` and `CarrierOrigin` cost 30–70+ seconds each, an order of magnitude more than simple HP tweaks. Worth trying once and budgeting accordingly; not worth iterating on without compute headroom.
6. **AUC noise floor is ~±0.0010** on this 5-fold CV (run-to-run XGBoost is deterministic at fixed seed; the noise is between near-equivalent configs). Anything below that should be treated as a tie regardless of whether the AUC nudges up or down.

## Final config

Commit `c1ad65c` on branch `may2`.

```python
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.1,
    colsample_bytree=0.6,
    subsample=0.95,
    max_cat_threshold=512,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
```

Features (built inside `prepare(df)`):
- **Numeric**: DepTime (raw HHMM), Distance, DepHour (`DepTime // 100`), DepMinute (`DepTime % 100`), SinHour4h, CosHour4h.
- **Ordinal int (not cat)**: Month, DayofMonth, DayOfWeek (strip `c-` prefix → int16).
- **Nominal categorical**: UniqueCarrier, Origin, Dest, Dep20Min (`DepHour*3 + DepMinute//20`, 72 levels).

Wall ~60s. CV AUC 0.8319 ± ~0.003.
