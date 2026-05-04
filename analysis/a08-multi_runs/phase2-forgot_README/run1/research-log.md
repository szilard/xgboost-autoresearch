# Research log — branch `may3` (started 2026-05-03)

Dataset: airline departure delays (binary classification, balanced 50k/50k via undersampling).
Train: `2005-slice1-100k.csv`. Eval: 5-fold stratified CV.
Features: 6 categorical (Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest) + 2 numeric (DepTime, Distance). Target: `dep_delayed_15min`.

## Exp 1 — baseline (commit 0841565)
- **CV AUC**: 0.7445 ± 0.0043
- **CV time**: 0.6s
- Hyperparams: n_estimators=30, max_depth=6, learning_rate=0.1, enable_categorical=True
- **Status**: keep (baseline anchor)
- Notes: Very fast (<1s). Lots of headroom on n_estimators given the budget of ~1 minute per experiment.

## Exp 2 — n_estimators 30 -> 500 (commit 2482599)
- **CV AUC**: 0.7563 ± 0.0059 (Δ = +0.0118)
- **CV time**: 7.1s
- **Status**: keep
- Hypothesis: baseline was grossly under-trained; bumping rounds at lr=0.1, depth=6 should help.
- Verified — big single-knob win. CV time still tiny vs 60s budget.
- Std went up slightly (0.0043 -> 0.0059) — variance increased, but mean rose much more.
- Next: explore whether even more trees help, or whether to start co-tuning learning_rate.

## Exp 3 — n_estimators 500 -> 1500 (commit a4b9ddb)
- **CV AUC**: 0.7590 ± 0.0048 (Δ vs exp2 = +0.0027)
- **CV time**: 21.7s
- **Status**: keep
- Hypothesis: still under-trained at 500; push further at fixed lr=0.1 to find saturation.
- Diminishing returns kicking in — gain is ~25% of the 30→500 jump.
- Next: try n_estimators=3000 to see if curve flattens, then pivot to learning_rate / depth tuning.

## Exp 4 — n_estimators 1500 -> 3000 (commit f337119)
- **CV AUC**: 0.7601 ± 0.0046 (Δ vs exp3 = +0.0011, < 1 std → noise-borderline)
- **CV time**: 43.2s; total run ≈ 61s (right at budget edge)
- **Status**: keep (rule: higher → keep) but flag as marginal
- Capacity-only knob is exhausted. No more `n_estimators` increases — they'd blow the time budget for tiny gains.
- Pivot: regularization (subsample, colsample_bytree, min_child_weight, gamma) and feature engineering. These don't add training time but can shift bias-variance favorably.
- Synthesis so far: tree count was the dominant lever; we've moved from 0.7445 → 0.7601 (+0.0156) purely from rounds. Next phase needs different ideas.

## Exp 5 — add subsample=0.8 + colsample_bytree=0.8 (commit c7ec3fb, REVERTED)
- **CV AUC**: 0.7443 ± 0.0038 (Δ = -0.0158 vs exp4) — huge regression, worse than baseline
- **Status**: discard, reset to f337119
- Surprise: classic stochastic regularization tanked the model. With only 8 features, colsample_bytree=0.8 ≈ 6-7 features/tree which seems fine; subsample=0.8 also routine. Yet combined effect at n_estimators=3000 + lr=0.1 was destructive.
- Hypothesis for the failure: with so few features and lr=0.1 (high), the model relies on consistent feature access early in boosting; randomly hiding features starves the early trees of signal. Or: at high-capacity 3000 trees with categorical splits, the stochastic noise pushes splits onto noisy patterns.
- Lesson: do not blindly add regularization at high capacity. Try one knob at a time, mild values, and re-test.
- Pivot: try max_depth tuning (orthogonal direction), or feature engineering (DepTime hour, time-of-day buckets).

## Exp 6 — max_depth 6 -> 4 (commit 2c2872d, REVERTED)
- **CV AUC**: 0.7501 ± 0.0045 (Δ = -0.010)
- **CV time**: 25.7s (depth-4 trees train ~40% faster, as expected)
- **Status**: discard, reset to f337119
- Lesson: depth=4 too shallow for this dataset — interactions matter. depth=6 stays. depth=8 is plausibly better but adds time risk; will explore via min_child_weight first as a no-time-cost regularization knob.

## Exp 7 — add min_child_weight=10 (commit 14feb9b)
- **CV AUC**: 0.7652 ± 0.0051 (Δ vs exp4 = +0.0051, ~1 std → real gain)
- **CV time**: 42.5s (no time cost vs exp4)
- **Status**: keep — new best
- Hypothesis confirmed: at 3000 trees + lr=0.1 + depth=6, the model overfits noisy splits with tiny leaves; requiring ≥10 samples/leaf removes those bad splits without losing capacity.
- Important contrast with exp5: subsample+colsample together tanked the model, but min_child_weight (deterministic regularization) is a clean win. So *the right kind* of regularization matters more than "more reg".
- Next: push min_child_weight further (20, 50?) to find optimum, then maybe gamma.

## Exp 8 — min_child_weight 10 -> 30 (commit aa0051c)
- **CV AUC**: 0.7718 ± 0.0048 (Δ +0.0066). keep.

## Exp 9 — min_child_weight 30 -> 100 (commit 8510814)  ← current best
- **CV AUC**: 0.7774 ± 0.0063 (Δ +0.0056). keep.
- Time also dropping (35s) since stronger leaf-size constraint = fewer splits explored.

## Exp 10 — min_child_weight 100 -> 300 (commit 94445f7, REVERTED)
- **CV AUC**: 0.7678 ± 0.0052 (Δ -0.0096). discard. Optimum is between 100 and 300; try 200 next.

## Synthesis @ exp10
- Total gain: 0.7445 → 0.7774 (+0.0329).
- Two big levers found: tree count, then min_child_weight. The model was wildly overfit by default.
- Knobs to try next, in order: bracket min_child_weight (200), then gamma, lower lr=0.05, single-knob stochastic reg (just subsample), feature engineering on DepTime/Distance.

## Compressed log of exp 11 onwards (long plateau exploration; full details in results.tsv)
- exp11–17: bracketing/regularization. min_child_weight=100 confirmed optimum; lr=0.05 marginal; gamma/reg_lambda/subsample mostly hurt.
- exp18: +DepHour gave **+0.0018** but pushed over 60s budget at n=3000. Discarded as over-budget.
- exp19–20: replacing DepTime with DepHour, and dropping DayofMonth — both worse.
- exp21: reg_alpha=1.0 gave marginal +0.0003. Kept.
- exp22–28: more parameter brackets — nothing meaningful.
- exp29: Route=Origin_Dest categorical worse + over-budget.
- **exp30: KEY BREAKTHROUGH**. Combined +DepHour with n_estimators 3500→2500. CV AUC 0.7801 (+0.0013 over previous best). Trade-off worked.
- **exp31: NEW BEST 0.7808**. min_child_weight 100→75 (less reg needed at lower capacity). Right at budget edge (~63s).
- exp32–37: more brackets at the new base. No further wins. Plateau.
- exp38: per-fold early-stopping CV refactor. Best_iter ≈ 1054 on 64k data, but 64k vs 80k per fold loses AUC.
- exp39: tried n=1500 in normal CV. Worse — at full-data scale we need ~2500.

## Final state (commit 1f6d496)
- **CV AUC: 0.7808 ± 0.0052**
- **Total improvement: +0.0363 over baseline**
- Config: n_estimators=2500, max_depth=6, lr=0.05, min_child_weight=75, reg_alpha=1.0, +DepHour cat feature
- Time: ~63s (right at budget edge)

## What worked vs what didn't (final)
- **Worked**: scaling n_estimators (capacity), min_child_weight (deterministic leaf-size reg), DepHour categorical (when paired with reduced trees), lr=0.05, reg_alpha=1.0
- **Didn't**: stochastic regularization (subsample/colsample, alone or together), gamma, reg_lambda, max_depth changes, dropping features, Route/MonthHour interactions, max_cat_threshold tuning, max_cat_to_onehot, lossguide, per-fold early stopping
- **Theory**: this dataset (small, low feature count) responds best to *deterministic* regularization (mcw, reg_alpha) and feature engineering that adds genuinely new categorical signal (DepHour). Stochastic tricks hurt because there are so few features that randomly hiding any of them starves split decisions.

## Final run summary

**Run window**: 2026-05-03 → 2026-05-04. Branch `may3`.
**Total experiments**: 43 (12 keeps, 31 discards/timeouts).
**Final best**: commit `1f6d496` (exp31) — **CV AUC 0.7808 ± 0.0052** (+0.0363 over baseline 0.7445).

### Kept progression (every change that stuck)
| # | Commit | AUC | Δ | Change |
|---|---|---|---|---|
| exp1 | 0841565 | 0.7445 | — | baseline (n=30, depth=6, lr=0.1) |
| exp2 | 2482599 | 0.7563 | +0.0118 | n_estimators 30→500 |
| exp3 | a4b9ddb | 0.7590 | +0.0027 | n_estimators 500→1500 |
| exp4 | f337119 | 0.7601 | +0.0011 | n_estimators 1500→3000 |
| exp7 | 14feb9b | 0.7652 | +0.0051 | add min_child_weight=10 |
| exp8 | aa0051c | 0.7718 | +0.0066 | min_child_weight 10→30 |
| exp9 | 8510814 | 0.7774 | +0.0056 | min_child_weight 30→100 |
| exp12 | 8e1dac4 | 0.7782 | +0.0008 | learning_rate 0.1→0.05 |
| exp21 | b77bcd1 | 0.7785 | +0.0003 | add reg_alpha=1.0 |
| exp27 | 7bca30c | 0.7788 | +0.0003 | n_estimators 3000→3500 |
| exp30 | 3f78881 | 0.7801 | +0.0013 | +DepHour cat & n_estimators 3500→2500 |
| **exp31** | **1f6d496** | **0.7808** | **+0.0007** | min_child_weight 100→75 |

### Two phases, two levers
1. **Capacity phase (exp 2–4, 7–9)**: contributed **+0.0329** of the total gain. Pure parameter tuning — more rounds + min_child_weight regularization on tiny noisy leaves.
2. **Polish phase (exp 12–31)**: contributed **+0.0034**. Diminishing-returns territory; mostly small wins from lr/reg_alpha/n_estimators tweaks, plus one feature-engineering breakthrough.

### The DepHour win
- Adding `DepHour = DepTime // 100` as a categorical feature was the only feature-engineering change that stuck.
- It only worked once paired with a **reduction** in `n_estimators` (3500→2500): the extra cat feature added ~40% per-tree training cost, blowing the time budget at full tree count. Net AUC gain came from the trade.
- *Why categorical DepHour and not numeric?* XGBoost can already split on the numeric `DepTime` continuously, so a numeric DepHour would be a monotonic transform — invisible to the splits. The *categorical* version exposes non-monotonic groupings (e.g., grouping late-evening + very-early-morning hours together) that continuous splits can't represent.

### Time budget pressure
The 60s/experiment budget was the binding constraint for the second half of the run. Once at the edge (~exp 15+), most knobs that *might* have helped (max_depth=8, more trees, max_cat_threshold=128, route encoding) ran over time and had to be discarded under the timeout rule, even when AUC moved positively. The exp30 trade — paying tree count to add DepHour — was the cleanest way to break out of this trap.

### What clearly didn't help
- **Stochastic regularization**: `subsample`/`colsample_bytree` alone or together, at any value. Only 8 features → randomly hiding columns starves splits.
- **`gamma`**: at gamma=1 most splits got pruned; the model collapsed.
- **`reg_lambda`**: no measurable effect at any value tried.
- **Depth changes**: depth=4 underfit, depth=8 over-budget, depth=5 underfit (with new feature set).
- **Feature removal**: dropping DayofMonth or DepTime both hurt. DepTime's minute precision matters.
- **Higher-cardinality features**: `Route = Origin_Dest` (~5–10k unique) and `MonthHour` (288) both regressed and added time.
- **Tree-growth alternative**: `grow_policy='lossguide'` matched AUC at higher cost.
- **Categorical knobs**: `max_cat_to_onehot`, `max_cat_threshold` (32 or 128) didn't move things.
- **Per-fold early stopping**: refactored CV with ES, but losing 16% of training data per fold to carve out an ES-val set cost more AUC than the ES gained. Useful diagnostic though: best_iteration on 64k data was ~1054, suggesting our 2500 trees on full 80k folds is near-optimal.

### Theory of the dataset
With 100k samples, 8 features, balanced 50/50 by undersampling, and a binary target driven by airline domain factors:
- Default XGBoost dramatically over-fits on tiny noisy leaves → **deterministic regularization** (`min_child_weight`, `reg_alpha`) is the dominant lever.
- Stochastic regularization is a *bad* fit because the feature set is too small for noise injection to be productive.
- Feature engineering opportunities are limited (most patterns are already capturable via Origin/Dest/Carrier/time interactions XGBoost finds itself); only the `DepHour` non-monotonic angle paid off.
- The CV variance band (~0.005) means anything below ~0.005 movement is noise. Many of the kept "marginal" wins (exp12, 21, 27, 31) are individually within noise — they're kept by the rule but in aggregate they may sum to less than they appear.
