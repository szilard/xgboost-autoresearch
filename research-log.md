# Research log — apr30 run

Branch: `apr30` (off no_FE / 14d61d4). 5-fold CV on `2005-slice1-100k.csv`. Goal: maximize CV AUC; no feature engineering.

## Top-line

| Stage | Best CV AUC | Δ from baseline |
|---|---|---|
| Baseline (commit 14d61d4) | 0.7445 | — |
| End of "first plateau" (mild colsample, depth=14) | 0.7872 | +0.0427 |
| Post-cs breakthrough (cs=0.55, depth=24) | 0.8028 | +0.0583 |
| Post-mcw rebalance (mcw=1, lr=0.04) | 0.8075 | +0.0630 |
| Post-onehot (small cats only) | 0.8180 | +0.0735 |
| **Final best (83f1e19)** | **0.8184** | **+0.0739** |

Final config: `depth=26, lr=0.045, mcw=1, colsample_bytree=0.55, max_bin=512, max_cat_to_onehot=32, ES_rounds=50, n_cap=8000` (~181 trees, CV time ~55s).

## The three breakthroughs

### Breakthrough #1 (exp #54) — colsample regime shift

For ~25 exps the model plateaued at ~0.7872 with `depth=14, mcw=7, colsample_bytree=0.85`. With only 8 features, `colsample_bytree=0.75/0.85/0.9` all rounded to 6-7 features per tree — coarse granularity hid the real curve. Pushing to 0.65 (5 feats) gave +0.0056; 0.55 (4 feats) was the sweet spot. Each tree becomes a deep specialist on a feature subset.

### Breakthrough #2 (exp #78-82) — lr+mcw rebalance at new regime

`lr=0.02, mcw=3` was no longer optimal. Raised lr=0.02→0.03→0.04, dropped mcw=3→2→1. Final: lr=0.04, mcw=1. Each step +0.0004 to +0.0023.

### Breakthrough #3 (exp #94) — `max_cat_to_onehot=32` for small-card cats

Default is 4. Setting to 32 forces small-card cats (Month=12, DayOfWeek=7, DayofMonth=31, UniqueCarrier=20) to use one-hot encoding internally instead of partition splits. **+0.0102 in a single exp.**

Critical detail: must NOT include Origin/Dest (282 each). Setting =300 (forcing them to one-hot too) gave -0.0128. Setting =16 (excluding UniqueCarrier) gave -0.0058. Setting =24 (excluding DayofMonth) gave -0.0071. All four small cats need one-hot; both high-card cats need partition.

Why partition wins for high-card cats: with 282 levels, optimal partition split groups categories by their gradient statistics, capturing meaningful structure. One-hot creates 282 sparse columns with only one non-zero each.
Why one-hot wins for small-card cats: with 7-31 levels, partition is over-flexible (combinatorial splits) and overfits; one-hot with `enable_categorical` is essentially a clean ordinal-ish split per category that the boosting can refine over many trees.

## Working axes

| Axis | Final value | Notes |
|---|---|---|
| max_depth | 26 | Was "saturated at 14" pre-breakthrough; deep trees only useful with low colsample |
| learning_rate | 0.045 | Was 0.015-0.02 pre-breakthrough; faster lr fine when ES kicks in early |
| min_child_weight | 1 (default) | Was 7-15 pre-breakthrough; reg moved to colsample axis |
| colsample_bytree | 0.55 (= 4/8 features) | THE key knob; 0.45 too few, 0.65 too many (with this depth) |
| max_bin | 512 | Up from 256 default; finer numeric splits help |
| max_cat_to_onehot | 32 | Up from 4 default; small cats (≤31 levels) want one-hot |
| early_stopping_rounds | 50 | Default-ish; ES typically fires ~150-300 trees in |

## Dead axes / negative results

- **Row subsampling** (any value): kills AUC consistently across all regimes.
- **gamma**, **reg_alpha**, **reg_lambda**: no meaningful gain in any regime.
- **max_cat_to_onehot ≠ 32** (different cardinality grouping): all worse.
- **max_cat_threshold>64**: slows down for no gain.
- **max_bin <512 or >512**: 512 sweet spot.
- **colsample_bynode**, **colsample_bylevel**: neutral or worse than colsample_bytree.
- **random_state changes**: deterministic, zero effect.
- **early_stopping_rounds changes (30, 75, 100)**: no gain.
- **eval_metric=logloss for ES**: worse for AUC.
- **grow_policy=lossguide**: much worse.
- **monotone_constraints**: relationships non-monotonic; -0.030 disaster.
- **tree_method=approx**: too slow.

## Theory of this dataset

- 100k rows balanced, 8 columns (6 cat: 4 small-card + 2 large-card; 2 num).
- Best model: many fast, deep trees, each looking at a random 4-of-8 features. Categorical features are split via:
  - **Partition** for Origin/Dest (282-card) — leverages cardinality structure
  - **One-hot** for Month/DOW/DayofMonth/UniqueCarrier (7-31 card) — avoids over-flexible partitions
- This combination gives a structurally simple but expressive model: random-feature-subspace + cat-aware splits.
- AUC ceiling on this hardware in the 1-min budget appears to be around 0.818-0.820.

## All experiments

See `results.tsv` — 110 experiments total: 33 keeps, 73 discards, 4 crashes/timeouts.

## Final summary

**Outcome.** Improved CV AUC from baseline **0.7445 → 0.8184** over 110 experiments — **+0.0739 absolute (~+9.9% relative).** Final config is structurally simple: standard XGBoost hyperparameters, no custom code, no feature engineering. The training script went from 13 lines of model config to about the same — most "work" was finding the right values, not writing new logic.

**The journey, in three regimes.**

| Regime | Best AUC | Defining choice |
|---|---|---|
| Naive | 0.7445 | Baseline `n_estimators=30, depth=6` (severely undertrained) |
| Conventional tuning | 0.7793 | Standard playbook: more trees, lower lr, deeper trees, ES, mild reg |
| With colsample subspace (depth=14→24) | 0.8075 | `colsample_bytree=0.55` (4-of-8 features) → "deep specialists" pattern |
| **+ asymmetric cat handling** | **0.8184** | `max_cat_to_onehot=32`: small cats one-hot, large cats partition |

**The biggest mistakes I made (worth remembering).**
1. **Coarse-binned conclusion.** Concluded "depth saturated at 14" and "colsample_bytree=0.85 is the sweet spot" after 25 experiments. Wrong on both — with only 8 features, all my colsample values 0.75/0.85/0.9 rounded to the same 6-7 features per tree, hiding the real signal at 0.55 (4 feats). Lesson: when a parameter is a fraction of a small integer, test the *bucket* directly, not arbitrary fractions.
2. **Time-budget bias toward incremental moves.** I stayed at depth=14 for ~25 exps because deeper was over budget. I needed to find the *budget-relaxing* knob (low colsample → fewer iters → time freed) before I could explore depth properly. Lesson: when blocked by budget, look for moves that *reduce* compute first, not just AUC-positive moves.
3. **One-hot vs partition was assumed away.** I treated `enable_categorical=True` as a binary on/off and never inspected what XGBoost was doing internally with each cat. The single biggest single-experiment jump (+0.0102) came from a parameter (`max_cat_to_onehot`) I didn't even read about until exp #94. Lesson: read the docs of the cat-handling subsystem when cat features carry most of the signal.

**What this dataset wants.**
- Many fast, deep trees (~180 trees at depth 26)
- Each tree sees only ~4 of 8 features (random feature subspace)
- Default `mcw=1` — colsample is doing the regularization work
- Asymmetric cat handling: one-hot for small-card (Month/DOW/DayofMonth/UniqueCarrier ≤ 31), partition for large-card (Origin/Dest 282)
- Every row matters — any row subsampling kills AUC
- AUC ceiling appears to be ~0.818-0.820 on this hardware in the 1-min budget; further gains would likely require either ensembling (compute-expensive) or feature engineering (out of scope).

**What I'd try next if budget were lifted.**
- Lower lr (0.025-0.03) with more iters — was +0.0003 over budget at 0.03.
- Seed-ensembling at the CV-fold level (re-CV with different fold seed and average), since model is otherwise deterministic.
- Stacking two configs (different colsample seeds) — this dataset's high variance across folds suggests ensemble diversity could help.

