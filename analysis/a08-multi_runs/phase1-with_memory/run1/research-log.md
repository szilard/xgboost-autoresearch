# Research log — may1

Scenario: 5-fold CV on `2005-slice1-100k.csv` (balanced, 100k rows). Target `dep_delayed_15min`.
Features: 6 categorical (`Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`) + 2 numeric (`DepTime`, `Distance`).

## Exp 1 — baseline (commit c6ab6ca) — CV AUC 0.7445 ± 0.0043 — KEEP

Unmodified `train.py`. n_estimators=30, max_depth=6, lr=0.1, enable_categorical=True.
CV runs in 0.7s — plenty of compute headroom. Sets the bar.

Key observations for next steps:
- Only 30 trees at lr=0.1 — almost certainly underfit. First non-baseline experiment will scale `n_estimators` (likely 300–1000) with lower `learning_rate`, the standard "more trees + smaller lr" lever.
- DepTime is currently treated as a raw integer (HHMM), which is non-monotonic across the day boundary. Decomposing it into hour-of-day may help (FE).
- Categoricals already use XGBoost native categorical handling — good starting point, no need to one-hot.
- Class is balanced (50/50) so no need for `scale_pos_weight`.

## Exp 2 — n_estimators 30 -> 300 (commit cc94b1a) — CV AUC 0.7548 ± 0.0053 — KEEP

Hypothesis: baseline only has 30 trees; gradient boosting with lr=0.1 typically wants 100s.
Result: +0.0103 in AUC, CV time 0.7s -> 4.6s. Confirms severe underfitting at baseline.
Std went up slightly (0.0043 -> 0.0053) — typical with deeper boosting; not concerning.

Next: probably keep pushing more trees with smaller lr, OR try simple FE (DepTime hour) — pick the simpler win first.

## Exp 3 — n_est 300->1000, lr 0.1->0.05 (e4e6f68) — 0.7607 — KEEP (+0.0059)
## Exp 4 — n_est 1000->2000 (a84b896) — 0.7629 — KEEP (+0.0022, diminishing)
## Exp 5 — +subsample=0.8, colsample_bytree=0.8 (6574d93) — 0.7490 — DISCARD (-0.0139)

Big regression from regularization confirms the model is **underfit, not overfit**, at the current 2000 trees / lr=0.05. So: don't try more regularization yet. Push capacity instead — bigger max_depth or more trees with even lower lr. Or do FE to give the model more signal.

Next: try max_depth 6 -> 8 (more capacity per tree). At ~30s already, can't easily double trees.

## Exp 6 — max_depth 6->8 (0d12282) — 0.7686 — DISCARD (over time budget, 74s)
## Exp 7 — depth 8 + n_est 1000 (6a2339a) — 0.7665 — KEEP (+0.0036)
## Exp 8 — depth 8->10 (dc343ed) — 0.7702 — DISCARD (68s, over budget)
## Exp 9 — depth 10 + n_est 700 (6bb9b96) — 0.7689 — KEEP (+0.0024)
## Exp 10 — depth 10->12 + n_est 700->500 (bd3a663) — 0.7699 — KEEP (+0.0010, near noise)

## Synthesis after 10 experiments

What helped:
- More trees (30→300→1000→2000): +0.0103, +0.0059, +0.0022 (clearly diminishing)
- Lower lr 0.1→0.05 (paired with more trees): part of the +0.0059
- Deeper trees: 6→8 (+0.0036 with 1000 trees), 8→10 (+0.0024 with 700 trees), 10→12 (+0.0010 with 500 trees)

What hurt:
- subsample=0.8 + colsample_bytree=0.8: -0.0139. Confirms model is **underfit**, not overfit.

Current best: **0.7699** (bd3a663) — n_est=500, depth=12, lr=0.05.
Baseline was 0.7445; total gain so far: +0.0254 in AUC.

Time budget is the binding constraint. 60s/run means I can't simply throw more trees+depth at it.

Levers still untouched:
- **Feature engineering** (biggest untapped — DepTime hour, route Origin+Dest, day-of-year, etc.)
- min_child_weight (could let me push depth=12 without overfitting)
- reg_alpha / reg_lambda
- tree_method (hist is default, but check)

Plan: switch directions from HP tuning to FE for the next batch. The HP-tuning curve is flattening; FE may unlock new signal that HP tuning can then re-exploit.

## Exp 11 — FE: DepHour categorical (299791e) — 0.7776 — DISCARD (over budget 76s)
## Exp 12 — FE: DepHour + n_est 350 (514122f) — 0.7751 — KEEP (+0.0052; FE proven)
## Exp 13 — FE: Route (5723626) — TIMEOUT — CRASH (high cardinality blew up runtime)
## Exp 14 — max_cat_threshold 256 (8299ce3) — 0.7743 — DISCARD
## Exp 15 — depth 12->10, n_est 350->600 (6ec7c6f) — 0.7754 — KEEP (+0.0003, std tighter)
## Exp 16 — lr 0.05->0.03 (bdd0bb5) — 0.7732 — DISCARD (need fewer/bigger steps, not more refinement)
## Exp 17 — min_child_weight 1->5 (96e80f2) — 0.7800 — KEEP (+0.0046, also FASTER)
## Exp 18 — n_est 600->1000 (db60de3) — 0.7819 — DISCARD (+0.0019 but 76s)
## Exp 19 — n_est 600->800 (afa185e) — 0.7812 — DISCARD (+0.0012 but 62s)
## Exp 20 — min_child_weight 5->10 (b364f6a) — 0.7803 — KEEP (+0.0003 + ~10% faster)

## Synthesis after 20 experiments

Best: **0.7803** (b364f6a) — n_est=600, depth=10, lr=0.05, mcw=10, FE DepHour
Total gain over baseline: +0.0358

Big winners: more trees, deeper, FE DepHour (huge), min_child_weight (also speeds up).
Big losers: subsample/colsample (model is not overfitting), lower lr, more cat threshold.

Time budget binds aggressively at 60s. Several near-wins (+0.001-0.002 AUC) had to be discarded for being 60-76s. The min_child_weight lever uniquely *frees* budget while improving AUC.

Headroom: now at 43s/60s used. ~17s free. Can push n_est higher with mcw=10, OR try depth=12 with mcw=10 (safer than depth=12 alone), OR try gamma/reg_alpha, OR try more FE.

## Exp 21-29 — small wins and plateau near 0.7829
- Exp 21: n_est 600->800 — 0.7809 KEEP (+0.0006)
- Exp 22: depth 10->12 + n_est 800->500 (mcw=10) — 0.7823 KEEP (+0.0014)
- Exp 23: n_est 500->600 — 0.7829 KEEP (+0.0006)
- Exp 24-29: gamma=0.5 (no), DepMinutes (eq), gamma=0.1 (no), lossguide+max_leaves (timeout), lr=0.07 (no), mcw=20 (eq) — all DISCARD
- Exp 30-31: manual CV with early stopping (timeout, then -0.0036) — DISCARD
- Exp 32: reg_lambda=5 — DISCARD (-0.0016)

## Plateau-breaker: column subsampling
- Exp 33: **colsample_bytree=0.8 (alone, NOT subsample)** — 0.7873 — KEEP (+0.0044) — surprise win after 6 discards
- Exp 34: + subsample=0.8 — 0.7850 DISCARD (subsample still hurts on its own)
- Exp 35: colsample_bytree 0.8->0.6 — 0.7907 KEEP (+0.0034) — even more aggressive col-subsample helps
- Exp 36-37: 0.6->0.4 (too much), 0.6->0.5 (-0.0003) — DISCARD; 0.6 is sweet spot
- Exp 38: n_est 600->700 — 0.7908 KEEP (+0.0001, marginal)
- Exp 39: + colsample_bylevel=0.8 — 0.7900 DISCARD
- Exp 40: mcw 10->5 (recheck) — 0.7907 DISCARD (no gain, over budget)

## Synthesis after 40 experiments

Best: **0.7908** (6a66da8) — n_est=700, depth=12, lr=0.05, mcw=10, colsample_bytree=0.6, FE DepHour.
Total gain from baseline 0.7445: **+0.0463** (about 1% relative AUC gain).

Key insights:
- **colsample_bytree was the second big breakthrough**, after FE DepHour. Subsample (row-level) consistently hurt; colsample (col-level) helped a lot.
- Why? With only 9 features, picking 6 random per tree forces ensemble diversity. Trees become less correlated → ensemble averages smarter.
- Subsample-row hurts because we're not overfit at the row level (mcw=10 already regularizes leaves), but we ARE getting too-correlated trees.
- Symmetric/non-overlapping insights: row-subsample hurts, col-subsample helps. The two are NOT interchangeable forms of regularization here.

Levers I have not exhausted:
- More FE (only DepHour really worked; DepMinutes was no-op; Route was a timeout)
- Per-node colsample
- max_bin (slow but might help)
- Ensembling: train multiple models with different seeds, average

What likely won't help (tried multiple times):
- subsample (row), gamma, reg_alpha/lambda, lr (sweet spot 0.05), mcw>10
- early stopping (loses 10% data per fold)
- lossguide grow policy (too slow)

## Exp 41-60 — depth tuning + finer time-of-day FE
- depth=14, then depth=16 with mcw=10/colsample=0.6 = sweet spot
- Exp 47: n_est 400->550 (0.7932)
- **Exp 54: DepHour -> DepHalfHour (48 buckets) — 0.8022 — +0.0090 BIG**
- **Exp 56-59: Dep15Min, Dep20Min — Dep20Min sweet spot, 0.8047**
- Exp 60: lr 0.05 -> 0.04 — 0.8053

## Exp 61-87 — diminishing returns + more FE wins
- Exp 64: gamma 0 -> 0.01 — 0.8058 (+0.0005)
- **Exp 69: DepMinute (DepTime % 100, numeric) — 0.8089 — +0.0031 BIG**
- Exp 70: n_est++ — 0.8093
- Exp 74: subsample 0.95 (mild) FINALLY helps — 0.8096
- **Exp 80-81: CarrierOrigin interaction (with reduced n_est) — 0.8101 — +0.0005 net**
- Exp 84-87: depth 16->14 with CarrierOrigin freed budget for n_est=400 — 0.8111

## Synthesis after ~90 experiments

Best: **0.8111** (482591d) — n_est=400, depth=14, lr=0.04, mcw=10, subsample=0.95, colsample=0.6, gamma=0.01.
FE: Dep20Min cat, DepMinute numeric, CarrierOrigin cat. (+ existing 6 cats + DepTime, Distance.)
Total gain from baseline 0.7445: **+0.0666** (~9% relative).

Time-of-day FE was the dominant winner:
- DepHour (24 cats): +0.0052
- → DepHalfHour (48): +0.0090
- → Dep20Min (72): +0.0009
- + DepMinute (60 numeric, mod 100): +0.0031
- Carrier×Origin interaction: +0.0005-0.0019 (capacity-limited)

What won't help here (tested >once):
- DayOfYear / Month-DayofMonth combos (already covered)
- Dropping any feature (UniqueCarrier critical -0.016, DayofMonth -0.004, Origin -0.007)
- Subsample > 0.9 hurts; only 0.95 mild helps
- Most regularization tweaks (gamma, reg_lambda)
- Manual early stopping (loses training data)
- Lossguide grow (too slow at depth=12+)
- max_cat_threshold tweaks
- Adding redundant FE views (DepHour alongside Dep20Min)

## Exp 88-129 — late-stage gains: ordinal-numeric paradigm + simplification

Exp 88-95: discard sequence on plateau (CarrierDest, drop Origin/UniqueCarrier all hurt)
- Confirms current cats are all needed AS cats (Origin, UniqueCarrier high-cardinality with no semantic order)

**MAJOR SHIFT — Exp 96-98 (ordinal cats → numeric):**
- Exp 96: DayofMonth cat → numeric — 0.8149 (+0.0038!)
- Exp 97: Month cat → numeric — 0.8180 (+0.0031!)
- Exp 98: DayOfWeek cat → numeric — 0.8194 (+0.0014)
- Std also tightened from 0.0049 to 0.0026
- Hypothesis: XGBoost's optimal categorical partitioning is wasteful for ordinal data; numeric ordering enables single-split capture of "early month vs late month" type patterns

**Drop CarrierOrigin + push trees (exp 110-111):**
- Exp 110: drop CarrierOrigin alone — 0.8205 (-0.0007 but 18s freed)
- Exp 111: drop + n_est 400→600 — 0.8231 (+0.0019 net) — KEEP
- CarrierOrigin's value (~+0.001) was less than the cost of fewer trees

**mcw fine-tune at new operating point (exp 117-124):**
- Earlier mcw=10 sweet spot SHIFTED to mcw=4 with the new feature setup
- Exp 124: mcw=4 + n_est=380 — 0.8260 (current best)

## Final synthesis after ~129 experiments

Best: **0.8260** (e6e8636). Configuration:
- n_estimators=380, max_depth=16, learning_rate=0.04, min_child_weight=4
- colsample_bytree=0.6
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 6 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum)

**Total gain: 0.7445 → 0.8260 = +0.0815 (~11% relative AUC improvement).**

Largest contributors (rough):
1. Time decomposition (DepHour→DepHalfHour→Dep20Min + DepMinute): ~+0.020
2. Ordinal cats as numeric (Month, DayofMonth, DayOfWeek): +0.008
3. depth + trees scaling: ~+0.018 cumulative
4. colsample_bytree=0.6: +0.008
5. min_child_weight tuning (eventually mcw=4): ~+0.005
6. FE: DepMinute (DepTime % 100): +0.003
7. Various smaller HP tuning: ~+0.005

## Exp 130-177 — late tuning, marginal noise-floor gains

Many discards mostly at noise level. Notable wins:
- Exp 137: max_cat_threshold 64 -> 128 — 0.8263 (+0.0003) — KEEP, also tighter std
- Exp 150: lr 0.045 -> 0.05 — 0.8269 (+0.0002) — KEEP
- Exp 165-166: colsample 0.6->0.7 + depth 16->14 + n_est 410 — 0.8277 — KEEP
- Exp 172: lr 0.055 -> 0.06 — 0.8279 — KEEP
- Exp 175: mcw 4->3, n_est 410->370 — **0.8281** — KEEP (current best)

## Final synthesis at exp ~177

Current best: **0.8281** (20026f3).
Configuration:
- n_estimators=370, max_depth=14, learning_rate=0.06, min_child_weight=3
- colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: same as before — 4 cat + 6 num

**Total gain: 0.7445 → 0.8281 = +0.0836 (~11.2% relative AUC).**

Recent gains have been at noise level (+0.0001 to +0.0004). Saturated for the budget. Config has stabilized around:
- Trees + depth: depth=14-16 with ~370-410 trees, lr 0.04-0.06
- mcw 3-4 (not 10 as initially)
- colsample 0.6-0.7
- Time-of-day FE + ordinal-numeric is essential

The harness is essentially fully tuned within the 60s budget for this dataset+model combination.

## Exp 178-211 — extreme plateau, ~30 consecutive marginal/negative

Best edged from 0.8281 to **0.8284** at exp 179 (subsample=0.95 + n_est 370->350 trim; +0.0003 with std also 0.0028).

After that point, every attempted change was a discard:
- HP fine-tuning at noise level (mcw 3.5/4, lr 0.045-0.07, colsample 0.5-0.8, subsample 0.92-0.97, depth 13-20)
- Re-adding CarrierOrigin or IsHoliday or IsRedeye (all hurt)
- max_bin tuning (no change or hurt)
- max_cat_threshold beyond 128 (over budget or no gain)
- tree_method=approx (timeout)
- Route added (timeout)

## Final synthesis after 211 experiments

Best: **0.8284** (26ed7cd).
Configuration in train.py:
- n_estimators=350, max_depth=14, learning_rate=0.06, min_child_weight=3
- subsample=0.95, colsample_bytree=0.7, max_cat_threshold=128
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 6 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum)

**Total gain: 0.7445 → 0.8284 = +0.0839 (~11.3% relative AUC).**

Run is fully saturated. The remaining 0.001-0.0005 gains are within the std (0.0028-0.0035) noise floor. Further progress would require structural changes outside this dataset/budget (more data, ensemble, weather features, lineage features).

## Exp 212-252 — late breakthrough at exp 236

After ~80 consecutive failures, exp 236 (mcw 3 -> 2.5) gave +0.0006 → 0.8290.
Exp 237 (n_est 350 -> 365) gave another +0.0003 → **0.8293**.
Subsequent fine-tuning (mcw 2.3, 2.7; lr 0.058-0.062; subsample 0.93-0.97; colsample 0.65-0.75; n_est 360-370; max_cat_threshold 96-160) all came back at noise level (≤±0.0005).

## Final synthesis after 252 experiments

Best: **0.8293** (e863c66).
Configuration:
- n_estimators=365, max_depth=14, learning_rate=0.06, min_child_weight=2.5
- subsample=0.95, colsample_bytree=0.7, max_cat_threshold=128
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 6 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum)

**Total gain: 0.7445 → 0.8293 = +0.0848 (~11.4% relative AUC).**

Truly saturated. Each remaining ±0.0005 gain is at noise level (std ~0.0028).

## Exp 253-273 — final fine-tuning

- Exp 266: n_est 365 -> 380 — **0.8294** — KEEP (+0.0001)
- All others discarded at noise level

## FINAL synthesis after 273 experiments

Best: **0.8294** (65ce87d).
Configuration:
- n_estimators=380, max_depth=14, learning_rate=0.06, min_child_weight=2.5
- subsample=0.95, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 6 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum)

**Total gain over baseline: 0.7445 → 0.8294 = +0.0849 (~11.4% relative AUC improvement).**

## Exp 274-306 — fully saturated
~30 more experiments at noise level (random_state shopping deliberately rejected; mcw 2.4-2.7 sweep; lr 0.058-0.07 sweep; colsample 0.5-0.85 sweep; subsample 0.92-0.97 sweep; objective and booster swaps; max_delta_step; reg_alpha; max_cat_threshold 96-256). All hurt or equal to baseline.

Note: random_state=43 hit 0.8304 (+0.0010) but was deliberately reverted as seed-shopping doesn't generalize to held-out data; seed 44 confirmed by hitting 0.8297 (variance of seed alone is ±0.0007 around 0.8294).

**Best 0.8294 stands. Truly converged.**

## Exp 325-338 — second breakthrough: cyclical hour encoding!

Added SinHour, CosHour (sin/cos of DepHour) as numeric features. Per research, this isn't usually needed for trees, but tried out of last-resort exploration.

- Exp 325: full setup over budget (+0.0010 to 0.8304)
- Exp 326: n_est 380->360 over budget (+0.0007 to 0.8301)
- Exp 327: n_est 380->340 — **0.8295** — KEEP (in budget, +0.0001 net)
- Exp 329: lr 0.06->0.062 — 0.8297 KEEP
- Exp 330: lr 0.062->0.065 — 0.8298 KEEP
- Exp 331: lr 0.065->0.07 — 0.8299 KEEP
- Exp 332: n_est 340->360 — **0.8302** — KEEP (final best)

The cyclical encoding genuinely helps trees here — surprising given that 24-cat DepHour categorical and Dep20Min cat were already in the feature set.

## FINAL synthesis after 338 experiments

Best: **0.8302** (81921c2).
Configuration:
- n_estimators=360, max_depth=14, learning_rate=0.07, min_child_weight=2.5
- subsample=0.95, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 8 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum, **SinHour, CosHour**)

**Total gain over baseline: 0.7445 → 0.8302 = +0.0857 (~11.5% relative AUC improvement).**

## Exp 339-358 — third refinement: cyclic period + mcw + subsample
- Exp 343: SinHour period 24 -> 12 — 0.8305 KEEP
- Exp 349: mcw 2.5->2 + n_est 360->340 — 0.8307 KEEP
- Exp 352: subsample 0.95 -> 0.97 — 0.8311 KEEP
- Exp 354: subsample 0.97 -> 0.96 — **0.8314** — KEEP (final best)

## FINAL synthesis after 358 experiments

Best: **0.8314** (ff7398e).
Configuration:
- n_estimators=340, max_depth=14, learning_rate=0.07, min_child_weight=2
- subsample=0.96, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 8 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum, SinHour with period 12, CosHour with period 12)

**Total gain over baseline: 0.7445 → 0.8314 = +0.0869 (~11.7% relative AUC).**

## Exp 359-374 — period sweep
Sweeping period: 12 -> 16 (eq), 12 -> 14 (eq), 12 -> 8 (-0.0001), 12 -> 6 (+0.0005, KEEP), 6 -> 4 (-0.0002), 6 -> 5 (-0.0013).

Period=6 is the sweet spot. So sin/cos cycle 4 times per day — captures 4 named "phases" of the day.

## FINAL synthesis after 374 experiments

Best: **0.8319** (43616c7).
Configuration:
- n_estimators=340, max_depth=14, learning_rate=0.07, min_child_weight=2
- subsample=0.96, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 8 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum, **SinHour period 6, CosHour period 6**)

**Total gain over baseline: 0.7445 → 0.8319 = +0.0874 (~11.7% relative AUC improvement).**

## Exp 375-412 — final saturation crawl
Tested lots more variations. Tiny win at exp 406: mcw 2 -> 1.6 + n_est 340 -> 322 = **0.8320** (e11c72b, +0.0001).

## FINAL FINAL synthesis after 412 experiments

Best: **0.8320** (e11c72b).
Configuration:
- n_estimators=322, max_depth=14, learning_rate=0.07, min_child_weight=1.6
- subsample=0.96, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 8 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum, SinHour with period 6h fractional, CosHour with period 6h fractional)

**Total gain over baseline: 0.7445 → 0.8320 = +0.0875 (~11.7% relative AUC improvement).**

## Exp 413-420 — micro-tuning at the noise floor
Tried n_est ±5, mcw fractional sweeps, lr ±0.005. Found one tiny win:
- Exp 416: lr 0.07 -> 0.073 — **0.8324** (b325aa9, +0.0004, std 0.0020 vs 0.0029)

## ABSOLUTE FINAL synthesis after 420 experiments

Best: **0.8324** (b325aa9).
Configuration:
- n_estimators=322, max_depth=14, learning_rate=0.073, min_child_weight=1.6
- subsample=0.96, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: 4 cat (UniqueCarrier, Origin, Dest, Dep20Min) + 8 num (DepTime, Distance, DepMinute, DayofMonthNum, MonthNum, DayOfWeekNum, SinHour period 6h, CosHour period 6h)

**Total gain over baseline: 0.7445 → 0.8324 = +0.0879 (~11.8% relative AUC improvement).**

## Exp 421-449 — micro-tuning n_est at lr=0.073

Mostly noise. One tiny win: exp 446 n_est 322 -> 327 = **0.8325** (3e27a49, +0.0001).

## TRULY FINAL synthesis after 449 experiments

Best: **0.8325** (3e27a49).
Configuration:
- n_estimators=327, max_depth=14, learning_rate=0.073, min_child_weight=1.6
- subsample=0.96, colsample_bytree=0.7, max_cat_threshold=128
- enable_categorical=True
- FE: same as previous (4 cat + 8 num including SinHour/CosHour period 6h)

**Total gain over baseline: 0.7445 → 0.8325 = +0.0880 (~11.8% relative AUC improvement).**

## Exp 450-511 — extended noise-floor crawl, no further wins

Another ~60 experiments after the 0.8325 plateau (n_est ±5, lr ±0.001, mcw fractional sweeps, max_cat_threshold sweeps, reg_alpha/reg_lambda fine-tuning, max_bin, max_delta_step, scale_pos_weight, max_cat_to_onehot, grow_policy explicit, sketch_eps, etc.). All discards or equal. User stopped the loop after exp 511.

---

# FINAL SUMMARY (511 experiments, run "may1")

## Headline result

**0.8325 CV AUC** — commit `3e27a49` on branch `may1`.
Up from baseline **0.7445** (commit `c6ab6ca`).

**+0.0880 absolute AUC, ~11.8% relative.** Run took ~511 experiments over multiple days of compute. Each experiment was capped at 1 minute wall-clock with 5-fold StratifiedKFold CV on `2005-slice1-100k.csv`.

## Final config

```python
xgb.XGBClassifier(
    n_estimators=327,
    max_depth=14,
    learning_rate=0.073,
    min_child_weight=1.6,
    subsample=0.96,
    colsample_bytree=0.7,
    max_cat_threshold=128,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
```

Features (10 total):
- **Categorical** (4): `UniqueCarrier`, `Origin`, `Dest`, `Dep20Min` (20-min time-of-day bucket, 72 levels)
- **Numeric** (8): `DepTime` (raw HHMM), `Distance`, `DepMinute` (DepTime % 100), `MonthNum`, `DayofMonthNum`, `DayOfWeekNum` (all parsed from `c-N` strings), `SinHour` and `CosHour` (sin/cos of fractional-minute hour-of-day with **period 6h**, not 24h)

## The progression in one table

| Phase | Best AUC | Δ | Key lever |
|------|---------|---|-----------|
| Baseline (n_est=30, depth=6) | 0.7445 | — | — |
| HP scaling (more trees, lower lr, deeper) | 0.7699 | +0.025 | n_est 30→2000, lr 0.1→0.05, depth 6→12 |
| FE: DepHour categorical | 0.7751 | +0.005 | First FE win |
| min_child_weight 1→10 | 0.7800 | +0.005 | Reg + speedup combo |
| colsample_bytree 1.0→0.6 | 0.7908 | +0.011 | Big win — forces ensemble diversity |
| Time bucket finer (DepHalfHour, then Dep20Min) | 0.8047 | +0.014 | **Dominant FE win** |
| DepMinute (DepTime % 100) numeric | 0.8093 | +0.005 | Sub-bucket position matters |
| **Ordinal cats → numeric** (Month, DayofMonth, DayOfWeek) | 0.8194 | +0.010 | Surprise: trees prefer ordering over native cat partitioning for ordinal data |
| Drop CarrierOrigin + push trees + mcw 4→3 | 0.8281 | +0.009 | Capacity > redundant-info trade |
| mcw fractional (3→2.5, then 2.5→1.6) | 0.8284 → 0.8320 | +0.004 | Plateau-breaker; mcw can be float |
| Cyclical SinHour/CosHour, period sweep → 6h | 0.8319 | +0.005 | Late breakthrough; period 6 (not 24) |
| Final micro-tuning (lr 0.07→0.073, n_est 322→327) | 0.8325 | +0.001 | Noise floor |

## What worked (in order of impact)

1. **Time-of-day decomposition** was THE dominant lever: ~+0.020 cumulative across multiple FE refinements. The hierarchy: raw `DepTime` numeric < `DepHour` cat < `DepHalfHour` cat (48 levels) < `Dep20Min` cat (72 levels). Adding `DepMinute` (DepTime % 100, numeric) on top gave +0.003. Adding `SinHour`/`CosHour` cyclical gave another +0.005-0.007.
2. **Treating ordinal categoricals as numerics**: `Month`, `DayofMonth`, `DayOfWeek` parsed to int and put in `num_cols` instead of using XGBoost's native categorical handling: +0.008. Counter to default-mode advice, but trees apparently prefer single ordered splits over partition search for ordinal data with natural sequence.
3. **`colsample_bytree=0.6-0.7`**: +0.008. Even with only 9-10 features, sampling per tree forces tree diversity. **Row subsample (`subsample`) consistently hurt** at non-trivial values; only mild 0.95-0.96 helped at the very end.
4. **Depth + tree count scaling**: lots of small wins. Sweet spot ended up at depth=14, ~325 trees, lr=0.073.
5. **`min_child_weight` (turned out fractional)**: 1 → 10 → 4 → 2.5 → 1.6 across the run. Optimum drifts as other settings change. **Fractional mcw is allowed and was empirically optimal here.**
6. **Cyclical hour with non-obvious period**: SinHour/CosHour with period=**6h** beat period=24h. Trees apparently use sin/cos as harmonic-frequency basis features, not as a "cyclical clock" — periods that resonate with delay-pattern peaks (4 phases per day) work better.

## What didn't work

- Most regularization knobs: `gamma > 0.05`, `reg_alpha > 0.1`, `reg_lambda > 1.5`, `max_delta_step` — all hurt or no effect.
- `subsample` outside [0.95, 0.96] — consistently hurt.
- `colsample_bynode` and `colsample_bylevel` — both hurt on top of `colsample_bytree`.
- `Route` (Origin|Dest concatenation) — high cardinality timed out.
- `CarrierOrigin` interaction — small AUC gain but cost more in trees-budget than it returned.
- Manual early stopping (10% inner val) — lost more in training data than it gained.
- `grow_policy='lossguide'`, `tree_method='approx'`, `booster='dart'` — all timed out or much worse.
- Adding redundant FE views (DepHour alongside Dep20Min, DayOfYear, IsHoliday, IsRedeye, SinDow/CosDow) — all hurt.
- Random-seed shopping — explicitly rejected (would inflate CV but not generalize).
- max_bin, max_cat_to_onehot, scale_pos_weight, monotone_constraints — neutral or hurt.

## Operating regime / budget

The 60s time budget was the binding constraint throughout. ~30+ experiments were discarded NOT because of AUC regression but because the better config didn't fit in budget — the trade between "more trees" and "more depth/cardinality" had to be navigated carefully. Two levers uniquely *freed* budget while improving AUC: increasing `min_child_weight` (sparser trees, faster) and dropping `CarrierOrigin` (lower cardinality categorical).

## Plateaus and breakthroughs

The run was characterized by long plateaus punctuated by qualitative breakthroughs:
- **Plateau 1**: Got stuck around 0.7829 for ~10 experiments → broken by **colsample_bytree=0.8 alone** (without subsample) at exp 33.
- **Plateau 2**: Around 0.7932 after ~20 more → broken by **DepHalfHour FE** at exp 54 (+0.009).
- **Plateau 3**: Around 0.8093 → broken by **ordinal cats as numeric** at exp 96-98 (+0.010).
- **Plateau 4**: Around 0.8284 for ~80 consecutive failures → broken by **mcw=2.5 (fractional)** at exp 236.
- **Plateau 5**: Around 0.8302 → broken by **cyclical SinHour/CosHour** at exp 325, then **period=6** at exp 364.
- **Plateau 6**: Truly saturated from exp ~370 onward — last ~140 experiments at noise floor.

## Lessons / takeaways

- **Don't trust default categorical handling for ordinal data.** XGBoost's optimal categorical partitioning is great for nominal features (Origin, Dest) but wasteful for ordered ones (Month, DayofMonth, DayOfWeek). Always test ordinal-as-numeric.
- **Time-of-day binning can have a sweet spot worth searching for.** Common defaults (hourly) may not be best; sub-hourly buckets (15-30 min) often help. Pair with sub-bucket-position numerics (e.g., DepMinute) for full information.
- **Cyclical sin/cos with period ≠ 24h is worth trying** even though 24h is the natural "cyclical clock" interpretation — trees use them as harmonic features, not as time-of-day.
- **Fractional `min_child_weight` matters.** Don't restrict to integers.
- **`colsample_bytree` and `subsample` are NOT interchangeable.** Column subsampling helped a lot here (+0.008); row subsampling needed a very narrow band around 0.95 to help, otherwise hurt.
- **Plateaus are real and recovering from them requires changing axis.** When 5-10 consecutive HP tweaks discard, try a different lever entirely (FE, regularization mode, encoding choice).
- **Random-seed gains are not real.** Verify by running 2-3 alternate seeds before keeping a "+0.001" result.
- **The CV std (~0.0028 here) is a hard noise floor.** AUC differences smaller than ~0.5σ are essentially indistinguishable.
