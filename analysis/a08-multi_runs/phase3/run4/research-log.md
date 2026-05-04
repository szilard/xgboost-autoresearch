# Research log — may4

Dataset: airline departure-delay binary classification (`dep_delayed_15min`), 100k rows, 5-fold CV, AUC.
Baseline params: `n_estimators=30, max_depth=6, learning_rate=0.1`. Categoricals via `enable_categorical=True`. Features: Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest, DepTime, Distance.

## Background research (pre-exp1)
- Most XGBoost tuning guides agree the strongest first lever for an undertrained model is `n_estimators` (with lr held or slightly lowered).
- Baseline 30 trees is far below typical good ranges (200–500 for tabular binary classification).
- Airline-delay-specific feature engineering: **hour-of-day from DepTime** is repeatedly cited as a top engineered feature; **Origin–Dest pair** is also often impactful.
- Avoid count/frequency features here — data is undersampled by class so counts won't reflect true distribution (per program.md).

## Experiments

### exp0 — baseline (commit 7d0b368)
- CV AUC: **0.7445 ± 0.0043** — kept as baseline reference.

### exp1 — n_estimators 30→200 (commit 0f2d437) [exploration]
- Hypothesis: baseline is severely undertrained at 30 trees; +trees should help.
- CV AUC: **0.7538 ± 0.0050** (+0.0093). CV time 2.9s. **Keep.**
- Confirms the model is far from saturated; continue scaling trees before tuning lr/depth.

### exp2 — n_estimators 200→500 (commit 13ac8e2) [follow-up]
- CV AUC: **0.7563 ± 0.0059** (+0.0025). CV time 7.3s. **Keep.**
- Diminishing returns at fixed lr=0.1; variance up slightly. Time to try the classic lower-lr-more-trees pairing.

### exp3 — lr 0.1→0.05, n_estimators 500→1000 (commit 026e66d) [follow-up]
- Hypothesis: lower lr + more trees is the canonical recipe to extract more from boosting before overfitting.
- CV AUC: **0.7607 ± 0.0051** (+0.0044). CV time 14.3s. **Keep.**
- Confirms canonical pairing works here. Pivoting to feature engineering next — hyperparam ROI may diminish faster than FE.

### exp4 — add DepHour, DepMinute (commit 9fc6a5a) [exploration → FE]
- Hypothesis: DepTime is HHMM, so as a single numeric XGBoost can't cleanly model hour-of-day. Splitting into hour/minute exposes structure.
- CV AUC: **0.7695 ± 0.0054** (+0.0088). **Keep.** Strongest single-experiment win so far. Validates FE direction.

### exp5 — add OriginDest route categorical (commit 5f9be45) [exploration]
- Hypothesis: route-specific delay patterns. ~4200 unique pairs in 100k rows (avg 24/route).
- CV AUC: **0.7451 ± 0.0031** (-0.024). CV time 34.6s. **Discard.**
- Clearly hurts. Theory: at max_depth=6 with avg 24 rows/route, the model is wasting capacity on noisy route splits. If we ever try this again, need either deeper trees or much stronger regularization (or shrink route cardinality, e.g., only top-N routes).

### exp6 — max_depth 6→8 (commit 0de93d2) [follow-up]
- Hypothesis: with FE'd DepHour/DepMinute now present, deeper trees should capture more useful interactions. Standard tuning order.
- CV AUC: **0.7753 ± 0.0046** (+0.0058). CV time 27.2s. **Keep.**
- Solid win. Variance also slightly down. CV time getting larger though — keep an eye on budget.

### exp7 — subsample=0.8, colsample_bytree=0.8 (commit 1cd184e) [follow-up regularization]
- Hypothesis: stochastic regularization at deeper trees often helps generalization.
- CV AUC: **0.7656 ± 0.0044** (-0.0097). CV time 28.7s. **Discard.**
- Stochastic dropout actually hurt — likely the per-tree info loss outweighs regularization benefit at 100k rows. Variance was tighter but mean dropped a lot. Regularization via tree-structural constraints (e.g., min_child_weight, gamma) might fare better than stochastic.

### exp8 — max_depth 8→10 (commit c257718) [follow-up]
- Hypothesis: depth=8 was a big win; depth=10 may further capture interactions. Risk: time budget.
- CV AUC: **0.7806 ± 0.0037** (+0.0053). CV time 48.4s. **Keep.**
- Variance DOWN (0.0046→0.0037) despite more capacity — no overfitting yet on CV. Budget tight; can't push depth further.

### exp9 — min_child_weight 1→5 (commit 86470fb) [exploration regularization]
- Hypothesis: depth=10 has lots of capacity; min_child_weight is structural regularization (different from stochastic exp7). Should reduce overfitting without info-loss-per-tree that exp7 suffered from.
- CV AUC: **0.7841 ± 0.0045** (+0.0035). CV time 38.7s (down from 48.4s — bonus). **Keep.**
- Confirms: structural reg works, stochastic reg didn't. Faster too — extra room in budget.

### exp10 — min_child_weight 5→10 (commit d6166e9) [follow-up]
- Hypothesis: same direction, more aggressive structural reg.
- CV AUC: **0.7853 ± 0.0041** (+0.0012). CV time 35.3s. **Keep.** Diminishing returns; structural reg near optimal for this depth.

## Synthesis @ 10 experiments
Score 0.7445 → 0.7853 (+0.0408). Theory: dataset rewards capacity, punishes noisy high-card splits, prefers structural over stochastic reg. DepHour decomposition was the strongest single FE win. Wins to leverage further: more FE likely to outperform additional hyperparam tuning from here.

Research note (10-exp milestone): XGBoost `enable_categorical=True` uses optimal partitioning for high-cardinality (>max_cat_to_onehot, default 4) — so the OriginDest issue isn't cardinality alone, it's signal-to-noise per route at our sample size.

### exp11 — re-add OriginDest under strong config (commit 53ef5a8) [exploration]
- Hypothesis: prior failure was depth/reg issue; mcw=10 should prevent overfitting noisy routes.
- CV AUC: **0.7845 ± 0.0041** (-0.0008). CV time **71.5s — over budget**. **Discard.**
- OriginDest confirmed dead end (failed twice across very different configs). Stop trying it.

### exp12 — gamma 0→0.1 (commit 55fa445) [exploration regularization]
- Hypothesis: gamma trims weak splits; complementary to mcw.
- CV AUC: **0.7853 ± 0.0039** (=0). CV time 31.5s. **Discard** (equal AUC + extra param = simpler is better).

### Workflow note
For discarded experiments, I'm now committing the research-log entry separately AFTER the train.py reset, so log entries persist across discards.

### exp13 — drop DepTime (commit 999f845) [simplification]
- Hypothesis: DepHour + DepMinute reconstruct DepTime; the extra column is redundant.
- CV AUC: **0.7800 ± 0.0038** (-0.0053). CV time 34.4s. **Discard.**
- Surprising: DepTime is NOT redundant. Theory: as a single int, DepTime gives the model a finer continuous splitting axis (HHMM combined). Trees can find patterns at split boundaries that aren't naturally exposed by just hour+minute.

### exp14 — n_estimators 1000→1500 (commit 3762829) [follow-up]
- Hypothesis: at lr=0.05, mcw=10, depth=10, model may benefit from more boosting rounds.
- CV AUC: **0.7865 ± 0.0042** (+0.0012). CV time 49.7s. **Keep.**

### exp15 — add LogDistance feature (commit c36657a) [exploration FE]
- Hypothesis: complementary axis for skewed distance values.
- CV AUC: **0.7865 ± 0.0042** (=0). CV time 54.6s. **Discard** (equal AUC, more complex; trees handle Distance fine without the log).

### Plateau check
Last 5 experiments: keep, discard×2, keep, discard. We have ~3 of 5 discards but not 3 consecutive (exp14 broke the streak). Still: hyperparam space mostly explored, FE attempts narrow; need fresher angles.

### exp16 — add MonthNum/DayofMonthNum/DayOfWeekNum ordinals (commit f4525e8) [exploration FE]
- Hypothesis: ordinal versions of c-X cats expose monotonicity that categorical optimal-partition doesn't.
- CV AUC: **0.7865 ± 0.0044** (=0). **Discard** — categorical splits already capture what matters.

### exp17 — lr 0.05 → 0.04 (commit fae2b8a) [follow-up]
- Hypothesis: with 1500 trees, slightly lower lr may still squeeze more out without underfitting.
- CV AUC: **0.7866 ± 0.0032** (+0.0001, variance ↓). CV time 51.2s. **Keep.**

### exp18 — lr 0.04 → 0.03 (commit 4627b9a) [follow-up]
- Hypothesis: continue lr direction.
- CV AUC: **0.7866 ± 0.0044** (=0, variance ↑). **Discard.** lr=0.04 is the local optimum.

### Direction reset
Hyperparam space mostly explored.

### exp19 — grow_policy=lossguide max_leaves=512 [exploration]
- Crashed (>2.5min). **Discard.**

### exp20 — add MinutesOfDay [exploration FE]
- CV AUC **0.7866 ± 0.0032** (=0). **Discard.**

### exp21 — reg_lambda 1 → 5 [exploration]
- CV AUC **0.7848 ± 0.0040** (-0.0018). **Discard.** Too aggressive.

## Plateau check
Plateau ~0.7866 confirmed. Last 5 changes: 4 discards. Research suggests AirportDelayRate (forbidden) is the big lever. Otherwise: tweak existing parameters at margins, or try genuinely different architecture/encoding choices.

### exp22 — min_child_weight 10 → 20 (commit 61851ca) [exploration]
- Hypothesis: even stronger structural reg.
- CV AUC: **0.7873 ± 0.0044** (+0.0007). CV time 45.7s. **Keep.** Small win.

### exp23 — min_child_weight 20 → 40 (commit 8d80e45) [follow-up]
- Hypothesis: continue pushing structural reg.
- CV AUC: **0.7874 ± 0.0048** (+0.0001). CV time 40.4s (faster). **Keep.**

### exp24 — mcw 40 → 80 (commit b0df901) [follow-up]
- CV AUC: **0.7882 ± 0.0050** (+0.0008). CV time 34.0s. **Keep.** Surprise win — apparent plateau was mcw being too small.

### exp25 — mcw 80 → 160 (commit dfff153) [follow-up]
- CV AUC: **0.7826** (-0.0056). **Discard.** Underfit.

### exp26 — mcw 80 → 120 (commit a054b2f) [bisect]
- CV AUC: **0.7863** (-0.0019). **Discard.** mcw=80 confirmed optimum.

### exp27 — max_depth 10 → 11 (commit 6c4c99c) [follow-up]
- CV AUC: **0.7881** (~=). **Discard.**

### exp28 — n_estimators 1500 → 2000 (commit 9078a38) [follow-up]
- CV AUC: **0.7887** (+0.0005). CV time 44.9s. **Keep.**

### exp29 — n_estimators 2000 → 2500 (commit daa4e84) [follow-up]
- CV AUC: **0.7891** (+0.0004). CV time 54.7s. **Keep.**

### exp30 — n_estimators 2500 → 2800 (commit 33d9719) [follow-up]
- CV AUC: **0.7891** (=0). CV time **62.5s — over budget**. **Discard.**

### exp31 — lr 0.04 → 0.03 (commit 9d4d742) [follow-up]
- CV AUC: **0.7888** (-0.0003). **Discard.**

### exp32 — mcw 80 → 60 (commit 109857f) [bisect]
- CV AUC: **0.7888** (-0.0003). **Discard.**

### exp33 — mcw 80 → 100 (commit a37d7c9) [bisect]
- CV AUC: **0.7881** (-0.0010). **Discard.**

### exp34 — max_bin 256 → 512 (commit ffdf899) [exploration]
- CV AUC: **0.7890** (~=). **Discard** (equal + extra param).

### exp35 — lr 0.04 → 0.05 (commit 268e2d7) [bisect]
- CV AUC: **0.7882** (-0.0009). **Discard.** lr=0.04 optimum.

### exp36 — add CarrierHour interaction cat (~600 levels) (commit 0f90d8d) [exploration FE]
- Hypothesis: explicit Carrier×Hour captures peak congestion patterns specific to each carrier.
- CV AUC: **0.7896 ± 0.0050** (+0.0005). CV time 59.0s. **Keep.** New best, FE win after long discard streak.

### exp37 — also add DayHour interaction (commit 1234cc3) [follow-up FE]
- CV AUC: **0.7824** (-0.0072). CV time **64.8s over budget**. **Discard.** Surprising regression — adding 168-cat DayHour somehow disrupted the model. Maybe XGBoost categorical optimization couldn't handle two new high-cardinality features simultaneously.

## Synthesis @ 37 experiments
Score 0.7445 → **0.7896** (+0.0451 from baseline, ~6.0% relative). Final config: n_estimators=2500, max_depth=10, lr=0.04, min_child_weight=80, plus DepHour, DepMinute, and CarrierHour engineered features.

Top contributors (cumulative gain):
1. n_estimators 30→2500 + lr 0.1→0.04: ~0.022 (capacity)
2. max_depth 6→10: 0.011
3. DepHour/DepMinute FE: 0.0088
4. min_child_weight 1→80: 0.0051
5. CarrierHour FE: 0.0005

Dead ends: OriginDest cat, stochastic regularization, gamma, reg_lambda, log(Distance), ordinal cats, MinutesOfDay, lossguide, max_bin, max_depth=11, DayHour interaction.

## exp38–60: simplification + mcw rebisect (the phase change)

### exp38 — n_est 2500→2000 (with CarrierHour) — Keep, 0.7896 (simpler, equal AUC, faster)
### exp39 — n_est 1500 — Discard
### exp40 — depth 11 (mcw=80 regime) — Discard
### exp41 — also CarrierMonth — Discard
### exp42 — drop DepMinute — Discard (-0.0095, DepMinute is critical!)

### CarrierHour-induced mcw rebisect (the key insight!)
The plateau at mcw=80 was specific to the pre-CarrierHour feature set. With CarrierHour adding real signal, the model no longer needs heavy regularization to avoid overfitting to noise:

### exp43 — mcw 80→70 — Keep, 0.7902
### exp44 — mcw 70→60 — Keep, 0.7912
### exp45 — mcw 60→40 — Keep, 0.7930
### exp46 — mcw 40→30 — Discard (CV 62s over budget despite +0.0023)
### exp47 — mcw 40→35 — Keep, 0.7943
### exp48 — mcw=30, n_est=1500 — Keep, 0.7950 (fits budget)
### exp49 — mcw=20 — Keep, 0.7970
### exp50 — mcw=10 — Discard (CV 61.5s over despite +0.003)
### exp51 — mcw=10, n_est=1200 — Keep, 0.7995
### exp52 — mcw=5 — Keep, **0.8014** (broke 0.80!)
### exp53 — mcw=3 (n_est=1200) — Discard (CV 62s over)
### exp54 — mcw=3, n_est=1000 — Keep, 0.8035
### exp55 — mcw=2 — Keep, 0.8040
### exp56 — mcw=1 (n_est=1000) — Discard (over budget)
### exp57 — mcw=1, n_est=800 — Discard (worse)
### exp58 — mcw=1, n_est=900 — Keep, **0.8043** (current best)
### exp59 — depth=11 (mcw=1 regime) — Discard
### exp60 — lr=0.03 (mcw=1 regime) — Discard

## Synthesis @ 60 experiments
**Score 0.7445 → 0.8043 (+0.0598, ~8.0% relative AUC improvement)**

Final config:
- n_estimators=900, max_depth=10, learning_rate=0.04, min_child_weight=1
- Numeric features: DepTime, Distance, DepHour, DepMinute
- Categorical features: Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest, **CarrierHour** (interaction)

**Key insight**: CarrierHour was a phase-change unlock. Pre-CarrierHour the regularization optimum was mcw=80; post-CarrierHour it collapsed all the way to mcw=1 (default), and each step downward was a real gain. **Hyperparameters are not absolute — they depend on the feature set.** A "well-tuned" model with one feature set may need totally different hyperparameters with another.

Top contributors (cumulative):
1. CarrierHour FE + mcw rebisect: ~0.015
2. n_estimators + lr (capacity): ~0.022
3. max_depth 6→10: 0.011
4. DepHour/DepMinute FE: 0.0088

## Continued: exp61–67 (finer tuning)

### exp61 — also DayHour (mcw=1) — Discard (-0.0153, still hurts)
### exp62 — max_depth 10→9 (mcw=1) — Keep, **0.8044** (+0.0001 + much faster CV 42s)
### exp63 — depth 9→8 — Discard
### exp64 — n_est 900→1200 (depth=9) — Keep, **0.8056** (+0.0012)
### exp65 — n_est 1200→1500 — Discard (over budget)
### exp66 — n_est 1200→1300 — Keep, **0.8058** (+0.0002)
### exp67 — lr 0.04→0.035 — Keep, **0.8059** (+0.0001)
### exp68 — lr 0.035→0.03 — Discard (over budget)

## Final state @ exp67 (current best)
- **CV AUC: 0.8059 ± 0.0039** (baseline 0.7445, +0.0614 / +8.2% relative)
- Config: n_estimators=1300, max_depth=9, learning_rate=0.035, min_child_weight=1
- Features: numeric (DepTime, Distance, DepHour, DepMinute) + cat (Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest, **CarrierHour**)

Lessons:
1. **Phase changes matter**: a "well-tuned" model is well-tuned for its features. Adding CarrierHour invalidated the prior mcw=80 optimum.
2. **DepMinute is critical** despite seeming redundant with DepHour+DepTime.
3. **Some interactions help (CarrierHour), others actively hurt (DayHour, CarrierMonth, OriginDest)** — couldn't predict in advance.
4. **Stochastic regularization (subsample/colsample) hurts** on this dataset size (100k rows).
