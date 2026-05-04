# Research log — `may4` run

Tracking thinking, hypotheses, and observations for each experiment.
Each entry references the git commit and corresponding `results.tsv` row.

## Setup

- Branch: `may4` (from `7d0b368` on master)
- Date started: 2026-05-04
- Eval: 5-fold stratified CV on `2005-slice1-100k.csv`
- Target: `dep_delayed_15min` (binary classification, balanced 50/50 by undersampling)
- Features (raw): 6 categorical (`Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`) + 2 numeric (`DepTime`, `Distance`)
- Note: data is class-balanced via undersampling, so any count-based features can be misleading (warned in program.md).

---

## Exp 1 — baseline

**Commit:** `7d0b368`
**Hypothesis:** None — baseline reference point. `train.py` as-is: XGBoost, 30 trees, depth 6, lr 0.1, native categorical handling.
**Result:** CV AUC **0.7445 ± 0.0043**, CV time 0.6s. → `keep` (reference).
**Notes:** Plenty of headroom: only 30 trees, no regularization tuned, no feature engineering. Runs in <1s so we have lots of compute budget per experiment.

---

## Exp 2 — n_estimators 30 → 300

**Commit:** `f6ed2ec`
**Type:** exploration (most fundamental knob, baseline severely under-trained).
**Hypothesis:** With only 30 trees at lr=0.1, model is severely under-trained. 10× more trees should give a substantial AUC bump even before any tuning of regularization or features.
**Result:** CV AUC **0.7548 ± 0.0053**, CV time 4.5s. Δ = **+0.0103** → `keep`.
**Notes:** Confirms strong under-training. Still ~55s of budget left per run, can afford much more.

---

## Exp 3 — n_estimators 300 → 1000, lr 0.1 → 0.05

**Commit:** `0f8e43e`
**Type:** follow-up to exp2.
**Hypothesis:** Classic "more trees + lower lr" — should give marginal gains while baseline still under-fit.
**Result:** CV AUC **0.7607 ± 0.0051**, CV time 14.5s. Δ = **+0.0059** → `keep`.

---

## Exp 4 — n_estimators 1000 → 2000, lr 0.05 → 0.03

**Commit:** `dc090da`
**Type:** follow-up.
**Hypothesis:** One more push on this axis to test for saturation; if gain <0.002, pivot to features.
**Result:** CV AUC **0.7626 ± 0.0051**, CV time 28.8s. Δ = **+0.0019** → `keep`.
**Notes:** Saturating on this axis. Pivot to feature engineering. Note CV time now ~29s — room is shrinking.

## Synthesis after 4 exps

- AUC trajectory: 0.7445 → 0.7548 → 0.7607 → 0.7626. Each step still positive but shrinking.
- We've spent all our wins on `n_estimators` × `lr`. Untouched: `max_depth`, regularization (`min_child_weight`, `gamma`, `reg_*`), sampling (`subsample`, `colsample_bytree`), feature engineering, `tree_method='hist'`.
- Plan: pivot to feature engineering (DepHour from DepTime is the canonical move on this dataset). Then tackle regularization/sampling. Then revisit lr/trees with early stopping if needed.

---

## Compressed log (exps 5-26)

- **exp5 KEEP**: +Hour categorical (DepTime//100) → 0.7728 (+0.0102). Big FE win.
- **exp6 DISCARD**: +Route Origin_Dest cat (TIMEOUT, ~5000 cats too slow at 2000 trees).
- **exp7 DISCARD**: subsample=0.8 + colsample_bytree=0.8 → 0.7643. Stochastic sampling hurts in this regime.
- **exp8 DISCARD**: drop DepTime → 0.7596. Minute-level info matters even with Hour.
- **exp9 DISCARD**: max_depth 6→8 (TIMEOUT).
- **exp10-12 KEEP**: mcw 1→5→20→50 monotonically up to 0.7792. mcw=50 peak.
- **exp13 DISCARD**: mcw 50→100 → 0.7781 past peak.
- **exp14 KEEP**: max_depth 6→7 → 0.7812.
- **exp15 DISCARD**: gamma=1.0 → 0.7648. Way too aggressive.
- **exp16-17 DISCARD**: reg_lambda=5 (-0.0004), subsample=0.9 (-0.0008).
- **exp18 DISCARD**: Hour×DOW concat → 0.7711. XGB learns this implicitly; explicit hurts.
- **exp19 DISCARD**: max_cat_threshold=16 (-0.0002).
- **exp20 DISCARD**: n_est 2000→3000 lr 0.03→0.02 (TIMEOUT).
- **exp21 DISCARD**: reg_alpha=1.0 (-0.0003).
- **exp22 DISCARD**: manual CV + ES (inner 90/10 holdout) → 0.7768. 8k row loss > ES gain.
- **exp23 DISCARD**: drop DayofMonth → 0.7746. Real signal, hurts.
- **exp24 DISCARD**: depth=8, n_est=1200 → 0.7807 (-0.0005).
- **exp25 KEEP**: depth=8, n_est=1500 → 0.7814 (+0.0002, marginal).
- **exp26 KEEP**: depth=8, n_est=1800 → 0.7820 (+0.0006).
- **exp27 DISCARD**: n_est=2000 at depth=8 → 0.7820 same, saturated.
- **exp28 DISCARD**: lr=0.025 at depth=8 → 0.7812.
- **exp29 DISCARD**: mcw=80 at depth=8 → 0.7817.

## Synthesis after ~28 exps

**Current best (a5a8593): 0.7820** with depth=8, n_est=1800, lr=0.03, mcw=50, +Hour. Runtime 46s.

**What clearly works:** more trees+lower lr (saturated), Hour categorical FE, min_child_weight (sweet spot 50), depth 7 vs 6 marginal, depth 8 + 1800 trees marginal-positive.

**What clearly doesn't:** subsample/colsample at 0.8-0.9, gamma>0, reg_alpha/lambda, dropping DepTime/DayofMonth, explicit Hour×DOW, high-card cats (Route timeout), manual ES (data loss > gain).

**Local optimum signs:** last 6+ tweaks barely move (within fold noise). Diminishing returns acute.

**Next ideas:** max_bin tuning, colsample_bynode, multi-seed ensemble (cost), light colsample_bytree=0.95.

---

## Compressed log (exps 30-69)

- **exp30 DISCARD**: max_bin=128 (-0.0011).
- **exp31 KEEP**: colsample_bytree=0.95 → 0.7827 (+0.0007).
- **exp32-33 DISCARD**: colsample 0.9 (rounds to same), colsample_bynode=0.8 (no diff).
- **exp34 DISCARD**: lr=0.025 (-0.0001 noise).
- **exp35 DISCARD**: drop Distance → 0.7799 (real signal).
- **exp36 KEEP**: **add Minute = DepTime % 100 numeric → 0.7933 (+0.0106). HUGE WIN.**
  Hypothesis: HHMM-encoded DepTime hides minute structure behind discontinuous gaps; explicit Minute lets trees factorize Hour × Minute in 2 splits instead of multiple DepTime range splits.
- **exp37 DISCARD**: drop DepTime (essentially equal -0.0003, kept for safety).
- **exp38 DISCARD**: Minute as categorical → 0.7857. Numeric better.
- **exp39 KEEP** (later marked DISCARD over budget): mcw 50→20 → 0.7950. Richer features need less reg.
- **exp40 DISCARD**: mcw 20→10 (-0.0004).
- **exp41 KEEP** (later DISCARD over budget): n_est 1800→2000 → 0.7952. Total runtime 81s, exceeded budget.
- **exp42 KEEP**: mcw=20 + n_est 1800→1500 (budget-fit) → 0.7944. Real keep.
- **exp43-44 DISCARD**: lr=0.025 (-0.0003), depth=7+n_est=2000 (TIMEOUT).
- **exp45 KEEP**: mcw 20→15 → 0.7947 (+0.0003 marginal).
- **exp46 DISCARD**: mcw 15→12 (TIMEOUT).
- **exp47-49 DISCARD**: lr=0.035, drop colsample (TIMEOUT), max_bin=128.
- **exp50 DISCARD**: drop DepTime (-0.0018, real signal).
- **exp51 DISCARD**: grow_policy=lossguide max_leaves=200 (TIMEOUT).
- **exp52 DISCARD**: add MinuteMod5 (TIMEOUT — no n_est cut).
- **exp53 KEEP**: **n_est 1500→1300 + add MinuteMod5 → 0.7965 (+0.0018). Off-schedule signal.**
  Hypothesis: scheduled departures cluster on x:00, x:05, x:10... so `Minute % 5 != 0` correlates with already-delayed flights.
- **exp54 KEEP**: n_est 1300→1400 → 0.7967 (+0.0002 marginal).
- **exp55 DISCARD**: MinuteMod30 → 0.7943 (subset overlap with Mod5).
- **exp56 DISCARD**: n_est 1400→1500 (TIMEOUT).
- **exp57-59 DISCARD**: mcw=10 (-0.0009), gamma=0.1 (no change), MinuteMod10 (TIMEOUT).
- **exp60 KEEP**: add LogDistance = log1p(Distance) → 0.7970 (+0.0003 marginal).
  Hypothesis: histogram quantile bins differ between raw and log-distance scale; log gives finer bins for short flights.
- **exp61-67 DISCARD**: drop Distance, subsample=0.95 (TIMEOUT), HourNum (TIMEOUT), n_est 1400→1200+Mod10, reg_lambda=3, replace DepTime with DepTimeMin (TIMEOUT/no help), depth 8→7+n_est=1900 (TIMEOUT).
- **exp68 DISCARD**: depth=7 n_est=1900 (TIMEOUT).
- **exp69 DISCARD**: max_cat_threshold=4 (-0.0055, too aggressive).

## Synthesis after ~69 exps

**Current best (3f0e7bc): 0.7970** with depth=8, n_est=1400, lr=0.03, mcw=15, colsample_bytree=0.95.
Features: Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest, DepTime, Distance + Hour cat, Minute num, MinuteMod5 num, LogDistance num.
Total runtime ~64s (right at budget edge).

**Total improvement vs baseline: 0.7445 → 0.7970 = +0.0525.**

**Key insights from this run:**
1. **Implicit-vs-explicit feature decomposition matters.** XGBoost couldn't extract minute-level signal cleanly from HHMM-encoded DepTime; making it explicit (`Minute`) gave +0.0106.
2. **Schedule-clean times correlate with on-time departures.** `MinuteMod5` (0 = on a 5-min schedule slot, ≠0 = off-schedule = already partially delayed) added +0.0018.
3. **Histogram bin distribution matters under hist tree method.** `log1p(Distance)` redistributes split candidates and gave +0.0003.
4. **Regularization optimum shifts with feature richness.** mcw=50 was optimal pre-Minute; mcw=15 is optimal post-Minute (richer features need less reg).
5. **Runtime budget is a real constraint.** Many promising experiments timed out and required sacrificing trees to add features.

**What's plateaued:** all standard hyperparameters (lr, depth, mcw, reg_*, colsample_*, subsample, gamma, max_bin, max_cat_threshold, grow_policy) explored; feature ideas exhausted within budget.

---

## Compressed log (exps 70-85)

- **exp70 DISCARD**: depth=9 n_est=700 → 0.7949 (too few trees) but only 40s.
- **exp71 KEEP**: depth=9 n_est=1000 → 0.7974.
- **exp72 KEEP**: depth=9 n_est=1100 → 0.7978.
- **exp73-76 DISCARD**: mcw=20/25 (equal/no help), n_est=1200 (TIMEOUT), lr=0.025 (-0.0007), mcw=10 (TIMEOUT).
- **exp77 DISCARD**: depth=10 n_est=600 → 0.7954 (too few).
- **exp78 KEEP**: depth=10 n_est=900 → 0.7980.
- **exp79 KEEP**: depth=10 n_est=950 → **0.7983**.
- **exp80-85 DISCARD**: mcw 25/18 (-0.0002/-0.0001), lr=0.025 (TIMEOUT variance), depth=11 n_est=500 (-0.0029), mcw=12 (TIMEOUT), Mod10 + n_est cut (-0.0019).

## Final synthesis (~85 exps)

**Final best (6b09446): 0.7983** with depth=10, n_est=950, lr=0.03, mcw=15, colsample_bytree=0.95.
Features: original 8 + Hour cat + Minute num + MinuteMod5 num + LogDistance num.

**Total improvement vs baseline: 0.7445 → 0.7983 = +0.0538.**

**Path of major wins (cumulative):**
1. baseline = 0.7445
2. + n_est 30→2000, lr 0.1→0.03 = 0.7626 (+0.0181)
3. + Hour categorical = 0.7728 (+0.0102)
4. + min_child_weight=50 = 0.7792 (+0.0064)
5. + max_depth=7 = 0.7812 (+0.0020)
6. + colsample_bytree=0.95 = 0.7827 (+0.0015)
7. + Minute numeric = 0.7933 (+0.0106) ← 2nd biggest win
8. + mcw=15 (post-Minute) = 0.7947 (+0.0014, smaller-mcw tighter optimum)
9. + MinuteMod5 = 0.7965 (+0.0018, schedule-clean signal)
10. + LogDistance = 0.7970 (+0.0003)
11. + depth=10 (with n_est=950) = 0.7983 (+0.0013)

**Two unexpected breakthroughs:**
- **Minute (DepTime % 100)** — biggest revealed insight: HHMM-encoded DepTime hides minute-level structure behind discontinuous gaps. Explicit Minute lets trees factorize Hour × Minute interactions cleanly.
- **MinuteMod5** — schedule-clean times (x:00, x:05...) correlate with on-time departures.

---

## Compressed log (exps 86-97)

- **exp86 KEEP**: lr 0.03→0.04 at depth=10 → **0.7985** (+0.0002).
- **exp87 DISCARD**: lr=0.045 (equal).
- **exp88-90 DISCARD**: n_est=850 (-0.0003), n_est=970 (TIMEOUT), subsample=0.97 (TIMEOUT).
- **exp91 DISCARD**: drop UniqueCarrier → 0.7845 (strong signal).
- **exp92-94 DISCARD**: dart booster (TIMEOUT), max_bin=512 (TIMEOUT), HourSin/Cos+n_est cut (-0.0012).
- **exp95 DISCARD**: drop Month → 0.7865 (strong seasonal signal).
- **exp96-97 DISCARD**: gamma=0.5 (-0.0094), gamma=0.2 (-0.0017).

## True final state (~97 exps): 0.7985

Final commit: 5b6b97f (exp86).
Final config: depth=10, n_est=950, lr=0.04, mcw=15, colsample_bytree=0.95.
Features: Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest (cats) + DepTime, Distance (num) + Hour cat, Minute num, MinuteMod5 num, LogDistance num.
Total runtime: ~63s (right at budget).

**Total improvement: 0.7445 → 0.7985 = +0.0540 AUC.**

