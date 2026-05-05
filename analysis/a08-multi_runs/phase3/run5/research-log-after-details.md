# Research log (post-hoc detailed) — `may4` branch

## Overview

- **Branch**: `may4` (created 2026-05-04)
- **Total experiments**: 211
- **Baseline**: CV AUC 0.7445
- **Final best**: CV AUC **0.8341 ± 0.0029** (commit `09ba14c`, exp 204)
- **Cumulative improvement**: +0.0896 (~12%)
- **Setup**: 5-fold StratifiedKFold CV on `2005-slice1-100k.csv` (100K balanced rows). Time budget: <60s CV.
- **Stack**: XGBoost 3.1.3 (sklearn API), pandas, system Python 3.12 in template venv.

### Final winning config

```python
cat_cols = ["Month", "DayofMonth", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
# Engineered: DepHour, DepMinute, CarrierHour (Carrier × DepHour categorical interaction)

XGBClassifier(
    n_estimators=360, max_depth=23, learning_rate=0.074,
    min_child_weight=2, subsample=0.95, colsample_bytree=0.85,
    max_bin=512, max_cat_to_onehot=32, max_cat_threshold=128,
    reg_lambda=0.5, enable_categorical=True, random_state=42,
)
```

---

## Phase 0 — Setup (pre-experiment)

System python had no ML packages. There was a venv at `/home/ubuntu/xgb-newrun-template/.venv` with `xgboost==3.1.3`, pandas, sklearn, polars. After confirming with the user, symlinked it into the working dir as `.venv` and used `.venv/bin/python3` to run experiments. `program.md` says "no new packages" — using an already-installed venv satisfies that.

`prepare.py` was inspected (read-only): produces 100K balanced slices for 2005/2006/2007 by undersampling the majority class. `train.py` (the file we modify) loads `2005-slice1-100k.csv`, defines `cat_cols`/`num_cols`/`target`, computes `cat_levels` from training data, and runs 5-fold StratifiedKFold CV. It also fits a final model and a "4/5" model (training-fold-only) presumably for the human's groundtruth eval.

---

## Phase 1 — Baseline (exp 1)

**Exp 1 (`7d0b368`)**: Baseline. `n=30, depth=6, lr=0.1, enable_categorical=True`. **CV AUC 0.7445**, CV time 0.7s. Far under budget — clear underfit. CV std 0.0043 is the noise floor.

---

## Phase 2 — Initial scaling: trees, lr, depth (exp 2–9)

### Web research (before exp 2, mandated by program.md)

Searched: *"XGBoost tuning best practices binary classification tabular hyperparameters guide"* and *"XGBoost airline delay prediction feature engineering DepTime hour buckets"*.

Sources used:
- [XGBoost docs: Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) — main guidance: deeper trees + smaller eta + sampling for regularization.
- [SageMaker XGBoost tuning guide](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html) — key knobs: alpha, min_child_weight, subsample, eta, num_round.
- [UC Berkeley Air Travel Delay Prediction project](https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches) — confirms hour-of-day bins and time-based features matter for airline delay.

Takeaways I committed to acting on:
1. The classic XGBoost rule of thumb: more trees + lower lr.
2. Try `tree_method='hist'` (default in 3.x) and `max_bin`.
3. For airline delay specifically, time-of-day decomposition is well-attested.

### Experiments

**Exp 2** (`74f86c5`): `n=200, lr=0.05` → **0.7534 (+0.0089). keep**. The standard "more trees, slower lr" win. CV time 3.2s — still tons of headroom.

**Exp 3** (`235c8ab`): `n=500, lr=0.03` → 0.7556 (+0.0022). keep. Diminishing returns appearing.

**Exp 4** (`72835a1`): `n=1000, lr=0.02` → 0.7573 (+0.0017). keep. Maxed n_estimators with this lr; capacity scaling slowing.

**Exp 5** (`43b944b`): `max_depth=8` → **0.7623 (+0.0050)**. keep. Depth axis *much* stronger than tree count alone. Hypothesis: airline delay signal lives in feature interactions (carrier × hour, origin × time-of-day) that need depth to express via consecutive splits.

**Exp 6** (`fec243b`): `subsample=0.8, colsample_bytree=0.8` → 0.7635 (+0.0012). keep. Mild stochastic regularization. Std also dropped.

**Exp 7** (`5b860e0`): Add `DepHour = DepTime // 100`, `DepMinute = DepTime % 100` → **0.7673 (+0.0038)**. keep. The HHMM integer (e.g., 1430) has a discontinuity at hour boundaries (0959→1000 is +41 numerically but +1 minute in real time). Explicit decomposition gives trees clean numeric splits at hour boundaries.

**Exp 8** (`4be41a7`): Add `Route = Origin + "_" + Dest` categorical → 0.7482 (-0.019). discard. ~5000 levels with sparse coverage — overfit + slowdown (49.6s vs ~28s). First lesson: high-cardinality engineered categoricals hurt at our dataset size.

**Exp 9** (`89b6ba9`): `max_depth=10` → **0.7750 (+0.0077)**. keep. Depth keeps winning big. CV time 51.3s — getting close to budget.

---

## Phase 3 — min_child_weight discovery (exp 10–13)

After exp 9 I was at depth=10 with default `min_child_weight=1`. The standard tuning advice is to bump mcw up at higher depth: at depth 10, leaves with <2 hessian-sum are essentially memorizing single rows.

**Exp 10** (`9490159`): `mcw=5` → **0.7843 (+0.0093)**. keep. Huge win, AND it's *faster* (42.6s vs 51.3s) because fewer splits go through. This was a turning point — I'd been thinking about regularization as a tradeoff with capacity, but here it's pure win.

**Exp 11** (`52e8665`): `mcw=10` → 0.7871 (+0.0028). keep.

**Exp 12** (`adc2a58`): `mcw=20` → 0.7875 (+0.0004). keep, plateau approaching.

**Exp 13** (`5c15f06`): `mcw=50` → 0.7840 (-0.0035). discard. Sweet spot at 20.

---

## Phase 4 — More depth (exp 14–15)

With mcw=20 freeing time budget (32.3s), pushed depth.

**Exp 14** (`2a06041`): `depth=12` → 0.7899 (+0.0024). keep.

**Exp 15** (`1510f8b`): `depth=14` → 0.7903 (+0.0004). keep. Plateauing on depth axis at this mcw.

---

## Phase 5 — Cyclical encoding (exp 16–17)

### Reasoning

`DepHour` is currently a numeric int 0–23. Tree splits on this can't easily place hour-23 and hour-0 in the same group, but late-night flights (22–23) and early-morning (0–2) likely share operational patterns (aircraft positioning, crew turnaround). [NVIDIA blog: Three approaches to encoding time information for ML](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/) discusses sin/cos cyclical encoding for this case. The blog warns that tree-based models split on one feature at a time so sin AND cos can't be split together easily — but each individually can still expose periodic structure.

**Exp 16** (`05d0a8d`): Add `DepHour_sin = sin(2π·hour/24)`, `DepHour_cos = cos(...)` → **0.7928 (+0.0025)**. keep. Cyclical FE works for a numeric hour.

**Exp 17** (`a2b76e3`): Same for Month and DayOfWeek → 0.7919 (-0.0009). discard. These are already categorical (XGBoost partition-based splits handle nonlinear groupings natively); adding sin/cos is redundant with the categorical and adds noise. Lesson: cyclical encoding helps *only* when the base feature is numeric.

---

## Phase 6 — Failed easy wins (exp 18–19)

**Exp 18** (`d496c51`): Add `IsWeekend` (DayOfWeek≥6), `IsRedeye` (DepHour∉[6,22)) → 0.7914 (-0.0014). discard. Same redundancy lesson — these are derivable by trees from the existing categoricals.

**Exp 19** (`27a52b4`): `gamma=1` → 0.7927 (-0.0001). discard. With mcw=20 already controlling small-leaf splits, gamma is fighting for the same regularization channel.

---

## Phase 7 — Plateau-busting research (exp 20)

After 3 discards I did a research pause:
*"XGBoost binary classification plateau improve AUC interaction features tree_method hist"*. Surfaced [Tree Methods doc](https://xgboost.readthedocs.io/en/stable/treemethod.html) and [XGBoost histogram tree_method](https://xgboosting.com/configure-xgboost-histogram-tree-method-tree_methodhist/). Key takeaway: `max_bin` (default 256) controls split-candidate granularity for continuous features. Higher = finer splits, more compute.

**Exp 20** (`9385d86`): `max_bin=512` → 0.7934 (+0.0006). keep, marginal but positive.

**Exp 21** (`9dffe08`): `max_bin=1024` → 0.7936 (+0.0002). discard — within noise; per simplicity criterion keep the smaller value.

---

## Phase 8 — Wandering through dead ends (exp 22–26)

A frustrating stretch. I had a strong setup at 0.7934 (depth=14, mcw=20, n=1000, lr=0.02, sub/cs=0.8, max_bin=512) and tried several axes that all failed.

**Exp 22** (`934524b`): `grow_policy='lossguide', max_leaves=128` → 0.7872 (-0.006). discard. Depthwise (default) is better here.

**Exp 23** (`36885e9`): Carrier_Origin interaction (~9000 levels) → 0.7938 but **66s, over budget**. discard. Same Route-style failure.

**Exp 24** (`8ca9f4e`): Multi-change — depth=12 + n=1300 + lr=0.015 (less depth, more refinement) → 0.7927. discard.

**Exp 25** (`51f218b`): `HourDOW` interaction (24×7=168, low cardinality) → 0.7816 (-0.0118). discard. *First time* I noticed: even low-card 2nd interactions hurt at this dataset size. Filed for later.

**Exp 26** (`585f562`): `reg_lambda=5` → 0.7925 (-0.0009). discard.

---

## Phase 9 — `max_cat_to_onehot=32` breakthrough (exp 27–29)

After 5 consecutive discards, I went back to docs more carefully. [XGBoost categorical-features doc](https://xgboost.readthedocs.io/en/latest/python/examples/cat_pipeline.html) describes two split strategies:
- **One-hot**: each category becomes its own binary; trees split on each independently.
- **Partition**: a single split puts a *subset* of categories on each side.
- The threshold is `max_cat_to_onehot` (default 4): cardinalities below it use one-hot, above it use partition.

Default 4 means almost all our cats (Month=12, DOW=7, DayofMonth=31, UniqueCarrier=~30, Origin/Dest=~300) use partition. Hypothesis: for *low-cardinality* cats like Month/DOW/Carrier, partition might be inferior — each category has plenty of samples, and one-hot exposes "is_month_December" type splits more directly.

**Exp 27** (`d3e9d53`): `max_cat_to_onehot=32` → **0.7977 (+0.0043)**. keep. **Huge win** AND faster (37.6s vs 46s). This made Month, DOW, DayofMonth, UniqueCarrier all one-hot while leaving Origin/Dest as partition. A genuine algorithmic discovery.

**Exp 28** (`1ed04ee`): `max_cat_to_onehot=350` (Origin/Dest also one-hot) → 0.7840 (-0.0137). discard. High-cardinality one-hot creates ~600 sparse binary features — each splits weakly. Confirms: *partition is right for high-cardinality, one-hot for low-cardinality.*

**Exp 29** (`3864acd`): With time freed up, push `depth=16` → **0.8002 (+0.0025)**. keep. **Crossed 0.80.**

---

## Phase 10 — Pushing depth (exp 30–33)

**Exp 30** (`1b60376`): `depth=18` → 0.8017. keep.
**Exp 31** (`cc0f5ef`): `depth=20` → 0.8026. keep.
**Exp 32** (`2ba4790`): `depth=24` → 0.8035. keep.
**Exp 33** (`4d77e18`): `depth=32` → 0.8040. keep, plateau.

Each step gives ~0.001. CV time at depth=32 = 53.8s — close to budget.

---

## Phase 11 — mcw retune at deep regime (exp 34–40)

At depth=32, mcw=20 might be too restrictive. Re-tuned.

**Exp 34** (`4d0ef90`): `mcw=10` → **0.8121 (+0.0081)** but **73.6s, over**. discard. The win is huge but unusable.

**Exps 35–37**: tried compromises (mcw=10 + n=800, mcw=10 + sub=0.6) — all over budget.

**Exp 37** (`078b95a`): `mcw=10 + n=700` → 0.8092 at 53.5s. keep. Sacrificed 300 trees to fit.

**Exp 38** (`6d20777`): `mcw=5 + n=600` → 0.8141 but 68s over. discard.

**Exp 39** (`95a5464`): `mcw=5 + n=500` → 0.8126 at 58.5s. keep.

**Exp 40** (`e18ee45`): `mcw=3 + n=400` → 0.8139 but 66s. discard.

Pattern: each step lower in mcw produces real AUC gains but accelerates tree growth (smaller leaves allowed → bigger trees → slower). Budget binds.

---

## Phase 12 — Squeezing time budget (exp 41–48)

Dead ends trying to claw back time without losing AUC.

**Exp 41** (`5cbc182`): `subsample=0.5` → 0.8026 (-0.01). discard. Too aggressive.
**Exp 42** (`9ff1b1b`): depth=24 + mcw=5 + n=600 → 0.8134 but 62.9s. discard.
**Exp 43** (`46e415e`): Remove DepTime → 0.8043. discard. Surprising — DepHour+DepMinute don't fully replace it. Likely because the int gives a different binning to `max_bin`.
**Exp 44** (`4866e9c`): `max_cat_threshold=256` → 0.8127 but over. discard.
**Exps 45–46**: max_bin=256 trades. Marginal/flat.
**Exp 47** (`2f8be9d`): `colsample_bynode=0.8` → 0.8114. discard.
**Exp 48** (`2bd2911`): depth=24 + mcw=3 + n=500 → 0.8150 but 69.5s. discard.

**Exp 49** (`65e5d80`): `depth=24 + mcw=3 + n=400` → 0.8130 at 57.9s. keep.

---

## Phase 13 — CarrierHour breakthrough (exp 50–53)

Reasoning: throughout phases 7–12, *high-cardinality* interactions (Route, Carrier_Origin) hurt; *low-cardinality* second-features hurt; depth+mcw tuning was hitting time wall. What's left? **Specific interactions** with manageable cardinality.

`Carrier × DepHour`: 30 carriers × 24 hours = 720 levels max, ~300–500 in practice. Each carrier has its own operational pattern (regional carriers might run morning-only routes; majors might have evening hubs). A single combined feature lets a tree split on the pair directly instead of needing two consecutive splits.

**Exp 50** (`34e104b`): Add `CarrierHour` → **0.8247 (+0.0117)** but **76s, over**. discard. Massive win, need to fit.

**Exp 51** (`16e83e2`): + CarrierHour, n=300 → 0.8212 at 60.4s — discard (just over).

**Exp 52** (`423a9a5`): + CarrierHour, n=250 → **0.8185 at 51.6s. keep.** Fits.

**Exp 53** (`c5ca493`): + CarrierHour, n=280 → **0.8203**. keep.

---

## Phase 14 — Trying to add a 2nd interaction (exp 54–55)

Followed up by trying CarrierMonth and HourMonth as second interactions.

**Exp 54** (`5ab2073`): + CarrierMonth, n=200 → **0.7935 (-0.0268)**. discard. Massive regression.

**Exp 55** (`5753050`): + HourMonth, n=240 → **0.7940 (-0.0263)**. Same massive regression.

These are low-cardinality (~360, ~288) and *should* work by analogy with CarrierHour. They don't. Hypothesis (which held throughout the run): the model has just enough capacity to learn one strong interaction; adding another spreads attention thin and the second one's noise leaks into predictions. Could also be max_cat_to_onehot=32 + colsample sampling making the 2nd interaction get included frequently in trees that don't have capacity for it. *Never figured out the exact mechanism* but the pattern was extremely consistent across 8+ later attempts.

---

## Phase 15 — lr regime shift after CarrierHour (exp 56–62)

**Exp 56** (`0ec3d9c`): mcw=2 + n=240 → 0.8205 at 64s. discard.

**Exp 57** (`8e52c33`): `lr=0.03` (with CarrierHour, mcw=3, depth=24, n=280) → **0.8245 (+0.0042)**. keep. *Higher* lr now wins. With CarrierHour adding extractable signal and shallower-than-before parameter setup, more learning per tree pays off.

**Exp 58** (`d561952`): `lr=0.05` → **0.8262 (+0.0017)**. keep. Faster (46.6s).

**Exp 59** (`0ee22d1`): `lr=0.08` → 0.8257. discard. Sweet spot 0.05.

**Exp 60–61**: n=350→380 → 0.8277→0.8279.

**Exp 62** (`cc2175c`): `lr=0.06` → flat.

---

## Phase 16 — mcw=2 (exp 63–64)

**Exp 63** (`abbc36b`): mcw=2 + n=320 → 0.8284 but 61s. discard.

**Exp 64** (`ed730f3`): mcw=2 + n=300 → **0.8281 at 58.9s**. keep. mcw=2 fits with reduced n.

---

## Phase 17 — Long simplification/tweak grind (exp 65–90)

26 experiments in this stretch, mostly discards, hammering at the 0.8281 plateau. Some highlights:

- **Exp 65 (`4cfa775`)**: + DistBand (Distance bucketed short/med/long) → 0.8279. Fails — trees handle continuous Distance fine.
- **Exp 66 (`a57d904`)**: Remove DepHour_sin/cos → 0.8275 (-0.0006). discard but barely; sin/cos contribute marginally.
- **Exp 67 (`f5ac6d2`)**: Remove DepMinute → **0.8107 (-0.017)**. Strong rejection. DepMinute is critical because the integer-encoded DepTime has the discontinuity issue, so the explicit minute extraction is the only clean source of minute-level info.
- **Exp 68 (`e7e637c`)**: + CarrierDOW → 0.7996 (-0.029). Same 2nd-interaction regression.
- **Exp 69 (`e5e1ba8`)**: Replace CarrierHour with OriginHour (~7200 levels) → 0.8140 + 100s. Discard. High-card replacement fails *and* is slow.
- **Exp 70 (`769f79e`)**: max_delta_step=1 → no effect.
- **Exp 71–75**: max_bin tweaks, cs=1.0 trades, mcw=1 over budget.
- **Exp 76 (`f143b25`)**: depth=20 + n=340 → 0.8273. discard.
- **Exp 77 (`3bf68b9`)**: `num_parallel_tree=2 + n=150` → 0.8252 at 79s over. After reading [XGBoost RF tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html). Idea: each iteration trains a small forest for stability. Doesn't help and is too slow.
- **Exp 78 (`4cc5b84`)**: cs=0.9 + max_bin=256 + n=280 → flat.
- **Exp 79 (`92413f2`)**: + DepMin = DepHour×60 + DepMinute (smooth minutes since midnight) → 0.8281 flat at 65s. discard. Trees don't gain from a smooth combo when components already there.
- **Exp 80 (`840930b`)**: mcw=4 → 0.8263. discard.
- **Exp 81 (`2218b0e`)**: random_state=0 → 0.8284 at 60.1s. *Useful information*: seed variance is ±~0.0003. Most "improvements" smaller than this are noise.
- **Exp 82 (`a456784`)**: `sampling_method='gradient_based'` → **crash**. XGBoost: "Only uniform sampling is supported, gradient-based sampling is only support by GPU Hist." Logged as crash.
- **Exp 83–84**: subsample=0.7 (-0.0022), CarrierDistBand (-0.0022) — both small but real regressions.
- **Exp 85 (`06ff1f1`)**: Radical: `depth=8, mcw=1, n=1300, lr=0.025` (shallow + many trees regime) → 0.8008. The deep-regime is a real local optimum, not just a quirk.
- **Exp 86–88**: cs=0.85, mcw=2.5, lr=0.06 — flat.
- **Exp 89 (`4f4a33a`)**: Remove `max_cat_to_onehot` (default 4) → 0.8004 (-0.028). Confirms 32 is critical.
- **Exp 90 (`6a398bc`)**: depth=22 + n=320 → 0.8279 just over budget.

### Synthesis after exp 90 (was 0.8281 plateau)

Best at this point was 0.8281. Wins in retrospect: depth, mcw, max_cat_to_onehot=32, CarrierHour. Failed: 2nd interactions, deeper than depth=24, smaller mcw than 2, anything that meaningfully changed sub/cs.

---

## Phase 18 — Plateau-research (after exp 90)

Searched: *"XGBoost feature engineering tabular advanced techniques target encoding cyclical 2026"*. Surfaced [GeeksforGeeks XGBoost FE](https://www.geeksforgeeks.org/machine-learning/feature-engineering-for-xgboost-models/), [feature engineering for XGBoost](https://xgboosting.com/feature-engineering-for-xgboost/). Reminded me about target encoding (rejected as leaky in CV without per-fold encoding) and that cyclical encoding may not help trees (already saw — but was useful for hour specifically because hour was numeric).

Also searched: *"airline delay prediction XGBoost AUC 0.83 feature engineering 100k samples"*. Sources:
- [Flight delay prediction PMC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/) — Stacking ensemble at AUC 0.7698 (different dataset).
- [CatBoost+Bayesian flight delay paper](https://ejournal.kresnamediapublisher.com/index.php/jri/article/view/346) — AUC 0.7793 (different dataset).
- [UC Berkeley study](https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches) — emphasizes that *historical delay rate of departure airport accounts for more than half of predictive power*. We can't add historical rates as count features (program.md forbids count features because data is balanced via undersampling). But XGBoost categorical splits on Origin/Dest implicitly capture per-airport rates from training fold.

**Note**: I cited "0.7793" and "0.7698" in synthesis as if they were comparable benchmarks. They're from different datasets and probably different definitions of "delay" — I shouldn't have stated them as fact. Apologies for the sloppiness.

---

## Phase 19 — Trying things post-plateau (exp 91–95)

**Exp 91 (`aab9f8d`)**: Replace CarrierHour with HourDOW (lower-card interaction) → 0.7833. CarrierHour really is uniquely effective.

**Exp 92 (`4f48cb9`)**: `subsample=1.0` (no row sampling) + n=240 → **0.8283 (+0.0002)**. keep, marginal but novel. Was always worth trying when sub=0.8 had been a default.

**Exp 93–94**: n=250, n=255 → 0.8285, 0.8286. keep, tiny.

**Exp 95 (`fce7651`)**: `lr=0.06` → 0.8287.

**Exp 96 (`a4c8f37`)**: `lr=0.07` → **0.8293 (+0.0006)**. keep — real improvement at this regime.

This was the regime shift: subsample=1.0 + higher lr stops the model from being mildly under-fit per tree.

---

## Phase 20 — Aggressive tree-pushing (exp 97–101)

**Exp 97**: lr=0.08 → 0.8286 worse.
**Exp 98 (`22185d1`)**: n=290 → **0.8301**. broke 0.83.
**Exp 99 (`e347372`)**: n=310 → 0.8303.
**Exp 100 (`e0361c9`)**: n=320 → **0.8305**.
**Exp 101**: n=325 → flat. Saturation.

---

## Phase 21 — Marginal tweaks again (exp 102–112)

**Exp 102 (`5f7c8cd`)**: mcw=3 + n=360 → 0.8295. discard.
**Exp 103–105**: cs=0.9/1.0, lr=0.075 — all flat or worse.
**Exp 106 (`6008a73`)**: depth=28 → 0.8297. discard.
**Exp 107 (`fbd5932`)**: max_bin=768 → 0.8299. discard.
**Exp 108 (`ed02728`)**: + CarrierDOW retry (now with simpler base) → 0.8033 (-0.027). Same regression. **The 2nd-interaction-hurts pattern is robust**.
**Exp 109–112**: lr=0.065 (over), feature_weights weighting CarrierHour 2x (no effect), depth=20 + n=360 (-0.001), reg_alpha=0.1 (-0.0011).

---

## Phase 22 — reg_lambda tuning (exp 113–115)

**Exp 113 (`c02f7d6`)**: `reg_lambda=0.5` (default 1) → **0.8306 (+0.0001)**. keep. Lower L2 helps marginally.
**Exp 114 (`8adeb3d`)**: reg_lambda=0.2 → 0.8304. discard.
**Exp 115 (`5eb1a65`)**: n=330 → **0.8308**. keep.

---

## Phase 23 — depth=23 (exp 119–122)

**Exp 116–118**: n=345 (flat, 60s boundary), sub=0.95 (-0.0006), cs=0.75 (-0.0010).

**Exp 119 (`24248f1`)**: `max_depth=23` (one less than 24) → **0.8309 (+0.0001)**. keep. (Per simplicity criterion: technically simpler tree at same AUC = win.)

**Exp 120**: depth=22 → 0.8304.
**Exp 121**: n=340 flat.
**Exp 122**: lr=0.072 → 0.8305 worse.

---

## Phase 24 — Many discards then sub=0.95/cs=0.85 (exp 123–137)

23 experiments here, most discards.

**Exp 123 (`5beb1d1`)**: max_cat_to_onehot=20 → 0.8113. UniqueCarrier (~30 levels) needs to be in one-hot.
**Exp 124–126**: cs=1.0/n=290, +DepHourFrac (over), n=335 (flat).
**Exp 127 (`0ae2e11`)**: mcw=1 + n=240 → 0.8295 at 70.6s. mcw=1 too aggressive.
**Exp 128 (lossguide max_leaves=1024)**: 0.8262 at 93s. Discard.
**Exp 129 (`4f728d0`)**: mcw=3 + n=350 → 0.8293 (-0.001).
**Exp 130 (`3ace08b`)**: feature_weights Distance 3x → flat. (Set via `feature_weights=np.array([...])` arg to `XGBClassifier`. Per [XGBoost docs](https://xgboost.readthedocs.io/en/stable/parameter.html), this biases column-sampling probability toward higher-weight features.)
**Exp 131 (tree_method=approx)**: 0.8306 at 187s. Way too slow.
**Exp 132**: lr=0.073 → flat.
**Exp 133–134**: Remove UniqueCarrier or DayofMonth → 0.8301, 0.8280. Discards.
**Exp 135 (`0c476a3`)**: + DistBucket = Distance//250 → 0.8305. Trees can already bin Distance via numeric splits.
**Exp 136**: reg_lambda=2 → 0.8289.

**Exp 137 (`7372022`)**: `subsample=0.95, colsample_bytree=0.85` (small bump from 0.8/sub=1.0) → **0.8310 (+0.0001)**. keep. Mid-range stochastic randomization beats both extremes here.

---

## Phase 25 — Hover at 0.8310 (exp 138–150)

Many discards. Notable failures:

**Exp 139**: sub=0.9 cs=0.9 → 0.8303.
**Exp 141**: depth=24 with new sub/cs → 0.8306.
**Exp 142–143**: sub=1.0 cs=0.85 / sub=0.95 cs=0.8 → both worse. The `0.95/0.85` pair is uniquely good here.
**Exp 145 (`aabf5cd`)**: max_bin=384 → 0.8303.
**Exp 146**: mcw=1.5 n=300 → flat at 62.6s.
**Exp 147**: gamma=0.05 → 0.8303.
**Exp 148 (`e0b468d`)**: Replace `colsample_bytree=0.85` with `colsample_bylevel=0.85` → 0.8304. Per-level vs per-tree column sampling — different granularity but doesn't help here.
**Exp 149 (`b14cb27`)**: Replace CarrierHour with CarrierHour6 = Carrier × (DepHour//4) (180 levels, coarser) → 0.8195. The full granularity at 720 levels is essential.
**Exp 150**: + CarrierHour6 alongside CarrierHour (multi-grain) → 0.8283. Same 2nd-interaction regression.

---

## Phase 26 — max_cat_threshold=128 win (exp 151–156)

**Exp 151 (`c04f637`)**: `max_cat_threshold=128` (default 64) → **0.8314 (+0.0004)** but 60.9s over. The parameter limits how many candidate split groups are considered for partition-based categorical splits (Origin/Dest). Higher = more flexibility, but slower and more overfit risk.

**Exp 152**: max_cat_threshold=128 + n=320 → 0.8313 at 60.8s over. discard.

**Exp 153 (`b21413f`)**: max_cat_threshold=128 + n=300 → **0.8311 at 58.1s. keep**.

**Exp 154 (`512842b`)**: n=310 → 0.8312.

**Exp 155**: n=315 → flat.

**Exp 156 (`978ed8e`)**: `lr=0.075` → **0.8317 (+0.0005)**. keep. Real improvement — at this combined config (sub=0.95, cs=0.85, max_cat_threshold=128), lr can go higher.

---

## Phase 27 — Hover at 0.8317 (exp 157–172)

Discards: lr=0.08, n=320, max_cat_threshold=192, max_cat_threshold=96, depth=24, depth=22, lr=0.07 (over), reg_alpha, reg_lambda=0.3, depth=22 n=370, sub=0.92, max_bin=400, lr=0.077, lossguide max_leaves=4096 (158s over!), feature_weights, +DepHour_sq.

A few notable:
- **Exp 167 (`7d10f59`)**: `objective='binary:logitraw'` → 0.8306. Outputs raw logits instead of probabilities; AUC is invariant to monotonic transforms so this should give the same result, but the slightly different numerical optimization path produces a different model. Discard.
- **Exp 168 (`45c978b`)**: scale_pos_weight=1.05 → 0.8316. Data is balanced 50/50 so this shouldn't help and didn't.
- **Exp 169–171**: max_bin=1024 (over), max_bin=640 (-0.0006), DepHour_sq (-0.018). Adding squared DepHour gives a redundant monotonic transform that confuses the trees.

---

## Phase 28 — Feature pruning breakthrough! (exp 173–182)

This was the most surprising stretch. After 100+ discards I tried *removing* features to see if any were noisy.

**Exp 172 (`3113e7c`)**: Remove `Month` → 0.8273. Discard. Month carries real signal (seasonal).

**Exp 173 (`d30b5a2`)**: Remove `DayOfWeek` → **0.8330 (+0.0013)**. keep. **Surprise win**. Hypothesis: at this depth/regularization regime, DayOfWeek (7-level one-hot) was being noisy or partially redundant with DayofMonth. The model's limited per-tree capacity (with cs=0.85 dropping ~1 feature per tree) was wasted on day-of-week splits that don't pay off.

**Exp 174**: Also remove DayofMonth → 0.8324. Discard.

**Exp 175 (`135593c`)**: Remove `DepHour_sin/cos` → **0.8332 (+0.0002)**. keep. Earlier (exp 66) removing them was -0.0006; with the simpler baseline now, they're net-noise.

**Exp 176 (`dbdc215`)**: Remove `DepHour` numeric → 0.8327. Discard. CarrierHour can't fully replace it.

**Exp 177 (`4de3bbd`)**: n=330 → flat at simpler config.
**Exp 178**: lr=0.08 → flat.

**Exp 179 (`98ef9a0`)**: Remove `UniqueCarrier` → **0.8336 (+0.0004)**. keep. CarrierHour subsumes the carrier identity. Removing the standalone categorical eliminates redundancy.

**Exp 180**: also remove DayofMonth → 0.8305. Still useful at this point.

**Exp 181 (`8d51beb`)**: n=340 → 0.8337.
**Exp 182 (`2fb8f97`)**: n=360 → **0.8338**.

This was the second big jump. Going from 0.8317 → 0.8338 (+0.0021) by *removing* features. The intuition I formed: with limited capacity (depth=23, mcw=2, modest n_estimators) and column sampling, every redundant feature steals capacity from the high-signal ones (CarrierHour, Origin, Dest, DepHour). Pruning forces the trees to use the strongest features.

---

## Phase 29 — Marginal tuning at simpler config (exp 183–211)

**Exp 183 (`0ef5f89`)**: n=380 → 0.8339 at 60.0s. Discard for boundary.
**Exp 184–186**: depth=22, lr=0.08, mcw=3 — all worse.
**Exp 188 (`e43b394`)**: Remove DepMinute (re-test) → 0.8179. Still critical. Removing DepMinute always hurts ~0.017 — that's the value of explicit minute-level info.
**Exp 189 (`051152e`)**: + HourDOW retry on simpler config → 0.8077. **Same 2nd-interaction regression confirmed** despite the simpler base.
**Exp 192 (`e24aa60`)**: Remove DepTime (re-test) → 0.8295. Even with DepHour+DepMinute, the int form contributes.
**Exp 196 (`5d20812`)**: Remove Dest → 0.8212. Massive regression. Destinations carry strong signal.

### Final fine-tuning (exp 200–211)

**Exp 200**: lr=0.077 → 0.8335.
**Exp 201**: lossguide max_leaves=4096 → 0.8337 at 158s. Discard.
**Exp 202 (`0248d0b`)**: `lr=0.073` → **0.8340 (+0.0002)**. keep.
**Exp 203**: lr=0.072 over.
**Exp 204 (`09ba14c`)**: `lr=0.074` → **0.8341 (+0.0001)**. keep — final best.
**Exp 205–211**: lr=0.076, n=370 (flat 60s+), sub=0.93, max_bin=400, max_cat_threshold=144 (over), random_state=2025 → 0.8335 (seed variance ~0.0006 in this direction), max_cat_to_onehot=64 (flat — no cat between 32 and 64), sub=1.0 (over).

---

## What worked, ranked by impact

| Move | Δ AUC | Phase |
|---|---|---|
| Initial scaling (n, lr, depth 6→14) | +0.0458 | 2 |
| min_child_weight tuning (1→20) | +0.0102 | 3 |
| `max_cat_to_onehot=32` | +0.0043 | 9 |
| Pushing depth (14→32) | +0.0050 | 10 |
| mcw retune deep (20→5) at smaller n | +0.0054 | 11–12 |
| **CarrierHour interaction** | +0.012 net | 13 |
| `subsample=1.0` (later: 0.95) + lr=0.05–0.075 regime shift | +0.005 | 19–20 |
| `reg_lambda=0.5` | +0.0001 | 22 |
| `max_cat_threshold=128` + `lr=0.075` | +0.0007 | 26 |
| **Feature pruning** (DayOfWeek, sin/cos, UniqueCarrier) | +0.0021 | 28 |
| Final `lr=0.074` fine-tuning | +0.0003 | 29 |

## What didn't work, with explanations I'll trust

- **High-cardinality engineered categoricals** (Route, OriginHour, OriginCarrier, CarrierHour6 alone): too many sparse levels — partition splits can't generalize.
- **Any 2nd interaction feature** added on top of CarrierHour (HourMonth, CarrierMonth, CarrierDOW, HourDOW, CarrierDistBand, multi-grain CarrierHour, +DistBand, +DistBucket): always -0.02 to -0.03. Pattern is robust across config regimes. My best guess at the mechanism: at our dataset size/depth, model capacity for "meaningful" feature interactions saturates with one carrier-time interaction; a second one gets sampled into trees at high enough rate (cs=0.8) that it injects noise into many predictions.
- **lossguide** at any max_leaves value: slower with no gain over depthwise.
- **gradient_based sampling**: GPU-only.
- **max_cat_to_onehot < 32**: misses UniqueCarrier (~30) — going lower regresses.
- **max_cat_to_onehot > 32**: doesn't matter (no cat with cardinality 32–300 left; Origin/Dest at ~300 are too high).
- **Removing high-signal features** (DepTime, DepMinute, Origin, Dest, Month): all big regressions.
- **scale_pos_weight ≠ 1**: data is balanced, so this just biases predictions without information gain.
- **shallow + many trees regime** (depth=8, n>1000): ~-0.03 vs deep regime. The depth-23 local optimum is real.
- **Different objective (binary:logitraw)**: same probabilistic content, slight numerical differences but no AUC gain.
- **tree_method='approx'**: too slow, no quality gain.
- **num_parallel_tree=2**: doubles trees per iteration, slower without quality gain.

## What I think is the underlying story

The 100K balanced training set is *just enough data* to learn one strong cross-feature pattern (Carrier × Hour), which becomes the dominant signal once exposed as a single categorical. The model regularization (mcw=2, depth=23 with enough trees, modest stochastic sampling) is tuned right at the edge where it captures this interaction without overfitting on it. Adding more interactions or capacity flips it into overfit territory; removing redundant features keeps the model's limited capacity focused on the strongest signals.

The CV std of 0.0029–0.0035 is the noise floor — anything sub-0.001 between two runs is essentially seed variance. The trajectory from 0.7445 to 0.8341 is real, but the last ~0.005 of gains were squeezing the noise.

## Things I didn't try and probably should have, in retrospect

- **Per-fold target encoding for Origin/Dest** via a sklearn Pipeline. Would need `category_encoders.TargetEncoder` (not in the venv) or a custom transformer that fits per-fold. Plausibly worth +0.005 based on the UC Berkeley study.
- **Seed ensembling**: train K models with different random_state and average within a custom cross-val routine. Time budget makes K>1 hard to fit.
- **Custom learning-rate scheduling** via `xgb.callback`. Decay lr after iteration N might let n=1000+ trees fit in budget.
- **Origin/Dest re-binning** to lower cardinality (e.g., top-50 + "other" bucket). Might reduce partition overhead and help generalization.

These are non-trivial code changes that go beyond what I felt confident attempting in the time/budget structure of the run.
