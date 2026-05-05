# Research log — may4

## Setup notes
- Branch: `may4` (from master HEAD `7d0b368`).
- Eval: 5-fold stratified CV on `2005-slice1-100k.csv` (CV_AUC).
- Data is balanced via undersampling (per `prepare.py`), so count-based features should be avoided.
- Categorical handling: `enable_categorical=True` with `pd.Categorical` using fixed `cat_levels` learned from train.
- Used template venv (`xgboost 3.1.3`) — system python had no ML packages.

## Experiment 1 — baseline (`7d0b368`)
- Hypothesis: establish reference performance with the unmodified starter.
- Config: `XGBClassifier(n_estimators=30, max_depth=6, learning_rate=0.1, enable_categorical=True)`.
- Result: **CV AUC 0.7445 ± 0.0043**, CV time 0.7s.
- Status: keep (baseline).
- Observations: very fast — plenty of headroom to scale `n_estimators` / tune `lr`. Std is tight (~0.004).

## Experiment 2 — more trees + lower lr (`74f86c5`)
- Hypothesis: baseline is under-fit (only 30 trees at lr=0.1 trained in 0.7s). More trees + slower lr is the standard XGBoost first move.
- Change: `n_estimators=30 → 200`, `learning_rate=0.1 → 0.05`. Everything else unchanged.
- Result: **CV AUC 0.7534 ± 0.0051**, CV time 3.2s. (+0.0089 vs baseline)
- Status: keep.
- Observations: clear improvement. CV time still tiny (3.2s) — lots of headroom remaining. Std slightly higher (0.0043 → 0.0051) but still small.

## Experiments 3–9 (compressed)
- Exp 3 (`235c8ab`): `n=500, lr=0.03` → 0.7556 (+0.0022). keep.
- Exp 4 (`72835a1`): `n=1000, lr=0.02` → 0.7573 (+0.0017). keep. Diminishing returns on tree budget.
- Exp 5 (`43b944b`): `max_depth=8` → 0.7623 (+0.0050). keep. Depth is a strong axis.
- Exp 6 (`fec243b`): `subsample=0.8, colsample_bytree=0.8` → 0.7635 (+0.0012). keep. Std also dropped.
- Exp 7 (`5b860e0`): add `DepHour`, `DepMinute` (decompose HHMM) → 0.7673 (+0.0038). keep. FE win.
- Exp 8 (`4be41a7`): add `Route=Origin_Dest` categorical → 0.7482 (-0.019). discard. Too many sparse levels overfit.
- Exp 9 (`89b6ba9`): `max_depth=10` → 0.7750 (+0.0077). keep. Depth is still the strongest axis.

## Synthesis after 9 experiments
- **What works**: deeper trees (6→10), more trees + slower lr, mild stochastic regularization (subsample/colsample), DepTime decomposition.
- **What doesn't**: high-cardinality engineered categoricals (Route).
- **Time budget**: CV time is now ~51s — very close to 60s ceiling. Further depth pushes risk timeout; need to compensate with `n_estimators` reduction.
- **Best so far**: 0.7750 at `n=1000, depth=10, lr=0.02, subsample=0.8, colsample=0.8` + `DepHour/DepMinute`.

## Experiments 10–29 (compressed)
- Exp 10–12 (`9490159`,`52e8665`,`adc2a58`): mcw 1→5→10→20 → 0.7843→0.7871→0.7875. keep, sweet spot at 20.
- Exp 13 (`5c15f06`): mcw=50 → 0.7840. discard.
- Exp 14–15 (`2a06041`,`1510f8b`): depth 10→12→14 → 0.7899→0.7903. keep, depth plateau.
- Exp 16 (`05d0a8d`): DepHour sin/cos → 0.7928. keep. Cyclical FE wins for numeric hour.
- Exp 17 (`a2b76e3`): cyclical Month/DOW → 0.7919. discard, redundant with cats.
- Exp 18 (`d496c51`): IsWeekend/IsRedeye → 0.7914. discard, redundant.
- Exp 19 (`27a52b4`): gamma=1 → 0.7927. discard, no effect.
- Exp 20–21 (`9385d86`,`9dffe08`): max_bin 256→512→1024 → 0.7934→0.7936. keep 512, 1024 is noise.
- Exp 22 (`934524b`): grow_policy=lossguide max_leaves=128 → 0.7872. discard.
- Exp 23 (`36885e9`): Carrier_Origin → 0.7938 but timed out (66s). discard.
- Exp 24 (`8ca9f4e`): depth=12+lr=0.015+n=1300 → 0.7927. discard.
- Exp 25 (`51f218b`): HourDOW interaction → 0.7816. discard.
- Exp 26 (`585f562`): reg_lambda=5 → 0.7925. discard.
- Exp 27 (`d3e9d53`): **max_cat_to_onehot=32** → **0.7977 (+0.0043)**. keep. Big win + 8s faster.
- Exp 28 (`1ed04ee`): max_cat_to_onehot=350 (Origin/Dest one-hot) → 0.7840. discard. High-card one-hot hurts.
- Exp 29 (`3864acd`): **max_depth=16** → **0.8002**. Broke 0.80!

## Synthesis after 29 experiments
- **Biggest wins**: depth (6→16), n_estimators (30→1000), min_child_weight (1→20), DepHour decomp+cyclical, **`max_cat_to_onehot=32`** (one-hot for low-card cats unlocked +0.0043).
- **Failed**: high-card categoricals (Route, Carrier_Origin big), redundant FE (cyclical Month/DOW, IsWeekend, HourDOW), gamma, lossguide, reg_lambda=5.
- **Key insight**: encoding strategy matters a lot. One-hot for low-cardinality cats >> partition-based default.
- **Best so far**: 0.8002 at `n=1000, depth=16, lr=0.02, mcw=20, sub/colsam=0.8, max_bin=512, max_cat_to_onehot=32` + DepHour decomp + sin/cos.

## Experiments 30–55 (compressed)
- Exp 30–33 (`1b60376`,`cc0f5ef`,`2ba4790`,`4d77e18`): depth 18→20→24→32 → 0.8017→0.8026→0.8035→0.8040. keep all, marginal.
- Exp 34 (`4d0ef90`): mcw=10 alone → 0.8121 (+0.0081) but 73.6s budget bust. discard.
- Exp 35 (`43f0c2c`): mcw=10 + n=800 → 0.8105 but 61s. discard.
- Exp 36 (`281df97`): mcw=10 + subsample=0.6 → 0.8062 but over budget. discard.
- Exp 37 (`078b95a`): mcw=10 + n=700 → 0.8092. keep, fits.
- Exp 38 (`6d20777`): mcw=5 + n=600 → 0.8141 but 68s. discard.
- Exp 39 (`95a5464`): mcw=5 + n=500 → 0.8126. keep.
- Exp 40 (`e18ee45`): mcw=3 + n=400 → 0.8139 but 66s. discard.
- Exp 41 (`5cbc182`): subsample=0.5 → 0.8026. discard, too aggressive.
- Exp 42 (`9ff1b1b`): depth=24 mcw=5 n=600 → 0.8134 but 62.9s. discard.
- Exp 43 (`46e415e`): remove DepTime → 0.8043. discard, info loss.
- Exp 44 (`4866e9c`): max_cat_threshold=256 → 0.8127 but 66.7s. discard.
- Exp 45–46: max_bin=256 / +n=550 → marginal. discard.
- Exp 47 (`2f8be9d`): colsample_bynode=0.8 → 0.8114. discard.
- Exp 48 (`2bd2911`): depth=24 mcw=3 n=500 → 0.8150 but 69.5s. discard.
- Exp 49 (`65e5d80`): depth=24 mcw=3 n=400 → 0.8130. keep.
- Exp 50 (`34e104b`): + CarrierHour → 0.8247 but 76s. discard.
- Exp 51 (`16e83e2`): + CarrierHour, n=300 → 0.8212 but 60.4s. discard.
- Exp 52 (`423a9a5`): + CarrierHour, n=250 → 0.8185. **keep, big win**.
- Exp 53 (`c5ca493`): n=280 → 0.8203. keep.
- Exp 54 (`5ab2073`): + CarrierMonth, n=200 → 0.7935. discard, big regression.
- Exp 55 (`5753050`): + HourMonth, n=240 → 0.7940. discard, big regression.

## Synthesis after 55 experiments
- **Best**: 0.8203 at `n=280, depth=24, lr=0.02, mcw=3, sub=0.8, cs=0.8, max_bin=512, max_cat_to_onehot=32` + features: DepHour decomp, sin/cos, **CarrierHour interaction**.
- **Strongest single discoveries**: max_cat_to_onehot=32 (+0.0043), mcw 20→10 (+0.0081), CarrierHour interaction (+~0.012 net).
- **Time wall**: deep (24+) trees + low mcw (3-5) + CarrierHour caps usable n_estimators around 280.
- **Pattern**: high-cardinality categoricals (Route, CarrierMonth, HourMonth) hurt despite low cardinality — likely the *2nd* interaction adds noise XGBoost can't ignore. Only ONE interaction works.

## Experiments 56–90 (compressed)
- Exp 56–58: mcw=2 over budget; lr=0.03→0.05 was huge (+0.0042 then +0.0017). lr=0.05 is sweet spot.
- Exp 60–61: n=350→380 → 0.8277→0.8279. keep, marginal.
- Exp 63–64: mcw=2 + n=300 → 0.8281. keep, current best.
- Exp 65–73: many tries (DistBand, remove sin/cos, remove DepMinute, max_delta_step, +CarrierMonth, +HourMonth, +CarrierDOW, OriginHour, cs=1.0, num_parallel_tree, gradient_based GPU-only crash, +CarrierDistBand, shallow regime, mcw=2.5, lr=0.06, remove max_cat_to_onehot, depth=22+n=320). All discarded.
- Exp 67: removing DepMinute → -0.017 (DepMinute critical).
- Exp 81: random_state=0 → 0.8284 vs 0.8281 (seed variance ~±0.0003 — minimal).
- Exp 89: removing max_cat_to_onehot → 0.8004 (-0.028, confirms it's critical).

## Synthesis after 90 experiments
- **Hard plateau** at 0.8281 since exp64. Most subsequent tweaks fall within ±0.001 noise range; CV std is 0.0035, so differences <0.001 are noise floor.
- **Pattern**: adding any 2nd interaction feature on top of CarrierHour causes huge regression (-0.02 to -0.03). XGBoost extracts one strong interaction well, second adds noise.
- **Stable findings**:
  - Deep trees (24+) + low mcw (2-3) + lr=0.05 + n~300 + max_cat_to_onehot=32 + CarrierHour is the locally optimal regime.
  - Each base feature contributes (DepMinute, DepHour, DepTime all critical).
  - Time budget hard-binds at 60s CV.
- **Compared to literature**: 0.8281 beats published CatBoost+Bayesian benchmarks (0.7793) and many ML approaches (0.7698).

## Experiments 91–136 (compressed) — late breakthrough
- Exp 92 (`4f48cb9`): subsample=1.0 + n=240 → 0.8283. keep — new regime.
- Exp 93–96: n=250→255, lr=0.06→0.07 → 0.8285→0.8286→0.8287→0.8293. keep all.
- Exp 98–100: n=290→320 → 0.8301→0.8303→0.8305. keep all. Broke 0.83!
- Exp 113 (`c02f7d6`): reg_lambda=0.5 → 0.8306. keep.
- Exp 115 (`5eb1a65`): n=330 → 0.8308. keep.
- Exp 119 (`24248f1`): max_depth=23 → **0.8309**. keep — current best.
- Exp 120–136: many tries (max_depth, lr fine, mcw, sub, cs, max_bin, max_cat_to_onehot, reg_alpha, reg_lambda, +CarrierDOW retry, lossguide retry, tree_method=approx, feature_weights, +DistBucket, +DepHourFrac, remove UniqueCarrier, remove DayofMonth) — all flat or worse.

## Synthesis after 186 experiments — CURRENT BEST = 0.8338 (simplification breakthrough)
- **Best (exp 182, `2fb8f97`)**: `n=360, depth=23, lr=0.075, mcw=2, sub=0.95, cs=0.85, max_bin=512, max_cat_to_onehot=32, max_cat_threshold=128, reg_lambda=0.5`.
  - cat_cols **= ["Month", "DayofMonth", "Origin", "Dest"]** (removed DayOfWeek, UniqueCarrier).
  - Engineered: DepHour, DepMinute, CarrierHour. (removed DepHour_sin/cos).
- **Surprise breakthrough (exp173-182)**: feature pruning kept finding wins.
  - exp 173 (remove DayOfWeek): 0.8317 → 0.8330 (+0.0013)
  - exp 175 (remove DepHour_sin/cos): 0.8330 → 0.8332 (+0.0002, simpler)
  - exp 179 (remove UniqueCarrier — subsumed by CarrierHour): 0.8332 → 0.8336 (+0.0004)
  - exp 181-182 (push n to 340, 360): 0.8336 → 0.8338
- **Why pruning helps**: DayOfWeek, UniqueCarrier, sin/cos were partially redundant with other features (CarrierHour subsumes Carrier, DepHour subsumes sin/cos for trees, day of week was apparently noisy at this regime). Simpler model uses limited capacity better.

## Final state (after 206 experiments) — best 0.8341
- Final config (exp 204, `09ba14c`):
  - cat_cols = ["Month", "DayofMonth", "Origin", "Dest"]
  - num_cols = ["DepTime", "Distance"]
  - Engineered: DepHour (int), DepMinute (int), CarrierHour (categorical interaction)
  - Hyperparams: `n=360, depth=23, lr=0.074, mcw=2, sub=0.95, cs=0.85, max_bin=512, max_cat_to_onehot=32, max_cat_threshold=128, reg_lambda=0.5`
  - CV time: ~59-60s, AUC 0.8341 ± 0.0029
- Trajectory: 0.7445 (baseline) → 0.7977 (max_cat_to_onehot=32) → 0.8126 (deep+mcw=5) → 0.8203 (CarrierHour) → 0.8281 (lr=0.05+sub=0.8) → 0.8310 (sub=0.95+lr=0.07+reg_lambda=0.5+depth=23) → 0.8317 (max_cat_threshold=128+lr=0.075) → **0.8338 (feature pruning: drop DayOfWeek/UniqueCarrier/sin/cos)** → 0.8341 (lr=0.074).
- Total improvement: 0.7445 → 0.8341 (+0.0896, +12.0%).
- Beats published CatBoost+Bayesian (0.7793), Stacking (0.7698), and most ML benchmarks I'm aware of.
- **Trajectory**: 0.7445 (baseline) → 0.7977 (max_cat_to_onehot=32) → 0.8126 (mcw=5+depth=32) → 0.8203 (CarrierHour) → 0.8281 (lr=0.05+sub=0.8) → 0.8310 (sub=0.95+cs=0.85+reg_lambda=0.5+depth=23+lr=0.07) → 0.8317 (max_cat_threshold=128+lr=0.075).
- **Strongest individual moves**: max_cat_to_onehot=32 (+0.0043), CarrierHour interaction (+~0.012), depth, mcw, lr/n_estimators tuning.
- **Persistent dead ends**: any 2nd interaction feature (HourMonth, CarrierMonth, CarrierDOW, OriginHour, HourDOW, DistBand, CarrierDistBand, OriginCarrier, DistBucket, CarrierHour6 added, multi-grain CarrierHour) always regresses by 0.002 to 0.03. Hypothesis: model has just enough capacity to learn one strong interaction; second adds noise XGBoost can't ignore at this dataset size.
- **CV variance ceiling**: random_state variance ~±0.0003. CV std is 0.0035. Most marginal tweaks beyond exp156 fall in noise.
- **Compared to literature**: AUC 0.8317 well beats published CatBoost+Bayesian (0.7793), Stacking ensembles (0.7698), and most ML benchmarks for this task.
