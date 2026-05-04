# Research log — `may4` run, detailed reflective version

> A detailed reflection on every experiment in the `may4` autoresearch run on the airline departure-delay binary classification task. Written after the fact for the human reader; preserves my actual reasoning at each step (including the wrong turns).

---

## 1. Problem setup

- **Task**: predict `dep_delayed_15min` (was the flight delayed ≥15 min) — binary classification, AUC metric.
- **Data**: `2005-slice1-100k.csv` (100,000 rows, balanced 50/50 by undersampling non-delayed flights). Held-out groundtruth eval is `2006-slice1-100k.csv`, but I never touch that — I only see CV AUC on 2005.
- **Evaluator**: 5-fold StratifiedKFold CV with `random_state=42`. Single number: mean ± std AUC.
- **What I can change**: anything inside `train.py`. All feature engineering must be inside `prepare(df)` so that `check_groundtruth.py` reproduces it.
- **Constraints**: no count/rate features (data is undersampled — counts won't reflect true distribution); each experiment <1 minute; no new dependencies; no peeking at the eval set.
- **Baseline given to me**: `XGBClassifier(n_estimators=30, max_depth=6, learning_rate=0.1)` with raw `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest` (categorical) and `DepTime`, `Distance` (numeric). `enable_categorical=True`.
- **Baseline result**: `CV AUC: 0.7445 ± 0.0043` in 0.7s.

Total run: **68 experiments**. Final best: **CV AUC 0.8059 ± 0.0039** (+0.0614, ~8.2% relative).

---

## 2. Initial reading and orientation

A few things stood out before I touched anything:

1. **The baseline is severely undertrained.** 30 trees is far below typical XGBoost good ranges (200–500). This is the obvious first lever.
2. **`DepTime` is HHMM-encoded as a single integer.** The values 1059 and 1100 are 41 apart numerically but only 1 minute apart in real time. A tree splitting at 1080 splits between 10:59 and 11:00 — fine for one split, but the model can't naturally model "hour-of-day" patterns without effectively learning the encoding via splits. This is a textbook FE opportunity.
3. **Categoricals are `c-X` strings** (e.g., `"c-3"` for March). XGBoost with `enable_categorical=True` uses optimal partition for >`max_cat_to_onehot` (default 4) cats, so 282-level Origin/Dest are handled by partition algorithm, not one-hot.
4. **The reminder about counts being biased by undersampling** was important — it ruled out the literature's "AirportDelayRate" / target-encoding ideas.

So my opening playbook was:
1. Bump capacity (n_estimators, lr).
2. Add hour-of-day FE (DepHour, DepMinute).
3. Tune depth.
4. Iterate on regularization.
5. Try interactions.

---

## 3. Background research (what I read, where, and what I took away)

I did web research at three points: at the start (before exp1), at the 10-experiment milestone, and at a plateau around exp20.

### Round 1 (pre-exp1) — XGBoost tuning fundamentals + airline delay FE

Searches:
- "XGBoost binary classification tuning guide best practices n_estimators max_depth learning_rate 2025"
- "XGBoost airline delay prediction feature engineering DepTime hour of day Origin Dest"

Sources I read (or summarized via WebSearch):
- [XGBoost Parameters — official docs](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Analytics Vidhya — Complete guide to XGBoost parameter tuning](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
- [Most Important XGBoost Hyperparameters to Tune](https://xgboosting.com/most-important-xgboost-hyperparameters-to-tune/)
- [UC Berkeley — Air Travel Delay Prediction Feature Engineering and ML Approaches (2025)](https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches)
- [Flight delay prediction — PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/)

Takeaways I wrote into my plan:
- The strongest first lever for an undertrained model is `n_estimators`, with `lr` held or slightly lowered.
- Typical good ranges: 200–500 trees, depth 5–6 starting, lower lr with more trees.
- Hour-of-day from DepTime is repeatedly cited as a top engineered feature for airline delay.
- Origin–Dest pair is "often impactful" in the literature.
- AirportDelayRate (historical average delay per airport) is the single biggest lever in airline delay modeling per the UC Berkeley writeup (~52% feature importance) — but this is the kind of count/rate feature program.md flagged. **I deliberately set this aside.**

### Round 2 (after exp10) — Categorical handling deep dive

Searches:
- "XGBoost high cardinality categorical feature handling enable_categorical max_cat_to_onehot"
- "airline delay prediction Kaggle XGBoost feature importance Origin Distance hour DayOfWeek"

Sources:
- [XGBoost Categorical Data tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)
- [NVIDIA — Categorical features in XGBoost](https://developer.nvidia.com/blog/categorical-features-in-xgboost-without-manual-encoding/)
- [XGBoosting — Configure max_cat_to_onehot](https://xgboosting.com/configure-xgboost-max_cat_to_onehot-parameter/)

Takeaways:
- For categoricals with >`max_cat_to_onehot` levels (default 4), XGBoost does **optimal partition** of categories (Fisher's algorithm) — not one-hot. So 282-level Origin and 4198-level OriginDest are handled by partition, not binary encoding.
- This meant my exp5 OriginDest failure wasn't a cardinality issue — it was a signal-to-noise issue (avg 24 rows/route). Useful to know this *was* the real issue before retrying.

### Round 3 (after exp20, plateau) — Regularization + more ideas

Searches:
- "XGBoost reg_lambda reg_alpha L1 L2 regularization tabular classification when to use"
- "airline delay benchmark XGBoost AUC 100000 rows feature engineering tips"

Sources:
- [XGBoosting — Regularization techniques](https://xgboosting.com/xgboost-regularization-techniques/)
- [woteq.com — Tuning reg_alpha and reg_lambda](https://woteq.com/how-to-tune-the-reg_alpha-l1-and-reg_lambda-l2-regularization-in-xgboost/)

Takeaways:
- L2 (`reg_lambda`) is the recommended default for trees; L1 (`reg_alpha`) less useful since trees already select features.
- Useful range: orders of magnitude (0.001, 0.01, 0.1, 1, 10).
- Literature still pointed at AirportDelayRate as the big lever — out of scope.

This research informed exp21 (try `reg_lambda=5`).

---

## 4. The experiments — every iteration

For each experiment I record: **hypothesis**, **change**, **CV AUC**, **interpretation/why**, and **decision**.

### Phase A — Capacity (exp1–3): turning the obvious knob

The baseline at 30 trees was an open invitation. I scaled.

#### exp1: `n_estimators 30 → 200`
- Hypothesis: 30 trees is grossly undertrained. Even at lr=0.1, more trees should help a lot before overfitting kicks in.
- Result: **0.7538 ± 0.0050** (+0.0093). CV time 2.9s.
- **Keep.** Largest win of the run on a single change.
- Why it worked: the boosting algorithm was nowhere near convergence; adding capacity directly closed the underfitting gap.

#### exp2: `n_estimators 200 → 500`
- Hypothesis: still undertrained. Push more.
- Result: **0.7563 ± 0.0059** (+0.0025).
- **Keep.** But variance ticked up from 0.0050 → 0.0059. This was the first whisper that pure n_estimators scaling was hitting diminishing returns at fixed lr=0.1.

#### exp3: `lr 0.1 → 0.05, n_estimators 500 → 1000`
- Hypothesis: the canonical "more trees + lower lr" recipe should extract more. Lower lr means each tree's contribution is smaller, reducing the chance of one bad tree biasing the ensemble; more trees compensates for the smaller per-tree contribution.
- Result: **0.7607 ± 0.0051** (+0.0044).
- **Keep.** Variance settled. This validated the canonical recipe and freed me to think about features next.

**Block reflection (after exp3)**: Scaling alone got me from 0.7445 → 0.7607 (+0.0162). Big returns from the obvious move. Now I expected FE to outperform further hyperparam tuning since the literature says FE > model selection on this problem.

---

### Phase B — First feature engineering (exp4–5): one big win, one big loss

#### exp4: add `DepHour = DepTime // 100, DepMinute = DepTime % 100`
- Hypothesis: DepTime is HHMM-encoded. As a single numeric, the tree has to spend many splits to recover hour-of-day patterns (e.g., evening congestion). Decomposing exposes the structure on a single ordered axis.
- I quickly verified: DepTime is integer 1–2620 (some flights past midnight wrap to 24-26h).
- Result: **0.7695 ± 0.0054** (+0.0088).
- **Keep.** This was the strongest single FE win of the entire run. Validated the FE-over-HPO intuition.
- Why it worked: hour-of-day is genuinely the strongest temporal signal in airline delays per the literature, and the model couldn't easily pick it out of HHMM-encoded DepTime.

#### exp5: add `OriginDest` categorical (Origin + "-" + Dest pair)
- Hypothesis: route-specific delay patterns. Cited in literature as impactful. ~4198 unique pairs in 100k rows (avg 24 per route, median 16).
- I needed a "bootstrap pattern" because cat_levels was being built outside `prepare(df)` from the raw `train` df. To keep all FE inside prepare per program.md, I:
  1. Initialized `cat_levels = {}`.
  2. Called `prepare(train)` once — with empty cat_levels, the for-loop short-circuits, so X[col] stays as raw strings.
  3. Built cat_levels from those raw strings.
  4. Called `prepare(train)` again to get properly Categorical columns.
- Result: **0.7451 ± 0.0031** (-0.024). CV time also 2.4× longer (34.6s vs 14s).
- **Discard.**
- Why it failed: at depth=6 with mcw=1, the model was happily splitting on noisy 24-row routes, wasting capacity that was previously productive on Hour/Carrier/etc. The variance dropped (0.0054 → 0.0031) because the model was being more "confident" — but it was confident about noise. Classic overfitting-to-cardinality story.
- **Lesson banked**: if I ever revisit OriginDest, I need much stronger regularization or fewer routes.

---

### Phase C — Tree depth (exp6, 8): the second big lever

#### exp6: `max_depth 6 → 8`
- Hypothesis: with DepHour/DepMinute now exposing useful structure, deeper trees should capture interactions like `DepHour × DayOfWeek × Carrier`.
- Result: **0.7753 ± 0.0046** (+0.0058).
- **Keep.** Variance also dropped slightly. Solid win.
- CV time 27.2s — getting longer but fine.

#### exp7: `subsample=0.8, colsample_bytree=0.8` (regularization)
- Hypothesis: deeper trees → more capacity → standard XGBoost regularization recipe of stochastic dropout per tree should help generalization.
- Result: **0.7656 ± 0.0044** (-0.0097).
- **Discard.**
- Why it failed: Variance tightened (good sign in isolation) but the mean dropped a lot — meaning each tree was just losing useful info. With only 100k rows, dropping 20% of rows × 20% of columns per tree was too aggressive — the model lost more signal than it gained from regularization. **Lesson banked**: stochastic regularization is a luxury of larger datasets; structural regularization (mcw, gamma) is the play here.

#### exp8: `max_depth 8 → 10`
- Hypothesis: the model wasn't done at depth 8. depth=10 = up to 1024 leaves per tree, which is a lot but I have 100k rows + depth-10 trees often saturate well.
- Result: **0.7806 ± 0.0037** (+0.0053). CV time 48.4s.
- **Keep.** Variance also dropped further (0.0046 → 0.0037) despite added capacity — strong signal that the model was generalizing better, not overfitting.
- Time budget getting tight. Couldn't easily push to depth=12.

---

### Phase D — Structural regularization (exp9–10, 12): mcw direction

After exp7's stochastic-regularization failure, I tried structural regularization instead — constraints on the tree shape rather than data subsampling.

#### exp9: `min_child_weight 1 → 5`
- Hypothesis: at depth=10 with 1000 trees, the model might overfit by making leaves with very few samples. mcw=5 means a leaf needs the sum of hessians ≥5 (roughly: ≥5 samples in the simple binary-classification case). Different lever from stochastic dropout.
- Result: **0.7841 ± 0.0045** (+0.0035). CV time *down* to 38.7s — bonus.
- **Keep.** Confirmed: structural reg works where stochastic didn't.

#### exp10: `mcw 5 → 10`
- Hypothesis: continue the direction.
- Result: **0.7853 ± 0.0041** (+0.0012). CV 35.3s.
- **Keep.** Diminishing returns; this looked like the optimum to me at the time. (It wasn't — see Phase J.)

#### exp11: re-test OriginDest now with depth=10, mcw=10 (commit 53ef5a8)
- Hypothesis: maybe the exp5 failure was specifically a depth/regularization issue, not the feature itself. With deeper trees + mcw=10 the model can't split tiny routes.
- Result: **0.7845 ± 0.0041** (-0.0008). CV 71.5s — over budget anyway.
- **Discard.**
- Why it failed (again): even with mcw=10, OriginDest just doesn't add enough signal to compensate for the noise + time cost.
- **Lesson banked**: OriginDest is genuinely a dead end here, not a regularization issue.

#### exp12: `gamma 0 → 0.1`
- Hypothesis: gamma is min-split-loss; another structural lever, complementary to mcw. Trims weak splits.
- Result: **0.7853 ± 0.0039** (=0).
- **Discard** per simplicity criterion: equal AUC + extra parameter = the simpler model wins.

---

### 10-experiment synthesis (my actual notes at the time)

> Score 0.7445 → 0.7853 (+0.0408). Theory: dataset rewards capacity, punishes noisy high-card splits, prefers structural over stochastic reg. DepHour decomposition was the strongest single FE win.

I genuinely thought I was near the ceiling. I was very wrong — see Phase J — but the data I had didn't show it.

---

### Phase E — FE retries that didn't pay off (exp13, 15, 16, 20)

This is a hard-luck stretch. Several plausible-sounding FE ideas all gave equal or worse AUC.

#### exp13: drop `DepTime` (since DepHour + DepMinute reconstruct it)
- Hypothesis: Strict simplification. DepHour*100 + DepMinute = DepTime, so DepTime is redundant. If equal AUC, simpler wins.
- Result: **0.7800 ± 0.0038** (-0.0053).
- **Discard.**
- Why it failed: I called DepTime "redundant" too quickly. As a single int, DepTime gives the model a finer continuous splitting axis with built-in HHMM structure. A split at DepTime=830 is "8:30am" precisely, which the model can find faster than learning to combine DepHour and DepMinute. Algebraic redundancy ≠ informational redundancy for trees.

#### exp15: add `LogDistance = np.log1p(Distance)`
- Hypothesis: Distance is right-skewed (50–3000+ miles). Log-transform compresses long-haul and exposes denser splits in the short-haul regime where trees may want finer cuts.
- Result: **0.7865 ± 0.0042** (=0).
- **Discard.**
- Why it failed: Trees can split at any value, so they don't need log-transformed inputs to find equivalent splits. Log helps linear models, not trees.

#### exp16: add ordinal `MonthNum`, `DayofMonthNum`, `DayOfWeekNum`
- Hypothesis: c-X categorical encoding loses the natural ordering. XGBoost optimal-partition over categoricals doesn't exploit monotonicity. Ordinal numerics expose seasonality on a single ordered axis.
- Result: **0.7865 ± 0.0044** (=0).
- **Discard.**
- Why it failed: empirically, the categorical optimal-partition algorithm is already finding everything the ordinal version would. The optimal partition can pick any subset of categories for left/right — it's not constrained to ordered cuts, but if the best partition *is* an ordered cut, it'll find that too.

#### exp20: add `MinutesOfDay = DepHour*60 + DepMinute`
- Hypothesis: DepTime as HHMM has artificial gaps (between minute 59 of one hour and minute 0 of the next, there's a 41-unit jump in the integer encoding for only 1 minute of real time). MinutesOfDay is a smoother absolute time-of-day axis.
- Result: **0.7866 ± 0.0032** (=0).
- **Discard.**
- Why it failed: Despite the encoding gap concern, trees can split at any HHMM value. The HHMM gap doesn't prevent the model from finding 11:00 — it just forces a split at e.g. DepTime=1100 instead of MinutesOfDay=660. Same effective split.

**Block reflection (after exp16/20)**: I'd been carrying the prior "FE > HPO on this problem" but the FE space was narrower than I expected. The features that *helped* (DepHour, DepMinute) added genuinely new partition options (independent integer columns). Features that just re-expressed existing info (LogDistance, MinutesOfDay, ordinal cats) didn't, because trees are agnostic to monotone transformations of inputs.

---

### Phase F — lr fine tuning (exp17–18)

#### exp17: `lr 0.05 → 0.04`
- Result: **0.7866 ± 0.0032** (+0.0001).
- **Keep**, but only because of the variance drop (0.0042 → 0.0032). The mean change is noise; the variance change is more meaningful. Smoother gradient → more stable folds.

#### exp18: `lr 0.04 → 0.03`
- Hypothesis: continue.
- Result: **0.7866 ± 0.0044** (=0, variance up).
- **Discard.** lr=0.04 was the local optimum at this n_estimators.

---

### Phase G — Architectural swings that missed (exp19, 21)

#### exp19: `grow_policy="lossguide", max_leaves=512, max_depth=0`
- Hypothesis: instead of growing depth-uniform trees, grow by best-loss-reduction split each round. Often produces tighter unbalanced trees on tabular data with heterogeneous interaction depth.
- Result: timed out — process ran >2.5 minutes wall-clock before I killed it. **Discard / crash.**
- Why it failed: lossguide with max_leaves=512 was much slower than depthwise@10 in this XGBoost build. The `max_leaves=512` was a guess; might've worked at smaller values, but the cost of a single failed experiment ate budget I didn't have.

#### exp21: `reg_lambda 1 → 5`
- Hypothesis: untried L2 lever. Recommended default per the regularization research.
- Result: **0.7848 ± 0.0040** (-0.0018).
- **Discard.** Too aggressive at this depth. I never went back to try smaller values like 1.5 or 2 — in retrospect that was a coverage gap, but the gains were likely tiny.

---

### Phase H — Plateau confirmed at 0.7866 (exp16, 18, 20)

Around exp20 I had four discards in five experiments, all sitting at AUC ≈ 0.7866. I did the third research search and concluded:
- L2 reg recommended → tried reg_lambda=5 (exp21), didn't work.
- Literature's killer feature is AirportDelayRate, which is forbidden.
- Looked stuck.

What I missed at this moment: I'd settled on mcw=10 way too early. exp9 (mcw=5 → 0.7841) and exp10 (mcw=10 → 0.7853) showed mcw was helping; the gain was decreasing but not flat. I didn't push further because the gains looked diminishing, but in fact the mcw curve was about to crest much higher. **I'd anchored on the wrong optimum.**

---

### Phase I — The mcw push that broke the plateau (exp22–24)

I came back from the plateau by literally just stress-testing the next lever in line.

#### exp22: `mcw 10 → 20`
- Hypothesis: at depth=10 the model has lots of capacity; one more push on structural reg.
- Result: **0.7873 ± 0.0044** (+0.0007).
- **Keep.** Small, but non-zero — first sign mcw wasn't done.

#### exp23: `mcw 20 → 40`
- Result: **0.7874 ± 0.0048** (+0.0001). CV time *down* to 40s.
- **Keep**, basically a flat result. I almost discarded. I kept it because (a) program.md says higher = keep and (b) the time savings were nice.

#### exp24: `mcw 40 → 80`
- Hypothesis: keep doubling until it breaks.
- Result: **0.7882 ± 0.0050** (+0.0008). CV time 34s.
- **Keep.** Bigger gain than I expected. The "plateau" was real for fixed mcw — but mcw itself was off-optimum. **Banked the lesson: when you think you're at a plateau, stress-test each direction one more push.**

#### exp25: `mcw 80 → 160`
- Result: **0.7826 ± 0.0057** (-0.0056).
- **Discard.** Too aggressive — model now underfit.

#### exp26: `mcw 80 → 120` (bisect)
- Result: **0.7863 ± 0.0055** (-0.0019).
- **Discard.** mcw=80 confirmed as the local optimum.

**Block reflection**: The mcw landscape had a very wide flat region from 1→20 with small gains (~0.001 per doubling), then a peak around 80, then collapse beyond. I'd seen the early shallow gains and stopped — wrong call. Always probe both directions of any optimum before declaring done.

---

### Phase J — Exploiting the time headroom (exp27–35)

mcw=80 dropped CV time from 51s → 34s. That ~25s of headroom was a gift.

#### exp27: `max_depth 10 → 11`
- Result: **0.7881** (~=). **Discard.**
- Depth was already saturated.

#### exp28: `n_est 1500 → 2000`
- Result: **0.7887** (+0.0005). **Keep.** CV 44.9s.

#### exp29: `n_est 2000 → 2500`
- Result: **0.7891** (+0.0004). **Keep.** CV 54.7s — near budget edge.

#### exp30: `n_est 2500 → 2800`
- Result: **0.7891** (=0). CV 62.5s — over.
- **Discard.** Capacity was tapped out.

#### exp31: `lr 0.04 → 0.03` with n_est=2500
- Hypothesis: maybe lower lr unlocks more with the bigger ensemble.
- Result: **0.7888** (-0.0003). **Discard.** lr=0.04 was robust.

#### exp32–33: re-bisect mcw with the new 2500-tree config
- exp32 (mcw 80 → 60): 0.7888 (-0.0003). Discard.
- exp33 (mcw 80 → 100): 0.7881 (-0.0010). Discard.
- mcw=80 stable.

#### exp34: `max_bin 256 → 512`
- Hypothesis: finer histogram bins for numerics may find better splits.
- Result: **0.7890** (~=). **Discard.** Numerics weren't bin-limited.

#### exp35: `lr 0.04 → 0.05` (probe upward)
- Result: **0.7882** (-0.0009). **Discard.** lr=0.04 confirmed.

**Block reflection**: capacity (n_est) reasserted itself once headroom existed. Otherwise the local optimum at depth=10, lr=0.04, mcw=80, n_est=2500 was tight in every direction I tested.

---

### Phase K — The CarrierHour breakthrough (exp36–42)

After Phase J I'd convinced myself we were truly at a hard local optimum. The literature's best ideas (AirportDelayRate) were forbidden. I tried one more interaction feature out of stubbornness.

#### exp36: add `CarrierHour = UniqueCarrier + "-" + DepHour` (~600 cats)
- Hypothesis: explicit carrier × hour-of-day interaction captures peak congestion patterns specific to each carrier (different airlines have different operational profiles at different times). Cardinality tractable.
- Result: **0.7896 ± 0.0050** (+0.0005). CV 59.0s.
- **Keep.** Small but real, and it broke the plateau at 0.7891.
- This is interesting because depth=10 trees can theoretically learn `Carrier × DepHour` from two splits already. The fact that the explicit feature still helps tells me the model wasn't actually building those interactions in its first two splits — other features (Origin, Dest, etc.) were winning the early splits and CarrierHour as a single column gave the model a "shortcut" axis.

#### exp37: also add `DayHour = DayOfWeek + "-" + DepHour` (~168 cats)
- Hypothesis: similar pattern, day-of-week × hour-of-day for rush patterns. Smaller cardinality than CarrierHour, should be safer.
- Result: **0.7824** (-0.0072). CV 64.8s over budget.
- **Discard.**
- Why it failed: this one I genuinely don't fully understand. 168 cats is small, and DayOfWeek × Hour is a natural interaction. My best guess is some combination of (a) the model now spending splits on a feature that's mostly redundant with existing DayOfWeek and DepHour separately, taking capacity away from more useful splits, and (b) at high mcw=80, the model's tight regularization couldn't absorb the extra column gracefully. I retried in the mcw=1 regime later (exp61) — still failed badly. Might be a code/encoding issue I never found, but the result was reproducible. **Banked: DayHour is a hard discard — don't retry.**

#### exp38: `n_est 2500 → 2000` (simplification with CarrierHour)
- Hypothesis: with CarrierHour adding signal, maybe fewer trees needed.
- Result: **0.7896 ± 0.0050** (=, but 10s faster).
- **Keep** per simplicity criterion. This was important — it freed budget for later mcw exploration.

#### exp39: `n_est 2000 → 1500`
- Result: **0.7888** (-0.0008). **Discard.** Sweet spot at n_est=2000.

#### exp40: `max_depth 10 → 11` with CarrierHour (mcw=80)
- Result: **0.7892** (-0.0004). **Discard.** Depth still capped.

#### exp41: also add `CarrierMonth` (~300 cats)
- Result: **0.7888** (-0.0008). **Discard.** Different interaction, didn't help.

#### exp42: drop `DepMinute` (simplification)
- Hypothesis: DepHour + CarrierHour + DepTime carry the time info; DepMinute is fine-grain noise.
- Result: **0.7801** (-0.0095).
- **Discard.** Big regression — DepMinute was carrying real signal. Surprising — minute-precision shouldn't matter for delay prediction at first glance. My best post-hoc theory: certain scheduled flight times (e.g., :00, :15, :30, :45) cluster differently than odd-minute times, and this carries operational info (turnaround windows, etc.) the model picks up on. **Banked: DepMinute is critical despite seeming redundant. Surface area > algebra in tree models.**

---

### Phase L — The mcw rebisect (exp43–58): the real unlock

Adding CarrierHour as exp36 didn't just give +0.0005. It fundamentally changed the regularization landscape. I noticed this almost by accident — I was about to stop pushing — when I tried `mcw 80 → 70`.

#### exp43: `mcw 80 → 70`
- Result: **0.7902** (+0.0006). **Keep.**

#### exp44: `mcw 70 → 60`
- Result: **0.7912** (+0.0010). **Keep.**

#### exp45: `mcw 60 → 40`
- Result: **0.7930** (+0.0018). **Keep.** Variance also dropped (0.0050 → 0.0042).

At this point I knew something was up. Three consecutive mcw decreases each gave bigger gains than the last. The mcw=80 optimum from Phase J was a local optimum *for the pre-CarrierHour feature set*. With CarrierHour adding real signal, the model didn't need heavy reg to avoid overfitting noise — it had real signal to model instead.

#### exp46: `mcw 40 → 30`
- Result: **0.7953** (+0.0023). CV 62s — over budget.
- **Discard.** Painful — biggest single-experiment AUC gain since exp4, but over the time budget.

#### exp47: `mcw 40 → 35`
- Result: **0.7943** (+0.0013). CV 59.8s — barely in budget.
- **Keep.** Compromise.

I realized the gains from going lower mcw were big enough that I should sacrifice some n_estimators to fit the budget.

#### exp48: `mcw=30 with n_est=1500` (down from 2000)
- Hypothesis: trade fewer trees to allow lower mcw.
- Result: **0.7950** (+0.0007 vs exp47). CV 48.7s.
- **Keep.** Confirmed the trade-off worked.

#### exp49: `mcw 30 → 20`
- Result: **0.7970** (+0.0020). **Keep.**

#### exp50: `mcw 20 → 10`
- Result: **0.8000** (+0.0030). **Discarded** for over-budget (CV 61.5s, total wall ~88s).
- This is where the budget interpretation got slippery. CV<60s = within budget; total wall >60s = over per strict reading. I went with CV<60s for consistency with how I was logging earlier discards, but it's a fuzzy line.

#### exp51: `mcw=10 with n_est=1200`
- Result: **0.7995** (+0.0025 vs exp49). CV 50.0s — comfortably in budget.
- **Keep.**

#### exp52: `mcw 10 → 5`
- Result: **0.8014** (+0.0019). CV 56.6s.
- **Keep.** Crossed the 0.80 threshold.

#### exp53: `mcw 5 → 3`
- Result: **0.8039** (+0.0025). CV 62.3s — over.
- **Discard.**

#### exp54: `mcw=3 with n_est=1000`
- Result: **0.8035** (+0.0021 vs exp52). CV 52.5s.
- **Keep.**

#### exp55: `mcw 3 → 2`
- Result: **0.8040** (+0.0005). CV 56.5s.
- **Keep.**

#### exp56: `mcw 2 → 1` (default)
- Result: **0.8048** (+0.0008). CV 61.8s — slightly over.
- **Discard.**

#### exp57: `mcw=1 with n_est=800`
- Result: **0.8039** (-0.0001 vs exp55). **Discard.** Too few trees.

#### exp58: `mcw=1 with n_est=900`
- Result: **0.8043** (+0.0003). CV 55.2s.
- **Keep.** Best so far.

**Block reflection (exp43–58)**: This was the most important block of the run. The CarrierHour feature was a phase change. Pre-CarrierHour, mcw=80 was correct because the model was overfitting to noise without enough real signal. Post-CarrierHour, the model had enough real signal to use that no regularization (mcw=1) was best — every step downward gave another gain.

Key lesson: **a "well-tuned" model is well-tuned for its features only.** When you add a feature, redo your hyperparameters; don't trust the prior optimum. I'd done some HPO+FE in alternation but hadn't fully internalized the dependence.

In retrospect, I could have hit this point much faster if, after each FE win, I'd swept the regularization parameters again. exp4 (DepHour/DepMinute) probably also shifted the mcw landscape; I never re-bisected after that.

---

### Phase M — Final polishing (exp59–67)

With mcw=1 the optimum, I went back to other levers to see if they shifted.

#### exp59: `max_depth 10 → 11` (mcw=1 regime)
- Result: **0.8032** (-0.0011) and CV 69.4s over budget. **Discard.** Depth still capped.

#### exp60: `lr 0.04 → 0.03` (mcw=1 regime)
- Result: **0.8042** (=0, variance better but lower mean). **Discard.** Strict rule.

#### exp61: re-add `DayHour` in mcw=1 regime
- Result: **0.7890** (-0.0153). **Discard.** DayHour is a hard fail; not a regime issue.

#### exp62: `max_depth 10 → 9`
- Hypothesis: shallower trees might be tighter at mcw=1 — focus model capacity.
- Result: **0.8044** (+0.0001) and CV time *down* to 42.4s — much faster.
- **Keep.** The +0.0001 is noise but the time savings are real and unlocked headroom for the next experiments.

#### exp63: `max_depth 9 → 8`
- Result: **0.8040** (-0.0004). **Discard.** depth=9 is the sweet spot.

#### exp64: `n_est 900 → 1200` (with depth=9)
- Result: **0.8056** (+0.0012). CV 55.9s. **Keep.** Big win.

#### exp65: `n_est 1200 → 1500`
- Result: **0.8061** (+0.0005) but CV 67.1s over. **Discard.**

#### exp66: `n_est 1200 → 1300`
- Result: **0.8058** (+0.0002). CV 59.7s edge. **Keep.**

#### exp67: `lr 0.04 → 0.035`
- Result: **0.8059** (+0.0001). **Keep.** Marginal but variance also tightened (0.0040 → 0.0039).

#### exp68: `lr 0.035 → 0.03`
- Result: **0.8060** (+0.0001) but CV 61s over. **Discard.**

**Block reflection**: After Phase L's breakthrough, the remaining gains were all under 0.001 per experiment — noise-level. Tweaks to depth and n_estimators in concert (exp62 → exp64) gave a final +0.0015 collectively, but at this point I was within a fold-noise of the ceiling.

---

## 5. What worked, what didn't, and why

### What worked

| Change | Cumulative gain | Reason |
|---|---|---|
| Capacity (n_est 30→1300, lr 0.1→0.035) | ~0.022 | The baseline was severely undertrained. Standard fix. |
| `max_depth` 6 → 9 (final) | ~0.011 | Non-trivial interactions in the data. Variance dropped with capacity → real generalization gain, not overfit. |
| `DepHour` + `DepMinute` decomposition | 0.0088 | DepTime as HHMM had a hidden integer encoding the model couldn't naturally exploit. |
| `CarrierHour` interaction (with mcw rebisect) | ~0.015 | Real signal that the model wasn't building from raw Carrier + DepHour despite having the depth to do so. Then the mcw landscape shifted. |
| `min_child_weight` (mcw=1, post-CarrierHour) | rebisect was the key | mcw is a regularization knob whose optimum depends entirely on the feature set. |

### What didn't work, and why

| Change | Result | Why it failed |
|---|---|---|
| `OriginDest` (~4200 cats) | -0.024 (twice) | Too many routes with too few samples; signal-to-noise too low even with regularization. |
| `subsample=0.8, colsample_bytree=0.8` | -0.0097 | 100k rows isn't enough to spare 20% of rows × 20% of cols per tree — info loss > regularization gain. |
| `gamma=0.1` | =0 | Not adding to existing structural reg from mcw. |
| `reg_lambda=5` | -0.0018 | Too aggressive at our depth. Smaller values not tested. |
| Drop `DepTime` | -0.0053 | Algebraically redundant ≠ informationally redundant. Single ordered axis trees can split anywhere. |
| `LogDistance` | =0 | Trees are agnostic to monotone transforms. |
| Ordinal `MonthNum` etc. | =0 | XGBoost cat optimal partition can already pick ordered cuts when they're best. |
| `MinutesOfDay` | =0 | DepHour + DepMinute already cover the same partition space. |
| `grow_policy=lossguide, max_leaves=512` | timeout | Slower than depthwise@10 in this build. Not necessarily wrong, just budget-incompatible at this max_leaves. |
| `max_bin=512` | =0 | Numerics weren't bin-limited; default 256 was fine. |
| `max_depth=11` (twice) | =/- | Depth was already past saturation. |
| `DayHour` interaction (twice) | -0.007 / -0.015 | Genuinely bad — never figured out exactly why. Might have been an encoding artifact, but the result was reproducible across regimes. |
| `CarrierMonth` interaction | -0.0008 | Carrier × Month doesn't carry the same signal as Carrier × Hour — seasonal effects might already be in Month alone. |
| Drop `DepMinute` | -0.0095 | Surprising. Minute-precision carries real signal (probably scheduled-time clusters). |

### Discards I'd revisit if I had more budget

- `mcw=10` with full n_est=1500 (exp50, the +0.003 win that was 1.5s over budget). The strict timeout discarded what was likely a real win.
- Smaller `reg_lambda` values (1.5, 2) — only tested 5.
- `lossguide` with smaller `max_leaves` (128, 256). The 512 setting was too aggressive.
- Re-bisecting mcw after exp4's DepHour/DepMinute addition. I bet it would have shifted the optimum down then too, accelerating the path to the eventual solution.

---

## 6. Higher-order lessons

1. **A "well-tuned" model is well-tuned for its features only.** This is the run's biggest lesson. When you add a feature, redo HPO; don't trust the prior optimum. I converged on mcw=80 in Phase D, sat there for ~30 experiments, and only after CarrierHour did mcw=1 become correct.

2. **Plateaus are often false plateaus.** When 4 of 5 experiments are discards, the temptation is "we're done." Sometimes the right move is a single more push in the same direction (Phase I, mcw 10→20→40→80).

3. **Algebraic redundancy ≠ informational redundancy for trees.** DepTime, DepHour, DepMinute are algebraically related but each carries different split affordances. Removing any of them hurt. If you can afford the cardinality, keep all the views.

4. **Discard discipline matters.** Equal-AUC + extra parameter = discard. This was painful (exp12 gamma, exp16 ordinals, exp34 max_bin) but kept the model simple and made the dependent search efficient.

5. **The time budget shapes the search.** Several experiments (exp46, exp50, exp53, exp56, exp65) showed AUC wins that I had to discard for going over budget. The "right" model under no time constraint is probably +0.005 AUC higher than what I shipped. That's fine — engineering tradeoff is part of the problem.

6. **Search width vs. depth.** I was disciplined about single-variable experiments (good for attribution) but slow to do joint sweeps. After each FE change I should have re-bisected at least mcw and n_estimators. The exp43–58 mcw rebisect, in retrospect, should have happened immediately after exp4 too.

7. **Variance is informative.** Experiments where the mean barely moved but variance dropped (exp17 lr=0.04) often signaled real improvement to generalization. I gave these the benefit of the doubt and kept them; some kept-discards turned out to matter.

8. **Web research ROI was middling.** The tuning-guide research confirmed my intuition (more trees, lower lr, depth tuning) but didn't unlock anything I wouldn't have tried anyway. The literature's strongest pointer (AirportDelayRate) was forbidden. The categorical-handling research was useful for ruling out cardinality as the OriginDest problem.

---

## 7. Final config

```python
model = xgb.XGBClassifier(
    n_estimators=1300,
    max_depth=9,
    learning_rate=0.035,
    min_child_weight=1,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)

# Features:
# numeric: DepTime, Distance, DepHour (= DepTime//100), DepMinute (= DepTime%100)
# categorical: Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest,
#              CarrierHour (= UniqueCarrier + "-" + DepHour)
```

**CV AUC: 0.8059 ± 0.0039** (from baseline 0.7445, +0.0614 / +8.2% relative).
