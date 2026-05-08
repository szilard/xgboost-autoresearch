# Detailed Research Log After Run

Run tag: `may7`

Branch: `may7`

Best valid commit at end of this session: `00d0b0f`

Best valid CV result: `0.7989 +/- 0.0051`

Baseline CV result: `0.7445 +/- 0.0043`

Net improvement over baseline: `+0.0544` CV AUC

Important guardrails followed:
- I only modified `train.py`.
- I did not read or run `check_groundtruth.py`.
- I did not read or run `run_groundtruth_all.sh`.
- I did not use any held-out ground-truth feedback.
- `results.tsv` and this research log are intentionally untracked bookkeeping files.
- All feature engineering stayed inside `prepare(df)` in `train.py`.

## Final Best Model

The final kept `train.py` at commit `00d0b0f` uses:

```python
model = xgb.XGBClassifier(
    n_estimators=260,
    max_depth=0,
    learning_rate=0.0465,
    min_child_weight=12,
    subsample=0.90,
    colsample_bytree=0.65,
    tree_method="hist",
    grow_policy="lossguide",
    max_leaves=704,
    eval_metric="auc",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
```

The final feature set keeps the original categorical variables:

- `Month`
- `DayofMonth`
- `DayOfWeek`
- `UniqueCarrier`
- `Origin`
- `Dest`

It keeps `Distance` as numeric.

It removes raw `DepTime` as a direct model input, but derives:

- `DepHour`
- `DepMinute`
- `DepMinutes`
- `DepTimeSin`
- `DepTimeCos`

The strongest final pattern was not a feature trick but a training-shape change: use high-capacity lossguide trees with a carefully tuned leaf cap and relatively few boosting rounds.

## External Research Sources Used

The external research was used as a design input throughout, not as a source of data or scores. The main sources were official documentation plus high-level flight-delay feature searches.

### XGBoost Parameter Documentation

Source: https://xgboost.readthedocs.io/en/stable/parameter.html

Information used:
- `learning_rate` / `eta` shrinks boosting updates, so lower learning rates usually need more boosting rounds.
- `max_depth`, `min_child_weight`, `gamma`, `reg_alpha`, and `reg_lambda` are the main tree-complexity and regularization controls.
- `subsample`, `colsample_bytree`, `colsample_bylevel`, and `colsample_bynode` add stochastic row/column sampling.
- `tree_method="hist"` is the fast histogram tree builder.
- `grow_policy="lossguide"` and `max_leaves` are available with histogram trees and change capacity allocation from level-wise depth growth to leaf-wise loss-driven growth.
- `max_bin` trades histogram split resolution against speed/memory.
- `max_cat_to_onehot` and `max_cat_threshold` affect categorical split behavior.

How it influenced the work:
- Experiments 1-10 started with ordinary depth/round/child-weight tuning.
- Experiments 11-13 tried categorical controls and split regularization.
- Experiments 31-40 tuned column and row sampling.
- Experiments 41-80 heavily explored `grow_policy="lossguide"` and `max_leaves`.
- Experiments 61-64 revisited `gamma`, `reg_lambda`, and `colsample_bynode`.
- Experiments 81 onward tested runtime/capacity boundary ideas such as `max_bin`.

### XGBoost Categorical Data Tutorial

Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

Information used:
- Native categorical support expects dataframe columns to have categorical dtype and `enable_categorical=True`.
- XGBoost can use one-hot splits for low-cardinality categoricals or partition-based categorical splits.
- `max_cat_to_onehot` controls the threshold for one-hot behavior.
- Category partitioning can be useful for categorical features, especially higher-cardinality features.

How it influenced the work:
- The baseline already used pandas categorical dtype, so I preserved that structure.
- Experiment 11 tested `max_cat_to_onehot=32`.
- Experiments 12, 67, and 68 tested `max_cat_threshold`.
- The conclusion was that default native categorical partitioning was best. Forcing one-hot was very harmful, and changing category thresholds was neutral, worse, or too slow.

### XGBoost Tree Methods Documentation

Sources:
- https://xgboost.readthedocs.io/en/stable/treemethod.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

Information used:
- Histogram tree building is the right performance baseline for tabular data here.
- Loss-guided growth can allocate splits to high-loss nodes rather than growing level by level.
- `max_leaves` becomes a natural capacity knob when using lossguide.

How it influenced the work:
- `tree_method="hist"` was set early and kept.
- Experiment 41 first tried lossguide.
- Experiment 41 was high-AUC but invalid on total runtime, which led to a constrained lossguide search.
- Experiments 42-90 became a systematic search over leaf caps, rounds, learning rate, and regularization under the timeout.

### XGBoost Scikit-Learn Estimator / Early Stopping Documentation

Sources:
- https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/stable/python/python_intro.html

Information used:
- XGBoost supports early stopping in the scikit-learn API.
- Early stopping would need an evaluation set inside each CV fold and would produce fold-dependent best iteration counts.

How it influenced the work:
- I did not add early stopping. It would have made the CV harness more complex and less comparable to earlier fixed-round experiments.
- The experiment loop was already close to the timeout, and the biggest gains came from explicit fixed-round tuning.

### Scikit-Learn Cyclical Feature Engineering Example

Source: https://scikit-learn.org/1.5/auto_examples/applications/plot_cyclical_feature_engineering.html

Information used:
- Periodic features can be represented with sine/cosine transforms so the model can see wraparound continuity.
- Time-like variables often have artificial discontinuities if encoded as raw ordinal values.

How it influenced the work:
- Experiment 14 added `DepTimeSin` and `DepTimeCos`.
- I also added `DepHour`, `DepMinute`, and `DepMinutes`.
- This was one of the most important feature-engineering wins.

### Flight Delay Feature Searches

Search theme:
- Flight delay prediction feature engineering.
- Airline delay models using time, day, season, distance, carrier, origin, and destination.

Information used:
- Common predictors include departure time, day/week/month, route/airport, carrier, and distance.
- The dataset already had those raw schedule/spatial variables.

How it influenced the work:
- Time-of-day engineering was prioritized before exotic features.
- Calendar cyclic features were tried and discarded.
- Route interaction was tried and discarded.
- Distance transforms were tried later but timed out under the high-capacity model.

## High-Level Timeline

### Block 0: Setup And Baseline

The baseline was the untouched starter model:

- 30 trees
- depth 6
- learning rate 0.1
- native categorical support
- 5-fold CV

Baseline result: `0.7445 +/- 0.0043`.

The baseline looked underpowered because 30 trees is small for a feature space with categorical airline, origin, and destination features. The first hypothesis was therefore simple: increase boosting rounds, lower the learning rate, and add row/column sampling and leaf-mass regularization.

### Block 1: Experiments 1-10, Depth And Child-Weight Capacity

Main question:

Can a conventional depthwise XGBoost model improve by using more rounds and deeper interactions while guarding against overfit with `min_child_weight`?

What worked:
- More trees with lower learning rate worked immediately.
- Increasing depth from 4 to 8 was highly effective.
- Pairing depth increases with higher `min_child_weight` was important.

What did not work:
- Depth 9 with `min_child_weight=60` was worse.
- Too low a child weight at depth 7 overfit.

Learning:

The data needed richer route/time/carrier interactions than the baseline could express. Depthwise trees improved until about depth 8. After that, extra depth stopped helping. The best result in this block was `0.7736`.

### Block 2: Experiments 11-20, Categorical Controls And First Feature Engineering

Main question:

Can categorical split controls or simple feature engineering improve the depthwise model?

What worked:
- Departure-time feature engineering was a large win.
- Removing raw `DepTime` after deriving cleaner time features was a simplification win.
- Slower schedules after adding time features continued to help.

What did not work:
- `max_cat_to_onehot=32` was very bad.
- `max_cat_threshold=256` was neutral/slightly worse.
- `gamma=0.1` added no rounded gain.
- Calendar numeric/cyclic features hurt.
- Explicit `Origin_Dest` route feature hurt and increased runtime.
- Shallower depth after adding time features underfit.
- Larger `max_bin` did not help.

Learning:

Raw HHMM `DepTime` was a poor representation. Converting it into hour, minute, absolute minutes, and cyclic encodings gave the first major feature-engineering improvement. However, most additional feature engineering did not help; XGBoost's native categorical handling was already strong for calendar and airport fields.

### Block 3: Experiments 21-40, Slower Schedules And Sampling

Main question:

After time features, how much smoother boosting and stochastic sampling does the depthwise model want?

What worked:
- Increasing to 1000 trees with lower learning rate improved gradually.
- Lowering `min_child_weight` from 30 to around 18 improved the 1000-tree model.
- Stronger `colsample_bytree` improved substantially.
- `subsample=0.90` was slightly better than 0.85.

What did not work:
- More extreme child-weight values hurt.
- `colsample_bytree=1.0` hurt.
- Too low or too high row subsampling hurt.

Learning:

The best depthwise model wanted:

- many trees,
- relatively low learning rate,
- depth 8,
- moderate leaf flexibility,
- substantial tree-level column sampling,
- mild row sampling.

Best result by experiment 40: `0.7821`.

### Block 4: Experiments 41-60, Lossguide Breakthrough

Main question:

Can lossguide growth allocate tree capacity better than fixed depthwise growth?

What worked:
- `grow_policy="lossguide"` was the largest training-shape breakthrough.
- High leaf caps were much better than shallow/small leaf caps.
- Trading fewer rounds for larger trees worked extremely well.
- The useful sequence was roughly:
  - 96 leaves: modest gain
  - 112 leaves: better
  - 128 leaves: better
  - 144 leaves: better
  - 160 leaves: better
  - 192 leaves: better
  - 256 leaves: better
  - 384 leaves: much better
- At 384 leaves, smoother boosting with 300-390 trees improved further.
- `min_child_weight=12` beat 18 in the high-leaf setting.

What did not work:
- 128 leaves with 1000 trees was high-AUC but too slow.
- 64 leaves underfit badly.
- 400 trees at 384 leaves timed out.
- 395 trees underperformed 390.
- Too high `min_child_weight` underfit.
- Too low `min_child_weight` overfit or timed out.
- `gamma`, `reg_lambda`, and `colsample_bynode` did not improve.

Learning:

The depthwise model was not using capacity as effectively as lossguide. Once lossguide was introduced, the best strategy was not more small trees, but fewer larger trees with carefully controlled leaf count and child weight.

Best result by experiment 60: `0.7964`.

### Block 5: Experiments 61-70, Regularization Attempts And More Leaf Capacity

Main question:

Can regularization stabilize the high-capacity lossguide model, or should capacity continue increasing?

What worked:
- Increasing leaf cap from 384 to 448, 512, and then 640/704 kept improving.
- The 704-leaf region became the new best.

What did not work:
- `gamma=0.05` tied but added complexity.
- `reg_lambda=0.5` hurt.
- `reg_lambda=2.0` hurt.
- `colsample_bynode=0.8` hurt.
- Distance transforms timed out.
- Changing categorical threshold either timed out or underperformed.
- `max_depth=16` cap hurt.
- `max_bin=128` hurt.

Learning:

The model did not want additional generic regularization. It wanted more lossguide leaf capacity, but the timeout began to dominate. Any feature addition or expensive categorical change could push the run over the limit.

Best result by experiment 70: `0.7973`.

### Block 6: Experiments 71-80, Narrow 704-Leaf Local Optimum

Main question:

Where is the best leaf/round/learning-rate balance near the high-capacity boundary?

What worked:
- 640 leaves with 250 trees improved.
- 704 leaves with 250 trees improved slightly.
- 704 leaves with 260 trees and learning rate 0.0465 improved again.

What did not work:
- 768 leaves underperformed.
- 265 trees timed out.
- Lower child weight timed out.
- Higher child weight hurt.
- Depth cap hurt.

Learning:

The best configuration became a narrow, runtime-constrained local optimum:

- 260 trees
- learning rate 0.0465
- 704 leaves
- child weight 12

Best result by experiment 80: `0.7989`.

### Block 7: Experiments 81-90, Local Exhaustion Around Best

Main question:

Can low-cost changes improve or simplify the 704-leaf model?

What worked:
- Nothing beat `00d0b0f`.

What did not work:
- Lower `max_bin` hurt.
- Lower column sampling timed out.
- Higher row sampling timed out.
- 672 leaves underperformed.
- 736 leaves underperformed.
- Schedule interpolations underperformed.
- Removing `DepMinute` timed out.
- Learning-rate nudges around 0.0465 underperformed.

Learning:

The 704-leaf model is very tight. Nearby HPO changes either lose AUC or exceed the timeout. At the end of the run, further progress likely requires a genuinely new idea that preserves runtime, not another small local interpolation.

## Iteration-By-Iteration Ledger

### Baseline

- Commit: `7d0b368`
- CV AUC: `0.7445 +/- 0.0043`
- Status: keep
- Change: untouched starter.
- Thinking: establish the reference before any modification. The starter used 30 trees, depth 6, learning rate 0.1, and native categorical handling.
- Interpretation: good enough to prove the loop works, but likely underfit because 30 trees is small for categorical route/carrier/time effects.

### Experiment 1

- Commit: `ba8f26d`
- CV AUC: `0.7487 +/- 0.0046`
- Status: keep
- Change: 200 trees, depth 4, learning rate 0.05, `min_child_weight=5`, `subsample=0.85`, `colsample_bytree=0.85`, `tree_method="hist"`, `eval_metric="auc"`.
- Thinking: based on XGBoost parameter docs, pair a lower learning rate with more rounds and add stochastic sampling/leaf-mass regularization.
- Interpretation: improved by `+0.0042`; the baseline was underpowered.

### Experiment 2

- Commit: `2ec6b62`
- CV AUC: `0.7496 +/- 0.0048`
- Status: keep
- Change: 400 trees and learning rate 0.03.
- Thinking: test whether experiment 1 was still underfit.
- Interpretation: smaller gain, but still positive. Smoother boosting helped.

### Experiment 3

- Commit: `d84eab8`
- CV AUC: `0.7579 +/- 0.0048`
- Status: keep
- Change: depth 5 and `min_child_weight=10`.
- Thinking: allow deeper carrier/origin/destination/time interactions while increasing leaf-mass guard.
- Interpretation: large jump. Interaction depth mattered.

### Experiment 4

- Commit: `49206b8`
- CV AUC: `0.7657 +/- 0.0051`
- Status: keep
- Change: depth 6 and `min_child_weight=20`.
- Thinking: continue depth/regularization sweep.
- Interpretation: another large jump. The model still needed deeper interactions.

### Experiment 5

- Commit: `23da8fc`
- CV AUC: `0.7691 +/- 0.0053`
- Status: keep
- Change: depth 7 and `min_child_weight=40`.
- Thinking: test whether even richer interactions help, with stronger guard against sparse leaves.
- Interpretation: improved, but marginal gain was smaller. Depth was approaching its useful boundary.

### Experiment 6

- Commit: `dfd242a`
- CV AUC: `0.7702 +/- 0.0052`
- Status: keep
- Change: depth 7 with child weight relaxed from 40 to 20.
- Thinking: check whether the depth 7 model was over-regularized.
- Interpretation: improved. The previous 40 child weight was too conservative.

### Experiment 7

- Commit: `922a158`
- CV AUC: `0.7686 +/- 0.0046`
- Status: discard
- Change: child weight 20 to 10 at depth 7.
- Thinking: test further relaxation.
- Interpretation: overfit or unstable leaves. Too much flexibility hurt.

### Experiment 8

- Commit: `408922b`
- CV AUC: `0.7704 +/- 0.0049`
- Status: keep
- Change: child weight 15 at depth 7.
- Thinking: interpolate between 20 kept and 10 discarded.
- Interpretation: slight improvement. The best child-weight region was around 15-20 at depth 7.

### Experiment 9

- Commit: `9566bf1`
- CV AUC: `0.7736 +/- 0.0052`
- Status: keep
- Change: depth 8 and `min_child_weight=30`.
- Thinking: test one more depth increase with stronger leaf-mass regularization.
- Interpretation: clear improvement. Depth 8 was useful.

### Experiment 10

- Commit: `1a8092b`
- CV AUC: `0.7722 +/- 0.0052`
- Status: discard
- Change: depth 9 and `min_child_weight=60`.
- Thinking: continue depth sweep but guard heavily.
- Interpretation: worse. The useful depthwise boundary was around depth 8.

### Experiment 11

- Commit: `bbaf29f`
- CV AUC: `0.7620 +/- 0.0052`
- Status: discard
- Change: `max_cat_to_onehot=32`.
- Thinking: XGBoost categorical docs suggested low-cardinality categorical features can be one-hot split. I expected Month/Day/Carrier might benefit.
- Interpretation: very harmful. Partition-based native categorical splitting was much better than forcing one-hot behavior.

### Experiment 12

- Commit: `06ab09b`
- CV AUC: `0.7735 +/- 0.0053`
- Status: discard
- Change: `max_cat_threshold=256`.
- Thinking: allow richer high-cardinality Origin/Dest category partitions.
- Interpretation: essentially neutral but not better. Default threshold was adequate in the depthwise model.

### Experiment 13

- Commit: `8e13809`
- CV AUC: `0.7736 +/- 0.0053`
- Status: discard
- Change: `gamma=0.1`.
- Thinking: prune weak deep splits.
- Interpretation: tied rounded AUC but added complexity. Discarded by simplicity criterion.

### Experiment 14

- Commit: `f02d374`
- CV AUC: `0.7784 +/- 0.0054`
- Status: keep
- Change: added `DepHour`, `DepMinute`, `DepMinutes`, `DepTimeSin`, `DepTimeCos`.
- Thinking: scikit-learn cyclical feature guidance suggested sine/cosine for periodic variables. Raw HHMM time has bad numeric geometry and midnight wraparound.
- Interpretation: major feature-engineering win. Time-of-day is predictive and raw HHMM was not enough.

### Experiment 15

- Commit: `09c2a1c`
- CV AUC: `0.7787 +/- 0.0052`
- Status: keep
- Change: removed raw `DepTime` from `num_cols`, but still used it to derive time features.
- Thinking: raw HHMM was likely redundant after cleaner time encodings.
- Interpretation: simplification win. Removing raw time slightly improved AUC.

### Experiment 16

- Commit: `ed3e2f1`
- CV AUC: `0.7784 +/- 0.0052`
- Status: discard
- Change: calendar numeric/cyclic features and weekend flag.
- Thinking: extend cyclic encoding to Month, DayOfMonth, and DayOfWeek.
- Interpretation: worse. Native categorical calendar fields were enough; extra numeric calendar encodings added noise/complexity.

### Experiment 17

- Commit: `93377f1`
- CV AUC: `0.7783 +/- 0.0054`
- Status: discard
- Change: `Route = Origin_Dest` categorical interaction.
- Thinking: route-specific delay risk might be direct and predictive.
- Interpretation: worse and slower. Trees already handled origin/destination interactions well enough.

### Experiment 18

- Commit: `1bdc8a4`
- CV AUC: `0.7742 +/- 0.0047`
- Status: discard
- Change: back to depth 7 and child weight 15 with time features.
- Thinking: after time features, perhaps shallower trees would suffice.
- Interpretation: no. Depth 8 remained important.

### Experiment 19

- Commit: `8031221`
- CV AUC: `0.7783 +/- 0.0052`
- Status: discard
- Change: `max_bin=512`.
- Thinking: finer histogram resolution might help continuous engineered time features.
- Interpretation: worse. Higher bin count did not help.

### Experiment 20

- Commit: `a81641c`
- CV AUC: `0.7795 +/- 0.0050`
- Status: keep
- Change: 600 trees and learning rate 0.02.
- Thinking: revisit smoother boosting after feature engineering.
- Interpretation: positive. More rounds/lower rate still helped.

### Experiment 21

- Commit: `6ff2d6c`
- CV AUC: `0.7797 +/- 0.0051`
- Status: keep
- Change: 800 trees and learning rate 0.015.
- Thinking: continue smoother schedule.
- Interpretation: small improvement; diminishing returns.

### Experiment 22

- Commit: `63104e6`
- CV AUC: `0.7799 +/- 0.0053`
- Status: keep
- Change: 1000 trees and learning rate 0.012.
- Thinking: one more lower-rate step while under runtime.
- Interpretation: small improvement; still valid runtime.

### Experiment 23

- Commit: `f2efbc1`
- CV AUC: `0.7784 +/- 0.0053`
- Status: discard
- Change: child weight 40.
- Thinking: more trees might need stronger regularization.
- Interpretation: underfit/over-regularized.

### Experiment 24

- Commit: `066e231`
- CV AUC: `0.7805 +/- 0.0050`
- Status: keep
- Change: child weight 20.
- Thinking: since stronger child weight hurt, try more flexibility.
- Interpretation: improved. The longer schedule wanted more flexible leaves.

### Experiment 25

- Commit: `bedc3db`
- CV AUC: `0.7808 +/- 0.0050`
- Status: keep
- Change: child weight 15.
- Thinking: continue flexibility sweep.
- Interpretation: improved. Best moved lower than expected.

### Experiment 26

- Commit: `ccde93f`
- CV AUC: `0.7797 +/- 0.0048`
- Status: discard
- Change: child weight 10.
- Thinking: test lower boundary.
- Interpretation: too loose; AUC fell.

### Experiment 27

- Commit: `1315f01`
- CV AUC: `0.7802 +/- 0.0045`
- Status: discard
- Change: child weight 12.
- Thinking: interpolate between 10 and 15.
- Interpretation: worse than 15.

### Experiment 28

- Commit: `a370e42`
- CV AUC: `0.7809 +/- 0.0049`
- Status: keep
- Change: child weight 18.
- Thinking: test upper side around 15 and 20.
- Interpretation: slight best. The optimum was around 18.

### Experiment 29

- Commit: `d10606c`
- CV AUC: `0.7806 +/- 0.0050`
- Status: discard
- Change: child weight 17.
- Thinking: immediate neighbor below 18.
- Interpretation: worse than 18.

### Experiment 30

- Commit: `a3b746f`
- CV AUC: `0.7808 +/- 0.0052`
- Status: discard
- Change: child weight 19.
- Thinking: immediate neighbor above 18.
- Interpretation: worse than 18.

### Experiment 31

- Commit: `ce7470c`
- CV AUC: `0.7792 +/- 0.0051`
- Status: discard
- Change: `colsample_bytree=1.0`.
- Thinking: maybe all features per tree would help interactions.
- Interpretation: worse. Column sampling was beneficial.

### Experiment 32

- Commit: `b4e34e4`
- CV AUC: `0.7811 +/- 0.0051`
- Status: keep
- Change: `colsample_bytree=0.75`.
- Thinking: since 1.0 hurt, stronger stochastic feature sampling might generalize better.
- Interpretation: improved.

### Experiment 33

- Commit: `86087a4`
- CV AUC: `0.7819 +/- 0.0050`
- Status: keep
- Change: `colsample_bytree=0.65`.
- Thinking: continue column-sampling sweep.
- Interpretation: improved again. Tree diversity mattered.

### Experiment 34

- Commit: `d0c6fb5`
- CV AUC: `0.7816 +/- 0.0052`
- Status: discard
- Change: `colsample_bytree=0.55`.
- Thinking: test stronger sampling.
- Interpretation: too much feature hiding.

### Experiment 35

- Commit: `3b5621f`
- CV AUC: `0.7819 +/- 0.0050`
- Status: discard
- Change: `colsample_bytree=0.60`.
- Thinking: interpolate between 0.55 and 0.65.
- Interpretation: tied rounded best but did not improve; kept simpler existing branch.

### Experiment 36

- Commit: `a5e1541`
- CV AUC: `0.7816 +/- 0.0050`
- Status: discard
- Change: `colsample_bytree=0.70`.
- Thinking: upper neighbor of 0.65.
- Interpretation: worse. 0.65 was best.

### Experiment 37

- Commit: `2a91b2c`
- CV AUC: `0.7812 +/- 0.0051`
- Status: discard
- Change: `subsample=0.75`.
- Thinking: pair row stochasticity with column stochasticity.
- Interpretation: too much row sampling hurt.

### Experiment 38

- Commit: `e1cbaf9`
- CV AUC: `0.7818 +/- 0.0050`
- Status: discard
- Change: `subsample=0.95`.
- Thinking: maybe row sampling should be lighter.
- Interpretation: close but not better than 0.85/0.90.

### Experiment 39

- Commit: `d9d5c93`
- CV AUC: `0.7821 +/- 0.0053`
- Status: keep
- Change: `subsample=0.90`.
- Thinking: interpolate between 0.85 and 0.95.
- Interpretation: improved. Mild row sampling was best.

### Experiment 40

- Commit: `64b2866`
- CV AUC: `0.7818 +/- 0.0053`
- Status: discard
- Change: `subsample=0.88`.
- Thinking: fine-tune near 0.90.
- Interpretation: worse. Keep 0.90.

### Experiment 41

- Commit: `1b4c93d`
- CV AUC: `0.7859 +/- 0.0049`
- Status: discard
- Change: lossguide, 128 leaves, 1000 trees.
- Thinking: XGBoost docs suggested lossguide could allocate splits more efficiently than fixed depth.
- Interpretation: AUC was excellent, but total printed time exceeded the timeout, so invalid.

### Experiment 42

- Commit: `70f193c`
- CV AUC: `0.7769 +/- 0.0051`
- Status: discard
- Change: lossguide with 64 leaves.
- Thinking: try smaller valid lossguide version.
- Interpretation: underfit badly.

### Experiment 43

- Commit: `9610702`
- CV AUC: `0.7825 +/- 0.0050`
- Status: keep
- Change: lossguide with 96 leaves.
- Thinking: middle between underfit 64 and too-slow 128.
- Interpretation: valid and slightly better than depthwise best.

### Experiment 44

- Commit: `03d5217`
- CV AUC: `0.7840 +/- 0.0051`
- Status: keep
- Change: 850 trees, learning rate 0.014, 112 leaves.
- Thinking: trade fewer rounds for more leaves.
- Interpretation: improved.

### Experiment 45

- Commit: `007f281`
- CV AUC: `0.7855 +/- 0.0052`
- Status: keep
- Change: 750 trees, learning rate 0.016, 128 leaves.
- Thinking: revisit 128 leaves with fewer rounds to satisfy timeout.
- Interpretation: improved, valid.

### Experiment 46

- Commit: `a1c2b6b`
- CV AUC: `0.7872 +/- 0.0051`
- Status: keep
- Change: 650 trees, learning rate 0.0185, 144 leaves.
- Thinking: continue leaf-for-rounds trade.
- Interpretation: improved. Larger trees carried useful interactions.

### Experiment 47

- Commit: `65a84fa`
- CV AUC: `0.7885 +/- 0.0051`
- Status: keep
- Change: 550 trees, learning rate 0.022, 160 leaves.
- Thinking: push larger-tree trend.
- Interpretation: improved.

### Experiment 48

- Commit: `a28fa57`
- CV AUC: `0.7906 +/- 0.0046`
- Status: keep
- Change: 450 trees, learning rate 0.027, 192 leaves.
- Thinking: larger lossguide trees still looked promising.
- Interpretation: large improvement.

### Experiment 49

- Commit: `a18be2e`
- CV AUC: `0.7918 +/- 0.0050`
- Status: keep
- Change: 350 trees, learning rate 0.034, 256 leaves.
- Thinking: test higher leaf cap with fewer rounds.
- Interpretation: improved.

### Experiment 50

- Commit: `2d47dfd`
- CV AUC: `0.7939 +/- 0.0050`
- Status: keep
- Change: 250 trees, learning rate 0.048, 384 leaves.
- Thinking: very high leaf cap may capture rich interactions with fewer rounds.
- Interpretation: strong improvement; high-leaf lossguide became the main path.

### Experiment 51

- Commit: `6085269`
- CV AUC: `0.7939 +/- 0.0046`
- Status: discard
- Change: 200 trees, learning rate 0.06, 512 leaves.
- Thinking: continue high-leaf trend.
- Interpretation: tied rounded best but did not improve; simpler 384-leaf model kept.

### Experiment 52

- Commit: `75d7f37`
- CV AUC: `0.7943 +/- 0.0051`
- Status: keep
- Change: 300 trees, learning rate 0.04, 384 leaves.
- Thinking: maybe 384 leaves wanted smoother boosting rather than more leaves.
- Interpretation: improved.

### Experiment 53

- Commit: `38316c9`
- CV AUC: `0.7946 +/- 0.0050`
- Status: keep
- Change: 350 trees, learning rate 0.034, 384 leaves.
- Thinking: continue smoother boosting at fixed 384 leaves.
- Interpretation: improved.

### Experiment 54

- Commit: `568aca6`
- CV AUC: `0.0000`
- Status: crash
- Change: 400 trees, learning rate 0.03, 384 leaves.
- Thinking: push smoother boosting close to timeout.
- Interpretation: timed out before CV AUC.

### Experiment 55

- Commit: `e86b1ac`
- CV AUC: `0.7952 +/- 0.0049`
- Status: keep
- Change: 375 trees, learning rate 0.032, 384 leaves.
- Thinking: interpolate between valid 350 and timed-out 400.
- Interpretation: improved and valid.

### Experiment 56

- Commit: `a290479`
- CV AUC: `0.7958 +/- 0.0047`
- Status: keep
- Change: 390 trees, learning rate 0.031, 384 leaves.
- Thinking: squeeze more boosting under timeout.
- Interpretation: improved and valid.

### Experiment 57

- Commit: `37f0f3e`
- CV AUC: `0.7953 +/- 0.0048`
- Status: discard
- Change: 395 trees, learning rate 0.0305, 384 leaves.
- Thinking: final narrow push before 400 timeout.
- Interpretation: valid but worse. 390 was best.

### Experiment 58

- Commit: `6489367`
- CV AUC: `0.7927 +/- 0.0050`
- Status: discard
- Change: child weight 25.
- Thinking: high-leaf model may need stronger leaf mass.
- Interpretation: over-regularized.

### Experiment 59

- Commit: `7d563ed`
- CV AUC: `0.7964 +/- 0.0047`
- Status: keep
- Change: child weight 12.
- Thinking: since stronger child weight hurt, try more flexibility.
- Interpretation: improved.

### Experiment 60

- Commit: `57c3219`
- CV AUC: `0.7957 +/- 0.0048`
- Status: discard
- Change: child weight 8.
- Thinking: test lower boundary.
- Interpretation: too loose; worse than 12.

### Experiment 61

- Commit: `dd71459`
- CV AUC: `0.7964 +/- 0.0047`
- Status: discard
- Change: `gamma=0.05`.
- Thinking: prune weak splits in high-leaf model.
- Interpretation: tied rounded AUC but added complexity.

### Experiment 62

- Commit: `7f28f86`
- CV AUC: `0.7959 +/- 0.0048`
- Status: discard
- Change: `reg_lambda=0.5`.
- Thinking: reduce L2 regularization because more flexible leaves helped.
- Interpretation: worse.

### Experiment 63

- Commit: `8482685`
- CV AUC: `0.7952 +/- 0.0044`
- Status: discard
- Change: `reg_lambda=2.0`.
- Thinking: try stronger L2 after lower L2 hurt.
- Interpretation: worse.

### Experiment 64

- Commit: `942a0cf`
- CV AUC: `0.7944 +/- 0.0049`
- Status: discard
- Change: `colsample_bynode=0.8`.
- Thinking: split-level feature randomness might regularize high-leaf trees.
- Interpretation: hurt AUC.

### Experiment 65

- Commit: `7247d86`
- CV AUC: `0.0000`
- Status: crash
- Change: distance log and square-root transforms.
- Thinking: distance may have nonlinear short/long haul effects.
- Interpretation: timed out. Extra features were too expensive in near-limit model.

### Experiment 66

- Commit: `1070d81`
- CV AUC: `0.7961 +/- 0.0046`
- Status: discard
- Change: 385 trees, learning rate 0.0315, 384 leaves.
- Thinking: get slight runtime headroom while near 390.
- Interpretation: worse than 390.

### Experiment 67

- Commit: `431f9f7`
- CV AUC: `0.0000`
- Status: crash
- Change: `max_cat_threshold=256`.
- Thinking: high-leaf lossguide might exploit richer high-cardinality categorical partitions.
- Interpretation: timed out.

### Experiment 68

- Commit: `c7e711e`
- CV AUC: `0.7962 +/- 0.0045`
- Status: discard
- Change: `max_cat_threshold=32`.
- Thinking: cheaper/regularized category partitions.
- Interpretation: close but worse than default.

### Experiment 69

- Commit: `6e3f9c3`
- CV AUC: `0.7966 +/- 0.0041`
- Status: keep
- Change: 330 trees, learning rate 0.0365, 448 leaves.
- Thinking: return to leaf-for-rounds trade after regularization attempts failed.
- Interpretation: new best.

### Experiment 70

- Commit: `918f195`
- CV AUC: `0.7973 +/- 0.0044`
- Status: keep
- Change: 300 trees, learning rate 0.04, 512 leaves.
- Thinking: continue leaf capacity direction.
- Interpretation: improved.

### Experiment 71

- Commit: `a80412e`
- CV AUC: `0.7979 +/- 0.0045`
- Status: keep
- Change: 250 trees, learning rate 0.048, 640 leaves.
- Thinking: more leaves with fewer rounds.
- Interpretation: improved.

### Experiment 72

- Commit: `a2d1397`
- CV AUC: `0.7962 +/- 0.0038`
- Status: discard
- Change: 200 trees, learning rate 0.06, 768 leaves.
- Thinking: push larger leaves further.
- Interpretation: underperformed. Too few rounds / too much per-tree capacity.

### Experiment 73

- Commit: `c4573f5`
- CV AUC: `0.7983 +/- 0.0045`
- Status: keep
- Change: 275 trees, learning rate 0.044, 640 leaves.
- Thinking: smooth the 640-leaf schedule.
- Interpretation: improved.

### Experiment 74

- Commit: `a79aed2`
- CV AUC: `0.7984 +/- 0.0040`
- Status: keep
- Change: 250 trees, learning rate 0.048, 704 leaves.
- Thinking: more leaves may offset fewer rounds.
- Interpretation: slight improvement.

### Experiment 75

- Commit: `6239b57`
- CV AUC: `0.7976 +/- 0.0047`
- Status: discard
- Change: 230 trees, learning rate 0.052, 768 leaves.
- Thinking: revisit 768 leaves with more rounds than experiment 72.
- Interpretation: still worse. 768 leaves was too far.

### Experiment 76

- Commit: `00d0b0f`
- CV AUC: `0.7989 +/- 0.0051`
- Status: keep
- Change: 260 trees, learning rate 0.0465, 704 leaves.
- Thinking: smooth the 704-leaf schedule slightly.
- Interpretation: final best. Strong local optimum under timeout.

### Experiment 77

- Commit: `d0aca0e`
- CV AUC: `0.0000`
- Status: crash
- Change: 265 trees, learning rate 0.0455, 704 leaves.
- Thinking: tiny additional smoothing.
- Interpretation: timed out.

### Experiment 78

- Commit: `76302ee`
- CV AUC: `0.0000`
- Status: crash
- Change: child weight 10 at 704 leaves.
- Thinking: larger leaf model might want more flexibility.
- Interpretation: timed out.

### Experiment 79

- Commit: `7ce5a0d`
- CV AUC: `0.7973 +/- 0.0045`
- Status: discard
- Change: child weight 14 at 704 leaves.
- Thinking: slightly stronger child weight might regularize.
- Interpretation: hurt AUC.

### Experiment 80

- Commit: `2a70be4`
- CV AUC: `0.7968 +/- 0.0040`
- Status: discard
- Change: max depth cap 16.
- Thinking: avoid very deep sparse paths in lossguide.
- Interpretation: hurt AUC.

### Experiment 81

- Commit: `55e5e9d`
- CV AUC: `0.7973 +/- 0.0044`
- Status: discard
- Change: `max_bin=128`.
- Thinking: maybe lower bin count buys runtime headroom.
- Interpretation: AUC fell and runtime did not improve enough.

### Experiment 82

- Commit: `3d49c0c`
- CV AUC: `0.0000`
- Status: crash
- Change: `colsample_bytree=0.60`.
- Thinking: high-leaf model might want more feature randomness.
- Interpretation: timed out.

### Experiment 83

- Commit: `d50b33c`
- CV AUC: `0.0000`
- Status: crash
- Change: `subsample=0.95`.
- Thinking: high-leaf model might want more rows per tree.
- Interpretation: timed out.

### Experiment 84

- Commit: `91ebdef`
- CV AUC: `0.7979 +/- 0.0039`
- Status: discard
- Change: 672 leaves.
- Thinking: test lower neighbor around 704.
- Interpretation: worse than 704.

### Experiment 85

- Commit: `07012cc`
- CV AUC: `0.7987 +/- 0.0041`
- Status: discard
- Change: 250 trees, learning rate 0.048, 736 leaves.
- Thinking: test upper neighbor without going to 768.
- Interpretation: close but below 704-leaf best.

### Experiment 86

- Commit: `e223999`
- CV AUC: `0.7980 +/- 0.0045`
- Status: discard
- Change: 255 trees, learning rate 0.0472, 704 leaves.
- Thinking: interpolate schedule between 250 and 260 trees.
- Interpretation: worse than 260/0.0465.

### Experiment 87

- Commit: `540e89b`
- CV AUC: `0.0000`
- Status: crash
- Change: removed `DepMinute`.
- Thinking: simplify redundant minute feature.
- Interpretation: timed out; no result.

### Experiment 88

- Commit: `d8044af`
- CV AUC: `0.7988 +/- 0.0040`
- Status: discard
- Change: learning rate 0.048.
- Thinking: fixed 260 trees might be under-boosted.
- Interpretation: very close but below best.

### Experiment 89

- Commit: `ce5cf47`
- CV AUC: `0.7983 +/- 0.0043`
- Status: discard
- Change: learning rate 0.045.
- Thinking: test lower learning-rate side.
- Interpretation: worse.

### Experiment 90

- Commit: `3db368e`
- CV AUC: `0.7979 +/- 0.0041`
- Status: discard
- Change: learning rate 0.047.
- Thinking: interpolate between best 0.0465 and worse 0.048.
- Interpretation: worse. The current learning rate is a tight local optimum.

## What Worked

The strongest wins:

- More boosting rounds with lower learning rate improved the starter model.
- Depthwise interactions helped until depth 8.
- Departure-time feature engineering was the biggest feature-engineering win.
- Removing raw `DepTime` after deriving time features was a simplification win.
- Column sampling at `colsample_bytree=0.65` was important.
- Row sampling at `subsample=0.90` was slightly helpful.
- `grow_policy="lossguide"` with high `max_leaves` was the biggest HPO breakthrough.
- Trading many smaller rounds for fewer large lossguide trees worked extremely well.
- Final best local point: 260 trees, learning rate 0.0465, 704 leaves, child weight 12.

## What Did Not Work

Consistently weak directions:

- Forcing one-hot categorical splits.
- Increasing categorical thresholds under high-capacity models.
- Calendar cyclic/numeric features.
- Explicit route categorical feature.
- Distance transforms under the near-timeout model.
- Generic split/leaf regularization after lossguide was tuned.
- Node-level column sampling.
- Depth caps for lossguide.
- 768-leaf variants.
- Local schedule tweaks around the final best.

## Interpretation Of The Dataset And Model

The useful signal appears to be interaction-heavy:

- time of day,
- airport/origin/destination,
- carrier,
- distance,
- calendar categories.

The model wants high-order interactions, but the best way to express them is not hand-built interaction features. It is high-capacity lossguide trees with the original categorical features plus clean departure-time encodings.

The final model is runtime-constrained. Many nearby experiments did not fail because the idea was nonsensical; they failed because they pushed total training time beyond the one-minute rule. That means any future improvement should either:

- reduce runtime while preserving AUC,
- introduce a very cheap feature,
- or find a better capacity allocation that does not add more split work.

## Suggested Next Directions

I would not keep locally interpolating learning rate, tree count, or leaf count around the final model; experiments 77-90 show the local region is mostly exhausted.

Reasonable future directions:

- Try a cheaper alternative to current high-leaf lossguide that preserves AUC, for example slightly fewer leaves with a different sampling setting, but only if runtime headroom is needed.
- Try one very targeted time feature at a time only if it does not increase runtime too much.
- Investigate whether current final model has fold-specific variance that suggests tuning for robustness rather than mean AUC.
- If allowed by the harness in a future run, consider a purpose-built CV loop with early stopping, but this would be a larger structural change and should be treated carefully.
