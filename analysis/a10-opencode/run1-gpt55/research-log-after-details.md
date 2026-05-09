# Detailed Research Log After Run

## Executive Summary

This run started from the repository baseline on branch `may8` and iterated through 60 logged experiments after setup. The baseline commit `7d0b368` scored `CV AUC 0.7445 +/- 0.0043`. The current best commit is `4d79cc8` with `CV AUC 0.7847 +/- 0.0054`, an absolute improvement of `+0.0402` AUC over baseline.

The best final model is still structurally simple in `train.py`: no external dependencies, no changed evaluation harness, no extra feature engineering in the current kept code, and native XGBoost categorical handling remains intact. The final improvement came mostly from HPO, not feature engineering.

Current best parameters in `train.py`:

```python
xgb.XGBClassifier(
    n_estimators=800,
    max_depth=14,
    learning_rate=0.017,
    min_child_weight=25,
    subsample=0.9,
    colsample_bytree=0.85,
    reg_lambda=0,
    tree_method="hist",
    max_bin=2048,
    enable_categorical=True,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
```

Main lessons:

- The baseline was substantially underfit. Longer low-rate boosting gave the first major improvement.
- Deep trees helped a lot, but only when paired with sufficient leaf support through `min_child_weight`.
- Native categorical partitioning was important. One-hot categorical splitting hurt badly.
- Feature engineering was not productive under the final runtime budget. Useful-sounding additions either did not improve AUC or timed out once the model became large.
- `max_bin` mattered. Increasing histogram resolution to `2048` improved AUC enough to justify the cost.
- L2 regularization was counterproductive once `min_child_weight` was high. `reg_lambda=0` was best.
- Runtime became the binding constraint after the depth/child-weight improvements. Many later ideas were rejected because they timed out or only tied the simpler kept model.

## Constraints Followed

- I did not modify `prepare.py`.
- I did not read, run, or reference `check_groundtruth.py` or `run_groundtruth_all.sh`.
- I did not install dependencies.
- I kept evaluation as the existing 5-fold `StratifiedKFold` CV in `train.py`.
- I used only `2005-slice1-100k.csv` for training/CV as required by the existing harness.
- I avoided count-derived features because the instructions warned that the balanced undersampled data can make counts misleading.
- Feature engineering attempts were implemented inside `prepare(df)` as required.
- `results.tsv` and research logs were left untracked by git.

## Source Research Summary

I used web research repeatedly before and during experiments. These were the main sources and what I took from them.

### XGBoost Parameter Documentation

Source: `https://xgboost.readthedocs.io/en/stable/parameter.html`

Information used:

- `learning_rate`/`eta` shrinks each boosting update and makes the ensemble more conservative.
- Lower `learning_rate` generally needs more boosting rounds.
- `max_depth` increases model complexity and can overfit, but also reduces bias.
- `min_child_weight` prevents splits that create leaves with too little Hessian support, making deeper trees more conservative.
- `subsample` and `colsample_bytree` add randomness and can control overfitting.
- `reg_lambda` is L2 leaf-weight regularization. Higher values make the model more conservative.
- `gamma`/`min_split_loss` requires a minimum loss reduction for further splits.
- `tree_method="hist"` is the fast histogram algorithm and is compatible with categorical features.
- `max_bin` controls numeric split resolution for histogram methods.
- `max_cat_to_onehot` and `max_cat_threshold` affect categorical splitting.
- `max_delta_step` can make logistic updates more conservative, mainly mentioned for imbalanced data.

How it influenced the run:

- It motivated the first major schedule change: many more trees with lower learning rate.
- It motivated the repeated depth/min-child-weight tradeoff experiments.
- It motivated sampling experiments, regularization experiments, and histogram resolution experiments.
- It motivated `max_delta_step=1`, which ultimately did not help.

### XGBoost Categorical Data Tutorial

Source: `https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html`

Information used:

- Pandas `category` dtype plus `enable_categorical=True` is the intended sklearn-interface path.
- Supported tree methods for categorical data include `hist` and `approx`; exact is not supported.
- XGBoost can use either one-hot categorical splits or partition-based categorical splits.
- `max_cat_to_onehot` controls the threshold for one-hot versus partition-based splits.
- Partition-based splitting uses optimal partitioning ideas for categorical groups.
- Consistent category encoding between train and inference matters.

How it influenced the run:

- It confirmed the baseline's categorical handling was structurally sound.
- It motivated keeping native categorical handling instead of manual one-hot encoding.
- It motivated the `max_cat_to_onehot=64` experiment.
- It motivated `max_cat_threshold` experiments at 128 and 32.

### XGBoost Scikit-Learn Estimator Interface

Source: `https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html`

Information used:

- Early stopping is supported through eval sets, but with CV each fold can produce a different best iteration.
- The docs explicitly warn that using early stopping during CV can produce fold-specific model sizes, and a cleaner pattern is to tune hyperparameters first then retrain.
- XGBoost and sklearn can oversubscribe threads if both are parallelized. The current code uses `cross_val_score(..., n_jobs=1)` and XGBoost `n_jobs=-1`, which matches the docs' recommendation to avoid thread thrashing.

How it influenced the run:

- I did not force early stopping into the CV harness because the current code uses `cross_val_score`, and adding fold-specific early stopping would be a larger evaluation-loop change with subtle comparability issues.
- I kept sklearn CV single-threaded and left XGBoost as the parallel part.

### XGBoost Parameter Tuning Notes

Source: `https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html`

Information used:

- Most tuning is a bias-variance tradeoff.
- Direct complexity controls include `max_depth`, `min_child_weight`, `gamma`, and `max_cat_threshold`.
- Randomness controls include `subsample` and column sampling.
- Lower `eta` should usually be paired with more rounds.
- Knowing the data and preprocessing can be as important as model tuning.

How it influenced the run:

- It framed the early schedule experiments as reducing bias.
- It framed deeper trees plus higher child weight as a controlled complexity increase.
- It motivated later sampling and categorical-threshold tests.

### XGBoost Feature Interaction Constraints

Source: `https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html`

Information used:

- Tree paths encode interactions between variables.
- Interaction constraints can restrict which variables interact to reduce spurious interactions.

How it influenced the run:

- This reinforced that the large depth gains were probably interaction-driven.
- I did not add interaction constraints because the empirical signal showed that broad high-order interactions were valuable, and restricting them would likely fight the successful direction.

### XGBoost `cat_in_the_dat` Example

Source: `https://xgboost.readthedocs.io/en/stable/python/examples/cat_in_the_dat.html`

Information used:

- The example compares native categorical support with one-hot encoded data.
- It uses `max_cat_to_onehot=1` to force partitioning in the categorical model.
- It reinforces that one-hot versus partitioning is a real modeling choice, not just implementation detail.

How it influenced the run:

- It motivated treating categorical split strategy as an experiment axis.
- The actual run showed partitioning was clearly better on this dataset.

### Bureau of Transportation Statistics Delay Documentation

Sources:

- `https://www.bts.gov/topics/airlines-and-airports/airline-time-performance-and-causes-flight-delays`
- `https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp`

Information used:

- Departure delay is defined relative to the scheduled departure time, with 15 minutes as the relevant threshold.
- Delay causes include carrier operations, weather, national aviation system issues, late-arriving aircraft, and security.
- Reporting is for domestic carrier/airport operations.

How it influenced the run:

- It motivated time-of-day feature engineering from `DepTime`.
- It motivated route-level `Origin-Dest` feature engineering.
- It supported the idea that carrier, airport, route, and time interactions matter.
- Empirically, the current deep categorical model already captured enough from raw fields, and added features became too expensive at the final model size.

### XGBoost Tree Methods

Source: `https://xgboost.readthedocs.io/en/stable/treemethod.html`

Information used:

- `hist` is fastest because it uses a global sketch and histograms.
- `approx` can sometimes be more accurate for non-constant Hessian objectives but is slower.
- `hist` supports categorical data, `grow_policy`, and `max_leaves`.
- Higher `max_bin` can improve split optimality at extra computation cost.

How it influenced the run:

- It motivated `max_bin=512`, `1024`, `2048`, and `4096` experiments.
- It motivated `grow_policy="lossguide"` with `max_leaves=512`.
- It supported staying with `hist` because runtime was already a bottleneck.

### XGBoost Monotonic Constraints

Source: `https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html`

Information used:

- Monotonic constraints can help when there is strong prior knowledge about feature-response direction.
- With `hist`, monotonic constraints can make trees unnecessarily shallow because candidate splits can be eliminated.

How it influenced the run:

- I did not try monotonic constraints because `Distance` and `DepTime` do not have obvious monotonic effects on delay probability, and the docs warned of shallow trees under `hist`.

### XGBoost DART Booster

Source: `https://xgboost.readthedocs.io/en/stable/tutorials/dart.html`

Information used:

- DART drops trees during training to reduce overfitting.
- DART can be slower because dropout prevents use of prediction buffers.
- Early stopping can be less stable due to dropout randomness.

How it influenced the run:

- DART was considered but not tried after the model reached the runtime cap. The docs specifically suggested it would likely be slower, and the successful model was not obviously overfitting according to CV improvements.

### XGBoost Callback and Evaluation Examples

Sources:

- `https://xgboost.readthedocs.io/en/stable/python/callbacks.html`
- `https://xgboost.readthedocs.io/en/stable/python/examples/sklearn_evals_result.html`

Information used:

- XGBoost supports callbacks, checkpointing, and eval logs.
- Evaluation histories can be accessed from fitted estimators.

How it influenced the run:

- I did not add callback machinery because the experiment harness already has a simple, comparable CV score path, and callbacks would add complexity without directly targeting AUC.

### XGBoost Random Forest Documentation

Source: `https://xgboost.readthedocs.io/en/stable/tutorials/rf.html`

Information used:

- XGBoost can train random forests via `num_parallel_tree` and strong row/column subsampling.
- `colsample_bynode` is typical for random forest style per-split feature sampling.

How it influenced the run:

- It motivated `colsample_bynode=0.8`, which hurt AUC.
- I did not try full XGBoost RF because the strongly tuned boosted-tree direction was working and runtime was tight.

### XGBoost API, QuantileDMatrix, and Boost From Prediction

Sources:

- `https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier`
- `https://xgboost.readthedocs.io/en/stable/python/examples/boost_from_prediction.html`

Information used:

- `DMatrix` and `QuantileDMatrix` can expose lower-level training paths and may save memory for histogram training.
- Boosting from prediction uses raw margins as base margins.

How it influenced the run:

- I did not rewrite the sklearn CV path into native DMatrix training because it would be a larger refactor and risk changing evaluation behavior or categorical handling details.
- Boost-from-prediction was not pursued because stacking/offset style training would add complexity and likely exceed the runtime budget.

### Scikit-Learn Cross-Validation Documentation

Source: `https://scikit-learn.org/stable/modules/cross_validation.html`

Information used:

- Cross-validation is the correct approach for estimating generalization under repeated model tuning.
- `StratifiedKFold` preserves class ratios and avoids folds with class imbalance problems.
- Preprocessing should be learned on training folds and applied to held-out folds where applicable.

How it influenced the run:

- I kept the given `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` setup.
- Since categorical levels are defined from the training slice and the same function is used consistently, the feature pipeline remained simple and reproducible.

## Block-Level Narrative

### Setup And Baseline

The starting point was intentionally simple: raw numeric features `DepTime` and `Distance`, raw categorical features `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, and `Dest`, and native XGBoost categorical support. The baseline used only 30 trees with depth 6 and learning rate 0.1.

Baseline result was `0.7445`, which became the reference for all subsequent decisions. The initial source research suggested that 30 trees was likely too few for a categorical tabular problem with airport/carrier/time interactions.

### Experiments 1-10: Boosting Schedule And Deep Supported Trees

The first block found the two most important early patterns.

First, longer low-rate boosting helped a lot. Moving from 30 trees to 160, 300, 500, and then 800 trees steadily improved AUC. The best schedule in this block was 800 trees at learning rate 0.015.

Second, tree depth mattered much more than expected. Depth 5 hurt badly, showing the problem was not overfitting from depth. Depth 7 with stronger `min_child_weight` produced a large jump, and depth 8 and 9 with higher child-weight constraints improved further.

End of block best: `c063231`, `0.7754`.

Interpretation: the dataset needs high-order interactions, likely involving airports, carriers, calendar fields, departure time, and distance. But deeper trees only work when unsupported leaves are constrained.

### Experiments 11-20: Categorical Strategy, Feature Engineering, Sampling, And Regularization

This block tested whether the strong depth result could be improved through categorical split options, basic feature engineering, sampling, and leaf regularization.

Categorical one-hot splitting was bad. `max_cat_to_onehot=64` dropped AUC to `0.7626`, showing that partition-based categorical splitting was important even for the medium-cardinality fields. Changing `max_cat_threshold` to 128 or 32 did not improve the default.

Departure-time feature engineering did not help. Adding `DepHour` and `DepMinutes` slightly underperformed, suggesting raw `DepTime` plus deep trees was enough.

Sampling had a narrow optimum. Both stronger sampling (`0.75/0.75`) and lighter sampling (`0.95/0.95`) were worse than `0.85/0.85` at that stage.

L2 regularization was surprisingly harmful. `reg_lambda=5` hurt, `reg_lambda=0.5` helped a little, and `reg_lambda=0` helped again. This indicated that `min_child_weight` was already the useful regularizer, and extra shrinkage on leaf weights limited signal.

End of block best: `eba944b`, `0.7757`.

### Experiments 21-30: Tree Method, Histogram Resolution, And More Depth

This block tested tree construction choices and numeric split resolution.

`gamma=0.1` did not help. The model already had enough split control from `min_child_weight`.

Lossguide tied AUC but was much slower, so depthwise growth remained preferable.

Increasing `max_bin` was productive. `512`, `1024`, and `2048` all improved AUC slightly. `4096` tied `2048`, so `2048` was the best simplicity/runtime tradeoff.

After increasing split resolution and removing L2, depth increases became productive again. Depth 10 with child 80 failed badly, proving that over-raising child support over-constrained the model. Depth 10 with child 40 worked well, then depth 11, 12, and 13 improved sequentially.

End of block best: `ca42c5c`, `0.7798`.

Interpretation: the earlier child-weight scaling idea was partly wrong. More depth helped, but child weight should not increase indefinitely with depth.

### Experiments 31-40: Depth Plateau, Lower Child Support, And Runtime Wall

Depth 14 improved to `0.7800`, while depth 15 tied and was discarded. That established depth 14 as the depth plateau.

Lowering `min_child_weight` at depth 14 was the next major gain. Child 30 improved to `0.7822`, and child 25 improved to `0.7834`.

At this point runtime became the controlling constraint. Schedule compression to 700 or 750 rounds lost AUC. Reducing `max_bin` to 1024 lost AUC. Feature engineering attempts began timing out, including route and numeric calendar features. Reducing depth to 13 with child 25 also lost AUC.

End of block best: `206181a`, `0.7834`.

Interpretation: the model had found a high-capacity configuration that worked, but the evaluation loop was now close to the one-minute limit. Future experiments needed to be same-cost or cost-reducing.

### Experiments 41-50: Runtime-Bounded Local Tuning

This block focused on small parameter changes because heavier changes timed out.

Per-node column sampling hurt. Full column exposure timed out. Lower row sampling hurt. `max_delta_step=1` tied but added complexity.

Lowering child weight to 23 timed out. This made child 25 the practical lower boundary.

Learning-rate tuning gave a useful gain. `0.016` improved to `0.7837`, and `0.017` improved to `0.7840`. `0.018`, `0.0175`, and `0.0172` did not beat the simpler `0.017` setting. Child 27 also underperformed.

End of block best: `b78c4ac`, `0.7840`.

Interpretation: the model was still slightly underfit at learning rate 0.015, but the local optimum was narrow around 0.017 for 800 rounds.

### Experiments 51-60: Row Sampling Peak And Final Best

The last block tuned row and column sampling around the new learning-rate optimum.

Slight schedule compression to 780 rounds lost AUC. Child 24 tied but did not improve.

Increasing row sampling was the final successful axis. `subsample=0.875` improved to `0.7844`, and `subsample=0.9` improved to `0.7847`. `0.925` dropped and `0.9125` timed out, so 0.9 is both the best tested row-sampling value and near the feasible runtime limit.

Column sampling could not be improved. `colsample_bytree=0.875` timed out. `0.825` tied but did not improve. With row sampling at 0.9, retesting learning rate 0.0175 tied but did not beat 0.017.

Final best: `4d79cc8`, `0.7847`.

## Detailed Iteration Log

### Iteration 0: `7d0b368` baseline

- Status: keep.
- Result: `CV AUC 0.7445 +/- 0.0043`.
- Starting code: `n_estimators=30`, `max_depth=6`, `learning_rate=0.1`, native categorical features, no explicit `tree_method`, no sampling, no regularization beyond defaults.
- Why this run: required by the instructions to establish the baseline before changing anything.
- Interpretation: baseline was fast but likely underfit. With categorical airport/carrier/calendar features and only 30 trees, the model probably lacked enough boosting capacity.

### Iteration 1: `d9c5c9e` conservative boosted trees

- Status: keep.
- Result: `CV AUC 0.7568 +/- 0.0048`.
- Change: `n_estimators=160`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=0.85`, `colsample_bytree=0.85`, `tree_method="hist"`, `eval_metric="auc"`.
- Why: XGBoost docs recommend lower learning rate with more rounds for conservative boosting, and sampling/child-weight as overfit controls.
- Interpretation: large gain confirmed baseline underfitting. More trees with shrinkage was immediately valuable.

### Iteration 2: `0971248` extended conservative schedule

- Status: keep.
- Result: `CV AUC 0.7586 +/- 0.0047`.
- Change: `n_estimators=300`, `learning_rate=0.03`.
- Why: test whether the underfit signal from iteration 1 continued with a slower schedule.
- Interpretation: improvement was smaller but positive. More rounds still helped.

### Iteration 3: `426dd6c` slower extended boosting

- Status: keep.
- Result: `CV AUC 0.7592 +/- 0.0045`.
- Change: `n_estimators=500`, `learning_rate=0.02`.
- Why: continue the lower-rate/more-round trend to find the plateau.
- Interpretation: gain persisted but tapered. Runtime was still well within the budget, so it was worth keeping.

### Iteration 4: `76d4a44` longer low-rate boosting

- Status: keep.
- Result: `CV AUC 0.7614 +/- 0.0050`.
- Change: `n_estimators=800`, `learning_rate=0.015`.
- Why: one more schedule extension after 500 rounds still helped.
- Interpretation: unexpectedly large gain versus iteration 3. The model still needed more boosting capacity.

### Iteration 5: `2844201` classic low-rate schedule

- Status: discard.
- Result: `CV AUC 0.7614 +/- 0.0049`.
- Change: `n_estimators=1200`, `learning_rate=0.01`.
- Why: test a classic very-low-rate schedule.
- Interpretation: tied the rounded AUC but cost more runtime. By the simplicity criterion, 800 rounds at 0.015 was better.

### Iteration 6: `9dff1dd` shallower trees

- Status: discard.
- Result: `CV AUC 0.7559 +/- 0.0045`.
- Change: `max_depth=5`.
- Why: test whether depth 6 was overfitting and a shallower model would generalize better.
- Interpretation: large drop. The model needed interactions that depth 5 could not express.

### Iteration 7: `5e50bb0` deeper regularized trees

- Status: keep.
- Result: `CV AUC 0.7696 +/- 0.0048`.
- Change: `max_depth=7`, `min_child_weight=10`.
- Why: depth 5 underfit, so try more depth with stronger child support.
- Interpretation: major gain. Depth was a key axis, and `min_child_weight` controlled the overfit risk.

### Iteration 8: `3746f13` deeper constrained trees

- Status: keep.
- Result: `CV AUC 0.7746 +/- 0.0050`.
- Change: `max_depth=8`, `min_child_weight=20`.
- Why: follow the depth-plus-support pattern one step further.
- Interpretation: another large gain. High-order interactions were valuable.

### Iteration 9: `c063231` very deep constrained trees

- Status: keep.
- Result: `CV AUC 0.7754 +/- 0.0053`.
- Change: `max_depth=9`, `min_child_weight=40`.
- Why: continue depth/child-weight scaling.
- Interpretation: smaller gain but still positive. The trend was tapering but not finished.

### Iteration 10: `cda04be` one-hot categorical splits

- Status: discard.
- Result: `CV AUC 0.7626 +/- 0.0055`.
- Change: added `max_cat_to_onehot=64`.
- Why: XGBoost categorical docs describe one-hot versus partition splits; low/medium-cardinality fields might benefit from one-hot split isolation.
- Interpretation: large drop. Partition-based categorical splits were much better for this data.

### Iteration 11: `408ee25` larger categorical partition threshold

- Status: discard.
- Result: `CV AUC 0.7751 +/- 0.0054`.
- Change: added `max_cat_threshold=128`.
- Why: try allowing larger airport category partitions per split.
- Interpretation: slightly worse. Default categorical threshold was adequate or better regularized.

### Iteration 12: `fa6b2e3` smaller categorical partition threshold

- Status: discard.
- Result: `CV AUC 0.7754 +/- 0.0055`.
- Change: added `max_cat_threshold=32`.
- Why: try regularizing airport partitions more strongly.
- Interpretation: tied rounded best but added complexity. Default remained preferable.

### Iteration 13: `a57f0de` departure time features

- Status: discard.
- Result: `CV AUC 0.7753 +/- 0.0054`.
- Change: added `DepHour` and `DepMinutes` from `DepTime` inside `prepare(df)`.
- Why: BTS delay docs made time of day a plausible operational signal; HHMM encoding may be awkward for trees.
- Interpretation: slight loss. Deep trees already extracted enough from raw `DepTime`.

### Iteration 14: `bdc49a3` stronger sampling regularization

- Status: discard.
- Result: `CV AUC 0.7746 +/- 0.0053`.
- Change: `subsample=0.75`, `colsample_bytree=0.75`.
- Why: test whether the deep model needed more randomness to reduce variance.
- Interpretation: worse. The model needed more row/feature exposure to learn sparse interactions.

### Iteration 15: `5d76b9b` lighter sampling regularization

- Status: discard.
- Result: `CV AUC 0.7752 +/- 0.0054`.
- Change: `subsample=0.95`, `colsample_bytree=0.95`.
- Why: since heavier sampling hurt, test more exposure.
- Interpretation: also worse at that stage. The old `0.85/0.85` mix was a local sweet spot before later row-only tuning.

### Iteration 16: `2ae42f3` stronger L2 regularization

- Status: discard.
- Result: `CV AUC 0.7749 +/- 0.0054`.
- Change: `reg_lambda=5`.
- Why: regularize leaf weights in very deep trees.
- Interpretation: over-regularized. High `min_child_weight` was already doing the useful regularization.

### Iteration 17: `55f7beb` relaxed L2 regularization

- Status: keep.
- Result: `CV AUC 0.7755 +/- 0.0053`.
- Change: `reg_lambda=0.5`.
- Why: if high L2 hurt, less-than-default L2 might help supported leaves fit signal.
- Interpretation: tiny gain. Leaf weights benefited from less L2.

### Iteration 18: `eba944b` no L2 regularization

- Status: keep.
- Result: `CV AUC 0.7757 +/- 0.0054`.
- Change: `reg_lambda=0`.
- Why: continue the trend from `reg_lambda=0.5`.
- Interpretation: another small gain. L2 was unnecessary under high child support.

### Iteration 19: `d610e9c` small split-gain threshold

- Status: discard.
- Result: `CV AUC 0.7756 +/- 0.0054`.
- Change: `gamma=0.1`.
- Why: prune weak deep splits while leaving leaf weights unconstrained.
- Interpretation: slightly worse. Additional split-gain regularization was not needed.

### Iteration 20: `15a0f69` lossguide tree growth

- Status: discard.
- Result: `CV AUC 0.7757 +/- 0.0054`.
- Change: `grow_policy="lossguide"`, `max_leaves=512`.
- Why: XGBoost tree-method docs say lossguide can focus splits by highest loss change and supports `max_leaves` with `hist`.
- Interpretation: tied rounded best but was much slower. Depthwise remained better by simplicity/runtime.

### Iteration 21: `2bc3828` higher histogram resolution

- Status: keep.
- Result: `CV AUC 0.7760 +/- 0.0054`.
- Change: `max_bin=512`.
- Why: tree-method docs say higher `max_bin` can improve split optimality for `hist`.
- Interpretation: clean small gain. Numeric split resolution mattered.

### Iteration 22: `ae1b2c3` very high histogram resolution

- Status: keep.
- Result: `CV AUC 0.7762 +/- 0.0054`.
- Change: `max_bin=1024`.
- Why: follow-up on the `max_bin=512` gain.
- Interpretation: another small gain with acceptable runtime.

### Iteration 23: `97e92b6` max histogram resolution probe

- Status: keep.
- Result: `CV AUC 0.7764 +/- 0.0056`.
- Change: `max_bin=2048`.
- Why: see if numeric split resolution still limited the model.
- Interpretation: small gain despite increased runtime. Kept because still under budget.

### Iteration 24: `95bce1f` extreme histogram resolution

- Status: discard.
- Result: `CV AUC 0.7764 +/- 0.0056`.
- Change: `max_bin=4096`.
- Why: find the point where split resolution stops helping.
- Interpretation: tied 2048 without improvement, so 2048 was the simpler setting.

### Iteration 25: `9b6922a` deeper higher-support trees

- Status: discard.
- Result: `CV AUC 0.7724 +/- 0.0058`.
- Change: `max_depth=10`, `min_child_weight=80`.
- Why: continue depth trend while doubling child support.
- Interpretation: much worse. Doubling child support over-constrained the model.

### Iteration 26: `9001593` depth 10 same child support

- Status: keep.
- Result: `CV AUC 0.7777 +/- 0.0056`.
- Change: `max_depth=10`, keep `min_child_weight=40`.
- Why: isolate depth from the failed child-weight increase.
- Interpretation: strong improvement. The previous failure was child weight, not depth.

### Iteration 27: `e4fdacb` depth 11 trees

- Status: keep.
- Result: `CV AUC 0.7787 +/- 0.0055`.
- Change: `max_depth=11`.
- Why: depth 10 worked, so test if the trend continued.
- Interpretation: depth continued to help.

### Iteration 28: `bef667f` depth 12 trees

- Status: keep.
- Result: `CV AUC 0.7795 +/- 0.0057`.
- Change: `max_depth=12`.
- Why: continue depth sweep under fixed child support.
- Interpretation: still improving, with rising runtime.

### Iteration 29: `ca42c5c` depth 13 trees

- Status: keep.
- Result: `CV AUC 0.7798 +/- 0.0056`.
- Change: `max_depth=13`.
- Why: one more depth step.
- Interpretation: gain was smaller but positive.

### Iteration 30: `9d3e816` depth 14 trees

- Status: keep.
- Result: `CV AUC 0.7800 +/- 0.0057`.
- Change: `max_depth=14`.
- Why: depth trend had not fully stopped.
- Interpretation: small positive gain. Depth was near plateau.

### Iteration 31: `ae01d40` depth 15 trees

- Status: discard.
- Result: `CV AUC 0.7800 +/- 0.0055`.
- Change: `max_depth=15`.
- Why: test whether depth trend continued.
- Interpretation: tied depth 14 but was deeper/slower. Discarded by simplicity.

### Iteration 32: `fa28534` relaxed child support at depth 14

- Status: keep.
- Result: `CV AUC 0.7822 +/- 0.0054`.
- Change: `min_child_weight=30` at depth 14.
- Why: depth 14 with child 40 may have been underusing capacity.
- Interpretation: strong gain. At high depth, child support could be lower than 40.

### Iteration 33: `206181a` lower child support

- Status: keep.
- Result: `CV AUC 0.7834 +/- 0.0049`.
- Change: `min_child_weight=25`.
- Why: follow the child-support relaxation from 30.
- Interpretation: strong gain, but runtime reached the practical boundary.

### Iteration 34: `bba5906` shorter deep schedule

- Status: discard.
- Result: `CV AUC 0.7831 +/- 0.0051`.
- Change: `n_estimators=700`, `learning_rate=0.017`.
- Why: try reducing runtime while preserving boosting strength.
- Interpretation: faster but worse. The 800-round schedule still mattered.

### Iteration 35: `6c8284d` lower bin resolution at depth 14

- Status: discard.
- Result: `CV AUC 0.7829 +/- 0.0053`.
- Change: `max_bin=1024`.
- Why: see if high depth made `max_bin=2048` unnecessary.
- Interpretation: worse. High bin resolution remained justified.

### Iteration 36: `b916663` route categorical interaction

- Status: crash.
- Result: timed out, logged as `0.0000`.
- Change: added `Route = Origin + "-" + Dest` as categorical feature in `prepare(df)`.
- Why: BTS context and airline domain intuition suggest route-specific delay behavior.
- Interpretation: too expensive for the near-cap model. It timed out before producing CV output.

### Iteration 37: `8ea18eb` numeric calendar features

- Status: crash.
- Result: timed out, logged as `0.0000`.
- Change: added numeric versions of Month, DayOfMonth, and DayOfWeek.
- Why: expose ordered seasonal/calendar signals alongside categorical versions.
- Interpretation: even lightweight extra features pushed runtime over the cap at this model size.

### Iteration 38: `0d89941` gentler schedule compression

- Status: discard.
- Result: `CV AUC 0.7833 +/- 0.0050`.
- Change: `n_estimators=750`, `learning_rate=0.016`.
- Why: preserve most of the 800-round model while reducing runtime.
- Interpretation: close but below best. Discarded.

### Iteration 39: `65d556c` depth 13 with lower child support

- Status: discard.
- Result: `CV AUC 0.7830 +/- 0.0052`.
- Change: `max_depth=13`, `min_child_weight=25`.
- Why: reduce depth/runtime while preserving lower child support.
- Interpretation: worse. Depth 14 provided real signal.

### Iteration 40: `1736b5d` per-node column sampling

- Status: discard.
- Result: `CV AUC 0.7815 +/- 0.0055`.
- Change: `colsample_bynode=0.8`.
- Why: XGBoost RF docs discuss per-split feature sampling as randomization.
- Interpretation: hurt badly. Deep split paths need reliable access to all core features.

### Iteration 41: `7e90b07` full column exposure

- Status: crash.
- Result: timed out, logged as `0.0000`.
- Change: `colsample_bytree=1.0`.
- Why: isolate whether more feature exposure improves interactions.
- Interpretation: too expensive. Column sampling was helping runtime.

### Iteration 42: `6cf99c2` slightly lower row sampling

- Status: discard.
- Result: `CV AUC 0.7825 +/- 0.0052`.
- Change: `subsample=0.8`.
- Why: try cheaper/more regularized row sampling.
- Interpretation: worse. Row exposure was important.

### Iteration 43: `14ef49e` conservative logistic step

- Status: discard.
- Result: `CV AUC 0.7834 +/- 0.0055`.
- Change: `max_delta_step=1`.
- Why: XGBoost docs mention it for conservative logistic updates.
- Interpretation: tied previous best but added complexity and some runtime. Discarded.

### Iteration 44: `818d803` slightly lower child support

- Status: crash.
- Result: timed out, logged as `0.0000`.
- Change: `min_child_weight=23`.
- Why: test if the child-support gain continued below 25.
- Interpretation: runtime exceeded the cap. Child 25 became the practical lower limit.

### Iteration 45: `439dad3` slightly higher learning rate

- Status: keep.
- Result: `CV AUC 0.7837 +/- 0.0050`.
- Change: `learning_rate=0.016`.
- Why: same number of trees but slightly stronger updates might reduce residual underfit.
- Interpretation: small gain. Model was still slightly underfit at 0.015.

### Iteration 46: `b78c4ac` higher deep learning rate

- Status: keep.
- Result: `CV AUC 0.7840 +/- 0.0051`.
- Change: `learning_rate=0.017`.
- Why: follow the gain from 0.016.
- Interpretation: another gain. Became the local learning-rate best before row-sampling tuning.

### Iteration 47: `355862a` learning rate 0.018

- Status: discard.
- Result: `CV AUC 0.7836 +/- 0.0053`.
- Change: `learning_rate=0.018`.
- Why: test whether stronger updates kept helping.
- Interpretation: overshot. 0.017 was better.

### Iteration 48: `c52a0f5` learning rate 0.0175

- Status: discard.
- Result: `CV AUC 0.7840 +/- 0.0051`.
- Change: `learning_rate=0.0175`.
- Why: midpoint between 0.017 and 0.018.
- Interpretation: tied rounded AUC but not better. Keep simpler 0.017.

### Iteration 49: `3434060` slightly higher child support

- Status: discard.
- Result: `CV AUC 0.7835 +/- 0.0053`.
- Change: `min_child_weight=27`.
- Why: see whether a little more leaf support improves regularization under learning rate 0.017.
- Interpretation: worse. Child 25 remained best.

### Iteration 50: `fa7622d` learning rate 0.0172

- Status: discard.
- Result: `CV AUC 0.7838 +/- 0.0053`.
- Change: `learning_rate=0.0172`.
- Why: finer micro-tune near 0.017.
- Interpretation: lower than best. 0.017 remained best.

### Iteration 51: `0c1251b` slightly shorter stronger schedule

- Status: discard.
- Result: `CV AUC 0.7837 +/- 0.0052`.
- Change: `n_estimators=780`, `learning_rate=0.0174`.
- Why: preserve total boosting strength with fewer trees.
- Interpretation: lower AUC. Full 800 rounds were still useful.

### Iteration 52: `1fe0af2` child support boundary

- Status: discard.
- Result: `CV AUC 0.7840 +/- 0.0053`.
- Change: `min_child_weight=24`.
- Why: midpoint between best 25 and timed-out 23.
- Interpretation: tied but did not improve. Keep 25.

### Iteration 53: `b04ea1a` row sampling 0.875

- Status: keep.
- Result: `CV AUC 0.7844 +/- 0.0053`.
- Change: `subsample=0.875`.
- Why: earlier row/column combined sampling tests suggested too little row exposure hurt; try row-only upward adjustment.
- Interpretation: good gain. Row sampling was a productive late axis.

### Iteration 54: `4d79cc8` row sampling 0.9

- Status: keep.
- Result: `CV AUC 0.7847 +/- 0.0054`.
- Change: `subsample=0.9`.
- Why: follow the 0.875 improvement.
- Interpretation: final best. More row exposure helped without exceeding timeout.

### Iteration 55: `73a2c7a` row sampling 0.925

- Status: discard.
- Result: `CV AUC 0.7845 +/- 0.0051`.
- Change: `subsample=0.925`.
- Why: test if row exposure trend continued.
- Interpretation: lower. Row sampling peak was around 0.9.

### Iteration 56: `34ae542` column sampling 0.875

- Status: crash.
- Result: timed out, logged as `0.0000`.
- Change: `colsample_bytree=0.875`.
- Why: see whether more column exposure improved interactions after row sampling improved.
- Interpretation: timed out before CV output. Column sampling could not be increased.

### Iteration 57: `93c044d` column sampling 0.825

- Status: discard.
- Result: `CV AUC 0.7847 +/- 0.0054`.
- Change: `colsample_bytree=0.825`.
- Why: test whether slightly lower column sampling preserved AUC with cheaper/regularized trees.
- Interpretation: tied rounded best but did not improve. Keep 0.85.

### Iteration 58: `244616f` learning rate with higher row sampling

- Status: discard.
- Result: `CV AUC 0.7847 +/- 0.0053`.
- Change: `learning_rate=0.0175` under `subsample=0.9`.
- Why: row sampling improvement could have shifted the learning-rate optimum.
- Interpretation: tied but did not beat 0.017.

### Iteration 59: `9778fb3` row sampling 0.9125

- Status: crash.
- Result: timed out, logged as `0.0000`.
- Change: `subsample=0.9125`.
- Why: midpoint between 0.9 and 0.925.
- Interpretation: exceeded runtime boundary. Confirms 0.9 is the best feasible row-sampling setting.

## What Worked

- More boosting rounds with lower learning rate worked strongly early.
- Very deep trees worked, especially depths 10 through 14.
- `min_child_weight` was the crucial regularizer. It needed to be high enough to support deep leaves, but not so high that the model underfit.
- Removing L2 regularization worked after child-weight regularization was in place.
- Higher `max_bin` worked up to `2048`.
- Raising `learning_rate` from 0.015 to 0.017 worked after the model reached high depth and lower child support.
- Increasing row sampling to `0.9` worked near the end.

## What Did Not Work

- Shallower trees did not work.
- One-hot categorical split strategy did not work.
- Categorical threshold tuning did not improve default behavior.
- Feature additions did not survive the final runtime constraints.
- `lossguide` was too slow for no gain.
- `gamma` did not improve deep split quality.
- Higher L2 regularization hurt.
- Per-node column sampling hurt.
- Full column exposure and slightly higher column exposure timed out.
- Lower row sampling hurt.
- Lower child weight below 25 timed out.
- Schedule compression lost AUC.

## Best Current Theory

The dataset is dominated by sparse but real high-order interactions among calendar fields, carrier, airport origin/destination, departure time, and distance. XGBoost's native categorical partitioning plus deep trees can discover these interactions directly from the raw columns. The right model is therefore high capacity, but it needs support constraints and sampling to avoid fitting unsupported leaves.

The best model is not feature-rich. It is parameter-rich in a very specific way: deep enough to express interactions, enough trees to refine them, enough histogram bins for numeric split precision, and enough row exposure to stabilize split decisions. Most hand-engineered features add redundant or expensive signals that the deep native categorical model already handles.

## Practical Next Ideas

- Try only same-cost micro-tuning unless the timeout rule changes.
- Candidate low-risk tests: `subsample=0.895`, `subsample=0.905`, `learning_rate=0.0168`, or `n_estimators=810` only if timeout margin allows.
- Candidate code-level speed refactor: investigate native XGBoost CV or reusable DMatrix/QuantileDMatrix, but only if it preserves the same evaluation semantics and categorical handling.
- Do not add features under the current model size unless paired with a faster configuration, because even small feature additions timed out.
- Do not raise `colsample_bytree` above 0.85 under the current settings because 0.875 timed out.
- Do not lower `min_child_weight` below 25 under the current settings because 23 timed out and 24 only tied.
