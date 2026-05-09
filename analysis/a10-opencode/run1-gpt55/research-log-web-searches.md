# Web Research Log

This file records the web research performed during the `may8` experiment run. I did not use a general search-engine results page through the available tools; the research consisted of direct web fetches of documentation and domain pages. Each entry below lists the URL fetched, the search/research intent, the information used, and the experiment decisions influenced.

## Chronological Fetch Log

### 1. XGBoost Parameters

- URL: `https://xgboost.readthedocs.io/en/stable/parameter.html`
- Research intent: understand XGBoost binary-classification tuning parameters before the first non-baseline experiment.
- Information obtained: `learning_rate`/`eta` shrinkage makes boosting more conservative; `max_depth` controls tree complexity; `min_child_weight` restricts weak child nodes; `subsample` and `colsample_bytree` add randomness; `reg_lambda`, `reg_alpha`, `gamma`, `tree_method`, `max_bin`, `max_cat_to_onehot`, `max_cat_threshold`, `scale_pos_weight`, and `max_delta_step` are relevant tuning levers.
- Used for: initial conservative schedule, depth/child-weight sweeps, sampling experiments, L2/gamma tests, categorical parameter tests, histogram resolution tests, and `max_delta_step=1`.

### 2. XGBoost Categorical Data Tutorial

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html`
- Research intent: understand native categorical support and whether to keep or alter categorical handling.
- Information obtained: pandas categorical dtype plus `enable_categorical=True` is the intended path; categorical data is supported with `hist` and `approx`; XGBoost can use one-hot or partition-based categorical splits; `max_cat_to_onehot` controls that choice; consistent category encoding matters.
- Used for: keeping native categorical handling, using `tree_method="hist"`, testing `max_cat_to_onehot=64`, and testing categorical partition thresholds.

### 3. XGBoost Scikit-Learn Estimator Interface

- URL: `https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html`
- Research intent: understand sklearn wrapper behavior, early stopping, CV interaction, and threading.
- Information obtained: early stopping requires eval sets and can produce different best iterations across CV folds; docs warn that early stopping during CV can be imperfect; sklearn CV plus XGBoost parallelism should avoid thread thrashing by not parallelizing both heavily.
- Used for: retaining `cross_val_score(..., n_jobs=1)` with XGBoost `n_jobs=-1`, not adding early stopping to the existing CV harness.

### 4. XGBoost Parameter Tuning Notes

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html`
- Research intent: refresh tuning strategy after the first 10 experiments.
- Information obtained: XGBoost tuning is mostly bias-variance tradeoff; complexity controls include `max_depth`, `min_child_weight`, `gamma`, `max_cat_threshold`; randomness controls include `subsample` and `colsample_bytree`; lower `eta` usually needs more rounds; understanding the data and preprocessing can be important.
- Used for: depth/min-child-weight continuation, sampling tests, categorical threshold tests, and later simplification attempts.

### 5. XGBoost Feature Interaction Constraints

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html`
- Research intent: understand interaction behavior in trees after deep trees became important.
- Information obtained: variables on a tree path interact; interaction constraints can restrict variable combinations to reduce noise or encode domain knowledge.
- Used for: interpreting depth gains as high-order interaction gains. I did not add constraints because the successful direction required broad interactions rather than restrictions.

### 6. XGBoost `cat_in_the_dat` Example

- URL: `https://xgboost.readthedocs.io/en/stable/python/examples/cat_in_the_dat.html`
- Research intent: look at XGBoost's categorical examples and one-hot versus native categorical comparison.
- Information obtained: XGBoost examples compare builtin categorical support against one-hot encoding; the example uses `max_cat_to_onehot=1` to force optimal partitioning.
- Used for: treating categorical split strategy as a real modeling axis and testing one-hot threshold behavior.

### 7. Bureau of Transportation Statistics Delay Page

- URL: `https://www.bts.gov/topics/airlines-and-airports/airline-time-performance-and-causes-flight-delays`
- Research intent: domain research for airline delay prediction feature ideas.
- Information obtained: departure delay is measured relative to scheduled time with a 15-minute on-time threshold; delay causes include air carrier, weather, national aviation system, late-arriving aircraft, and security; carrier and airport operations are central delay concepts.
- Used for: motivating departure-time features and route-level feature ideas.

### 8. BTS TranStats Delay Cause Page

- URL: `https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp`
- Research intent: follow the BTS delay-cause link for additional domain context.
- Information obtained: the fetched page mostly returned a “Please try it later” page, but it confirmed this was the TranStats/BTS delay-cause endpoint linked from the BTS delay information page.
- Used for: no direct modeling decision beyond confirming the domain source path; the useful delay-cause details came from the BTS page above.

### 9. XGBoost Tree Methods

- URL: `https://xgboost.readthedocs.io/en/stable/treemethod.html`
- Research intent: investigate tree construction options and histogram split behavior.
- Information obtained: `hist` is the fastest approximate tree method; `approx` can sometimes be more accurate for non-constant Hessian objectives but slower; `hist` supports categorical data, `grow_policy`, and `max_leaves`; higher `max_bin` can improve split optimality at extra cost.
- Used for: `grow_policy="lossguide"`, `max_leaves=512`, and `max_bin` sweeps at 512, 1024, 2048, and 4096.

### 10. XGBoost Monotonic Constraints

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html`
- Research intent: consider whether monotonic constraints might improve generalization.
- Information obtained: constraints are useful with strong prior beliefs about monotonic relationships; with `hist`, constraints can make trees unnecessarily shallow by eliminating split candidates.
- Used for: deciding not to test monotonic constraints because `DepTime` and `Distance` do not have obvious monotonic effects and shallow trees had already failed.

### 11. XGBoost Sklearn Eval Results Example

- URL: `https://xgboost.readthedocs.io/en/stable/python/examples/sklearn_evals_result.html`
- Research intent: understand eval metric logging through the sklearn interface.
- Information obtained: fitted XGBoost sklearn estimators can expose eval histories through `evals_result()` when eval sets are supplied.
- Used for: no code change. It confirmed eval logging options, but the existing `cross_val_score` path remained simpler and more comparable.

### 12. XGBoost DART Booster

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/dart.html`
- Research intent: consider dropout-style boosted trees after the depth axis started plateauing.
- Information obtained: DART drops trees to reduce overfitting; it can be slower because dropout prevents prediction-buffer reuse; early stopping may be less stable due to randomness.
- Used for: deciding not to try DART once runtime became tight and the current model was not clearly overfitting.

### 13. XGBoost Parameter Tuning Notes, Second Fetch

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html`
- Research intent: refresh tuning guidance during the later synthesis cycle.
- Information obtained: same as entry 4, with renewed emphasis on complexity controls, randomness controls, and the need to balance predictive power with model complexity.
- Used for: continued runtime-aware HPO and simplification attempts.

### 14. XGBoost Callback Functions

- URL: `https://xgboost.readthedocs.io/en/stable/python/callbacks.html`
- Research intent: investigate callback options and early-stopping/checkpointing mechanics.
- Information obtained: XGBoost has callback APIs including early stopping and checkpointing; custom callbacks can inspect training evaluation logs.
- Used for: deciding not to add callback machinery because it would add complexity and did not directly address the current AUC/runtime bottleneck.

### 15. XGBoost Callback Example

- URL: `https://xgboost.readthedocs.io/en/stable/python/examples/callbacks.html`
- Research intent: look at concrete callback implementation examples.
- Information obtained: examples show custom callback classes and checkpoint callbacks.
- Used for: no code change. It reinforced that callback work would be a larger harness change, not the best next step under the experiment constraints.

### 16. XGBoost Intercept / Base Score

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html`
- Research intent: understand whether changing `base_score` or using base margins could help.
- Information obtained: `base_score` is automatically estimated for selected objectives; `base_margin` can provide per-sample margins and override `base_score`; margins must be on the raw link scale.
- Used for: deciding not to tune `base_score` or implement base-margin stacking. The dataset is balanced and the current model had other more promising same-cost tuning axes.

### 17. XGBoost Random Forests

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/rf.html`
- Research intent: investigate random-forest style XGBoost and per-node column sampling.
- Information obtained: XGBoost random forests use `num_parallel_tree`, row sampling, column sampling, and often `colsample_bynode`; `XGBRFClassifier` is a separate sklearn-style wrapper.
- Used for: testing `colsample_bynode=0.8`. I did not try full random forest mode because the tuned boosted-tree path was working and runtime was tight.

### 18. XGBoost Python API Reference

- URL: `https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier`
- Research intent: inspect lower-level API options such as `DMatrix` and `QuantileDMatrix` for possible speed or memory improvements.
- Information obtained: `DMatrix` is XGBoost's optimized internal data structure; `QuantileDMatrix` directly generates quantized data for `hist` and can save memory; categorical feature handling can be specified through dataframe dtypes or feature types.
- Used for: considering, but not implementing, a native-DMatrix refactor. I did not change the harness because preserving sklearn CV semantics and categorical consistency was more important.

### 19. XGBoost Boost From Prediction Example

- URL: `https://xgboost.readthedocs.io/en/stable/python/examples/boost_from_prediction.html`
- Research intent: understand whether boosting from prior predictions/base margins might provide a new direction.
- Information obtained: base margins require raw prediction margins, not transformed probabilities; a model can be trained from existing prediction margins.
- Used for: no code change. Stacking/boost-from-margin would add complexity and likely exceed runtime constraints.

### 20. Scikit-Learn Cross-Validation Documentation

- URL: `https://scikit-learn.org/stable/modules/cross_validation.html`
- Research intent: confirm CV practices and evaluate whether changing CV strategy or preprocessing approach was appropriate.
- Information obtained: CV avoids evaluating on training data; `StratifiedKFold` preserves class ratios; preprocessing should be learned on training folds when applicable; `cross_val_score` averages fold scores.
- Used for: keeping the existing 5-fold `StratifiedKFold` evaluation harness unchanged.

## Unique Directly Fetched URLs

- `https://xgboost.readthedocs.io/en/stable/parameter.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html`
- `https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html`
- `https://xgboost.readthedocs.io/en/stable/python/examples/cat_in_the_dat.html`
- `https://www.bts.gov/topics/airlines-and-airports/airline-time-performance-and-causes-flight-delays`
- `https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp`
- `https://xgboost.readthedocs.io/en/stable/treemethod.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html`
- `https://xgboost.readthedocs.io/en/stable/python/examples/sklearn_evals_result.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/dart.html`
- `https://xgboost.readthedocs.io/en/stable/python/callbacks.html`
- `https://xgboost.readthedocs.io/en/stable/python/examples/callbacks.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html`
- `https://xgboost.readthedocs.io/en/stable/tutorials/rf.html`
- `https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier`
- `https://xgboost.readthedocs.io/en/stable/python/examples/boost_from_prediction.html`
- `https://scikit-learn.org/stable/modules/cross_validation.html`

## Notes On URLs Seen But Not Independently Fetched

Several fetched documentation pages contained links to papers, examples, source pages, Kaggle pages, and other XGBoost/sklearn references. I did not independently fetch or rely on those linked pages as separate sources during the experiment. The modeling decisions above were based on the contents of the directly fetched URLs listed in this file.
