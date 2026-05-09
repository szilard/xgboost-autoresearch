# Web Search And URL Audit

This file records the web research performed during the May 8 `may8` experiment run.

Important limitation: the local repo artifacts preserve URLs and source-use notes, but they do not preserve the exact text of every early web search query. For early checkpoints, this file records reconstructed search topics from `research-log.md` and the URLs that were cited or used. For the later post-compaction portion of the run, exact visible search-query strings are included.

No held-out ground-truth files were used for this audit. The source list comes from `research-log.md`, `research-log-after-details.md`, and the visible web-tool calls in the session transcript.

## Exact Visible Web Search Calls

These are the exact search query strings visible from the later part of the session.

### Categorical handling and split controls

Queries:

- `XGBoost categorical data max_cat_to_onehot max_cat_threshold documentation`
- `XGBoost parameters max_cat_threshold max_cat_to_onehot documentation`
- `XGBoost categorical optimal partitioning Fisher proof documentation`

URLs opened or used:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

Information obtained:

- XGBoost supports one-hot or partition-based categorical splits.
- `max_cat_to_onehot` controls when categories use one-hot splitting.
- Partition-based splits explain why high-cardinality airport features should not be one-hot encoded.

### Interaction constraints and split regularization

Queries:

- `XGBoost interaction constraints documentation`
- `XGBoost gamma min_split_loss min_child_weight subsample colsample docs`
- `XGBoost feature interaction constraints examples`

URLs opened or used:

- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
- https://xgboost.readthedocs.io/en/latest/parameter.html

Information obtained:

- Interaction constraints can restrict which features interact, but require strong domain priors.
- `gamma` / `min_split_loss` makes splitting more conservative by requiring loss reduction.
- This motivated trying a small `gamma=0.05` after several feature ablations failed.

### Parameter tuning and column sampling

URLs opened or used directly:

- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

Information obtained:

- `subsample` and column-sampling knobs are documented randomness/regularization controls.
- `colsample_bytree`, `colsample_bylevel`, and `colsample_bynode` compound.
- Tuning guidance supported testing `colsample_bylevel=0.9`, later row-sampling near `subsample=0.975`, and tree-count reductions when runtime became the bottleneck.

### Lossguide growth policy

Queries:

- `XGBoost grow_policy lossguide max_leaves hist documentation`
- `XGBoost lossguide max_leaves max_depth 0 documentation`

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html

Information obtained:

- XGBoost supports `grow_policy="lossguide"` with `tree_method="hist"` and a `max_leaves` cap.
- This motivated experiment 121, which was discarded because it was slower and much worse.

### Feature importance diagnostic

Queries:

- `XGBoost sklearn feature_importances_ gain documentation`
- `XGBoost feature importance sklearn interface documentation`

URLs/source family used:

- XGBoost documentation and sklearn-interface documentation search results.
- The actual diagnostic was run locally with `model.get_booster().get_score(importance_type=...)`.

Information obtained:

- XGBoost boosters can report feature importance by gain, total gain, and weight.
- The local diagnostic showed raw `DepTime`, `Distance`, `Dest`, `Origin`, `DepTimeSin`, and `DepMinute` remained important, which explained failed ablations of raw time/distance signals.

## Reconstructed Web Search Topics And URLs From Logged Research

The exact query strings below are reconstructed topics, not guaranteed verbatim search input. The URLs are the source URLs recorded in the experiment logs.

### Initial XGBoost setup research

Reconstructed search topics:

- XGBoost binary classification parameter tuning
- XGBoost sklearn categorical support pandas categorical enable_categorical
- XGBoost sklearn early stopping cross validation

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/python/examples/categorical.html
- https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html

Information obtained:

- `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, and `reg_alpha` are central XGBoost controls.
- Pandas categorical support requires `enable_categorical=True` and is documented with `tree_method="hist"`.
- Early stopping needs explicit validation sets and can complicate CV/final-model consistency, so the first phase focused on direct HPO instead.

### After 10 experiments: categorical split behavior and regularized boosting

Reconstructed search topics:

- XGBoost categorical one-hot vs partitioning
- XGBoost max_cat_to_onehot max_cat_threshold
- XGBoost regularized tree boosting paper shrinkage column subsampling

URLs used:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://arxiv.org/abs/1603.02754

Information obtained:

- Categorical splits can use one-hot or optimal partitioning.
- `max_cat_threshold` limits partition-based categorical split search and is partly an overfitting control.
- The XGBoost paper reinforced regularization, shrinkage, and column subsampling as core ideas.

### Feature engineering research

Reconstructed search topics:

- cyclical time feature engineering sin cos
- flight delay prediction machine learning feature engineering
- flight delay propagation carrier airport data

URLs used:

- https://scikit-learn.org/1.3/auto_examples/applications/plot_cyclical_feature_engineering.html
- https://arxiv.org/abs/1703.06118
- https://www.nature.com/articles/s41598-020-62871-6

Information obtained:

- Cyclical sine/cosine encodings avoid discontinuities in periodic time variables.
- Flight-delay prediction is a data-driven tabular ML problem where timing, carrier, airport, and distance signals are plausible predictors.
- Delay propagation through carriers and airports supports trying carrier/airport/time interactions carefully.

### After 20 experiments: sampling and regularization knobs

Reconstructed search topics:

- XGBoost subsample colsample_bytree overfitting
- XGBoost gamma reg_lambda reg_alpha documentation
- XGBoost tuning cost subsampling paper

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://arxiv.org/abs/2111.06924

Information obtained:

- `subsample` samples rows per boosting iteration.
- `colsample_bytree`, `colsample_bylevel`, and `colsample_bynode` sample features at different granularities and compound.
- `gamma`, `reg_lambda`, and `reg_alpha` are regularization controls.
- Tuning XGBoost is expensive, making targeted subsampling-related HPO reasonable.

### After 30 experiments: deeper regularization checks

Reconstructed search topics:

- XGBoost colsample_bylevel colsample_bynode
- XGBoost gamma min_split_loss
- XGBoost reg_lambda reg_alpha

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html

Information obtained:

- Level and node column sampling compound with tree-level column sampling.
- `gamma` makes tree growth more conservative.
- L1 and L2 regularization act on leaf weights.

### Plateau after experiment 33: growth policy

Reconstructed search topics:

- XGBoost grow_policy lossguide max_leaves
- XGBoost depthwise vs lossguide

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html

Information obtained:

- `grow_policy` supports `depthwise` and `lossguide`.
- `max_leaves` caps lossguide growth.
- This motivated the first lossguide trial, which did not work.

### After 40 experiments: histogram bins, constraints, and class imbalance

Reconstructed search topics:

- XGBoost max_bin hist
- XGBoost interaction constraints feature names
- XGBoost scale_pos_weight imbalance

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

Information obtained:

- `max_bin` controls histogram split precision and runtime.
- Interaction constraints can restrict feature interactions, but need a defensible grouping.
- `scale_pos_weight` is meant for imbalanced data and was not suitable once the local slice was confirmed balanced.

### Plateau after experiment 43: flight-delay domain features

Reconstructed search topics:

- flight delay XGBoost feature importance departure time carrier
- air travel delay feature engineering distance airport carrier
- flight delay prediction scheduled departure time distance features

URLs used:

- https://www.sciencedirect.com/science/article/pii/S2772415822000050
- https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches
- https://engj.org/index.php/ej/article/download/4376/1156

Information obtained:

- Departure time, carrier, airport/location, and distance-style features are common flight-delay predictors.
- Weather and aircraft-specific fields were not available in this dataset, so the practical feature search stayed within timing, carrier, airport, and distance transformations.

### After 50 experiments: early stopping and prediction handling

Reconstructed search topics:

- XGBoost sklearn early stopping eval_set cross validation
- XGBoost best_iteration prediction sklearn

URLs used:

- https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/stable/prediction.html
- https://xgboost.readthedocs.io/en/stable/parameter.html

Information obtained:

- Early stopping in sklearn uses `eval_set`.
- Predictions can use `best_iteration` after early stopping.
- Implementing this correctly inside the existing `cross_val_score` pattern would require restructuring evaluation, so the loop kept direct HPO.

### After 60 experiments: eta and number of boosting rounds

Reconstructed search topics:

- XGBoost eta learning_rate n_estimators tuning
- XGBoost number of boosting rounds eta

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/latest/r_docs/R-package/docs/reference/xgboost.html

Information obtained:

- `eta` / `learning_rate` controls update shrinkage.
- The optimal number of boosting rounds is problem-dependent.
- This supported the fixed-700-tree learning-rate sweep.

### After 70 experiments: runtime and CV mechanics

Reconstructed search topics:

- XGBoost hist fastest tree method runtime max_depth
- XGBoost max_depth n_estimators min_child_weight runtime
- scikit-learn cross_val_score n_jobs

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

Information obtained:

- `hist` is the practical fast tree method already in use.
- Deep trees, boosting rounds, and lower child weight increase runtime.
- Changing parallelism would be a runtime-only change and risks oversubscription, so the harness stayed unchanged.

### Plateau after experiment 76: categorical encoding alternatives

Reconstructed search topics:

- XGBoost categorical tutorial auto recoding max_cat_to_onehot
- scikit-learn TargetEncoder cross fitting
- XGBoost interaction constraints feature names

URLs used:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://sklearn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

Information obtained:

- XGBoost categorical handling can choose one-hot or partitioning.
- Target encoding is cross-fitted internally in sklearn, but correct use here would require pipeline restructuring and careful ground-truth consistency.
- Interaction constraints remained possible but lacked strong safe domain groups.

### After 80 experiments: categorical partition tuning

Reconstructed search topics:

- XGBoost categorical optimal partitioning max_cat_threshold
- XGBoost max_cat_threshold partition categorical split
- XGBoost category counts one hot threshold behavior

URLs used:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/latest/parameter.html

Information obtained:

- Partition-based categorical splits sort categories by leaf value and group similar categories.
- `max_cat_threshold` controls partition-based categorical split search.
- This justified tuning airport partition thresholds after `max_cat_to_onehot=32`.

### After 90 experiments: managed HPO guidance and categorical recap

Reconstructed search topics:

- XGBoost max_cat_to_onehot max_cat_threshold docs
- XGBoost categorical partitioning Origin Dest
- SageMaker XGBoost hyperparameter tuning eta min_child_weight subsample colsample

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html

Information obtained:

- Low-cardinality features should remain one-hot candidates while airport features should remain partitioned.
- AWS managed tuning guidance reinforced eta, min_child_weight, subsample, and colsample as high-impact knobs.

### After 100 experiments: L1 and high-impact knobs

Reconstructed search topics:

- XGBoost reg_alpha L1 regularization
- XGBoost high impact hyperparameters eta min_child_weight colsample subsample

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html

Information obtained:

- `reg_alpha` is L1 regularization on leaf weights and should not add meaningful runtime.
- This motivated a small L1 test, which was discarded.

### After 110 experiments: tree method, max_bin, and low-cardinality categorical time

Reconstructed search topics:

- XGBoost tree_method hist approx accuracy runtime
- XGBoost max_bin split optimality runtime
- XGBoost categorical low cardinality one-hot

URLs used:

- https://xgboost.readthedocs.io/en/stable/treemethod.html
- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

Information obtained:

- `hist` is already the right fast method for this runtime-constrained loop.
- `max_bin` can improve split precision at compute cost, but earlier direct tests failed.
- Low-cardinality categorical splits motivated `DepHourCat`, which became a retained feature.

### After three discards around experiments 112-114

Exact search queries are listed in the first section under categorical handling and interaction constraints.

URLs used:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

Information obtained:

- The current categorical setup likely worked because low-cardinality categories were one-hot while airports remained partition-based.
- A small `gamma` was a lower-risk regularization test than broad interaction constraints.

### After experiment 119: parameter tuning and class balance

URLs used:

- https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
- https://xgboost.readthedocs.io/en/latest/parameter.html

Local non-web diagnostic:

- Checked label balance locally: 50,000 positive, 50,000 negative.

Information obtained:

- `subsample` and `colsample_bytree` are randomness-based overfitting controls.
- `colsample_bylevel` and `colsample_bynode` compound with `colsample_bytree`.
- Class weighting was rejected because the slice was balanced.

### After 120 experiments: lossguide structural test

Exact search queries are listed above under lossguide growth policy.

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html

Information obtained:

- `grow_policy="lossguide"` and `max_leaves` are supported with histogram trees.
- The resulting experiment was discarded due to poor AUC and runtime.

### After experiment 123: feature importance and categorical interaction counts

Exact search queries are listed above under feature importance diagnostic.

URLs/source family used:

- XGBoost documentation/search-result summaries for feature-importance methods.

Local non-web diagnostics:

- Fit the current-best model and inspected `get_booster().get_score(...)`.
- Counted interaction cardinalities: `CarrierDayOfWeek` 140, `CarrierMonth` 236, origin/destination weekday interactions above 1,800.

Information obtained:

- Raw time and distance features remained important.
- Compact categorical interaction was plausible on cardinality grounds, but the actual carrier-weekday trial failed.

### After experiment 126 and 129: subsample runtime compromise

URLs used:

- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

Information obtained:

- `subsample` samples rows before growing each tree.
- Tuning docs supported reducing rounds when a promising sampling setting exceeded runtime.
- This motivated testing `subsample=0.975` with fewer trees, which still exceeded the runtime cap.

## Unique URL List

All unique URLs recorded as fetched, cited, summarized, or otherwise used:

- https://arxiv.org/abs/1603.02754
- https://arxiv.org/abs/1703.06118
- https://arxiv.org/abs/2111.06924
- https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
- https://engj.org/index.php/ej/article/download/4376/1156
- https://scikit-learn.org/1.3/auto_examples/applications/plot_cyclical_feature_engineering.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
- https://sklearn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches
- https://www.nature.com/articles/s41598-020-62871-6
- https://www.sciencedirect.com/science/article/pii/S2772415822000050
- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/latest/r_docs/R-package/docs/reference/xgboost.html
- https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/prediction.html
- https://xgboost.readthedocs.io/en/stable/python/examples/categorical.html
- https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/stable/treemethod.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
