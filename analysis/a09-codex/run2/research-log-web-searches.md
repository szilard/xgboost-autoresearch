# Web Search And Source Audit Log

This file records the web searches and web source URLs used during the `may7` experiment run and the later retrospective logging work.

Important limitation: the local repository does not store raw search-engine result pages. The reliable record available here is:

- the exact search-query strings I issued,
- the URLs I explicitly opened/fetched,
- the URLs I cited in research notes because I obtained information from them,
- and cases where I only used search-result summaries/snippets without opening a specific result page.

Where I used only search-result summaries and did not open or cite a specific page, I mark the source as `search summaries only`.

## URL Inventory

URLs explicitly opened, cited, or used as source material:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/stable/treemethod.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html
- https://xgboost.readthedocs.io/en/release_2.0.0/tutorials/dart.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
- https://xgboost.readthedocs.io/en/stable/tutorials/rf.html
- https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/release_3.2.0/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/stable/python/python_intro.html
- https://scikit-learn.org/1.5/auto_examples/applications/plot_cyclical_feature_engineering.html
- https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

## Search Log By Phase

### Initial XGBoost Tuning Research

Queries:

- `XGBoost documentation parameter tuning tree booster eta max_depth subsample colsample_bytree min_child_weight gamma`
- `XGBoost official documentation categorical data categorical feature support enable_categorical max_cat_to_onehot`
- `XGBoost official documentation sklearn API early stopping eval_metric auc`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

Information obtained:

- Lower `learning_rate` is commonly paired with more boosting rounds.
- `max_depth`, `min_child_weight`, `gamma`, `subsample`, and `colsample_bytree` are core complexity/regularization parameters.
- Native categorical support requires categorical dtype and `enable_categorical=True`.
- `tree_method="hist"` is compatible with native categorical handling and is the right fast tree method here.

Used for:

- Experiments 1-10.

### Categorical Controls And Lossguide Discovery

Queries:

- `XGBoost categorical parameters max_cat_to_onehot max_cat_threshold official documentation`
- `XGBoost parameters gamma reg_alpha reg_lambda min_split_loss official documentation`
- `XGBoost tree methods grow_policy max_leaves lossguide official documentation`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

Information obtained:

- `max_cat_to_onehot` controls when categorical features use one-hot splits.
- `max_cat_threshold` limits categories considered per categorical split.
- `gamma`, `reg_alpha`, and `reg_lambda` are regularization levers.
- `grow_policy="lossguide"` and `max_leaves` are available with histogram trees.

Used for:

- Experiments 11-13.
- Later lossguide experiments starting at experiment 41.

### Time Feature Engineering And Flight-Delay Feature Search

Queries:

- `paper airline delay prediction departure delay machine learning features day of week origin destination distance`
- `XGBoost official documentation grow_policy lossguide max_leaves max_bin colsample_bylevel colsample_bynode`
- `scikit-learn cyclical feature engineering time features sine cosine documentation`

URLs used:

- https://scikit-learn.org/1.5/auto_examples/applications/plot_cyclical_feature_engineering.html
- https://xgboost.readthedocs.io/en/stable/parameter.html

Search summaries only:

- Flight-delay feature searches surfaced recurring feature categories: departure time, day of week, month/seasonality, airline/carrier, origin, destination, route, and distance.
- I did not open or cite a specific flight-delay paper page during the experiment loop; I used the search summaries as high-level feature-prior evidence.

Information obtained:

- Cyclic sine/cosine encoding is useful for periodic variables.
- Flight-delay models commonly use time-of-day and airport/carrier/spatial variables.

Used for:

- Experiment 14, time-of-day features.
- Experiment 15, removing raw HHMM `DepTime`.
- Experiments 16-17, calendar and route feature attempts.

### Histogram Binning, Sampling, And DART

Queries:

- `XGBoost official documentation max_bin hist tree method higher max_bin accuracy`
- `XGBoost official documentation subsample colsample_bytree overfitting parameters tuning`
- `XGBoost official documentation dart booster dropout parameters binary classification`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html
- https://xgboost.readthedocs.io/en/release_2.0.0/tutorials/dart.html

Information obtained:

- `max_bin` trades histogram resolution against speed/memory.
- Row and column sampling are standard stochastic regularization controls.
- DART uses dropout-style boosting, but can be slower because prediction-buffer reuse is affected.

Used for:

- Experiment 19, `max_bin=512`.
- Experiments 31-40, sampling sweeps.
- DART was not tried because runtime risk was high.

### Feature Interactions And More Schedule Tuning

Queries:

- `flight delay prediction distance departure time day of week features paper machine learning`
- `XGBoost official documentation interaction constraints monotone constraints binary classification parameters`
- `XGBoost official documentation reg_lambda reg_alpha learning_rate n_estimators parameter tuning`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

Search summaries only:

- Flight-delay feature searches again reinforced time, distance, airport, carrier, and calendar predictors.
- No specific flight-delay article URL was opened/cited.

Information obtained:

- Lower learning rates with more trees remained a plausible tuning path.
- Interaction constraints exist but were not tried because the useful interactions were not obvious enough to constrain safely.
- Distance remained a plausible feature-engineering target.

Used for:

- Experiments 20-30.
- Later distance transform experiment 65.

### Column Sampling And Regularization

Queries:

- `XGBoost official documentation colsample_bytree colsample_bylevel colsample_bynode subsample parameters`
- `XGBoost official documentation reg_lambda reg_alpha L1 L2 regularization conservative model`
- `XGBoost official documentation max_delta_step binary logistic imbalanced classification`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/rf.html

Information obtained:

- Column sampling can happen by tree, level, or node.
- Sampling effects compound across column-sampling parameters.
- L1/L2 regularization affects leaf weights.
- `max_delta_step` is mainly relevant to highly imbalanced logistic classification, not this balanced slice.

Used for:

- Experiments 31-36.
- Experiments 61-64.

### Row Subsampling And Stochastic Gradient Boosting

Queries:

- `XGBoost official documentation subsample parameter overfitting stochastic gradient boosting`
- `XGBoost official documentation sampling_method gradient_based subsample hist CPU`
- `XGBoost paper stochastic gradient boosting subsample shrinkage Friedman`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

Information obtained:

- `subsample` samples training rows before growing trees.
- Stochastic gradient boosting motivates row subsampling as a decorrelation/generalization tool.
- Gradient-based sampling was not a practical path here because this run used CPU hist and normal `XGBClassifier` CV.

Used for:

- Experiments 37-40.

### Lossguide And Max Leaves

Queries:

- `XGBoost official documentation grow_policy lossguide depthwise max_leaves hist`
- `XGBoost official documentation reg_lambda reg_alpha gamma parameters regularization`
- `XGBoost official documentation max_depth max_leaves tree_method hist`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/treemethod.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

Information obtained:

- `grow_policy="lossguide"` grows high-loss nodes first.
- `max_leaves` controls leaf-wise tree capacity.
- Histogram tree method is still the appropriate method for this setup.

Used for:

- Experiments 41-50.

### Lossguide Regularization

Queries:

- `XGBoost official documentation max_leaves lossguide min_child_weight regularization`
- `XGBoost official documentation learning_rate max_leaves lossguide overfitting gamma reg_lambda`
- `XGBoost official documentation grow_policy lossguide max_leaves min_child_weight`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/treemethod.html

Information obtained:

- `max_leaves` is the explicit lossguide capacity cap.
- `min_child_weight`, `gamma`, `reg_lambda`, and `reg_alpha` are the main regularizers for large lossguide trees.
- Keep rough aggregate shrinkage similar when trading rounds against learning rate.

Used for:

- Experiments 51-60.

### Gamma And Leaf-Weight Regularization

Queries:

- `XGBoost official documentation gamma min_split_loss reg_lambda reg_alpha tree booster parameters`
- `XGBoost official documentation min_split_loss gamma conservative tree model`
- `XGBoost official documentation reg_lambda reg_alpha tree booster regularization`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html

Information obtained:

- `gamma` is also called `min_split_loss`; it requires a minimum loss reduction before a split is made.
- `reg_lambda` and `reg_alpha` regularize leaf weights rather than split creation.

Used for:

- Experiments 61-63.

### Max Delta Step, Scale Pos Weight, And Feature Weights

Queries:

- `XGBoost official documentation max_delta_step binary logistic parameter`
- `XGBoost official documentation scale_pos_weight binary classification balanced dataset`
- `XGBoost official documentation feature weights DMatrix colsample feature_weights sklearn`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html

Information obtained:

- `max_delta_step` is mainly useful in highly imbalanced logistic settings.
- `scale_pos_weight` is also aimed at class imbalance.
- Feature weights exist, but manually weighting features would be speculative and not clearly justified here.

Used for:

- Deciding not to try `max_delta_step` or `scale_pos_weight`.
- Experiment 64 instead tried `colsample_bynode`.

### Categorical Threshold Retesting

Queries:

- `XGBoost official documentation categorical max_cat_threshold high cardinality overfitting`
- `XGBoost official documentation categorical data parameters max_cat_to_onehot max_cat_threshold`
- `XGBoost official documentation pandas categorical auto recoding`

URLs used:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/stable/parameter.html

Information obtained:

- Category recoding and categorical handling are supported for dataframe categorical columns.
- `max_cat_threshold` can affect categorical partition complexity.
- Retesting categorical thresholds under high-leaf lossguide was plausible because the model capacity regime changed.

Used for:

- Experiments 67-68.

### Deep Histogram Performance And Max Cached Hist Node

Queries:

- `XGBoost official documentation max_cached_hist_node deep trees hist lossguide`
- `XGBoost official documentation max_leaves grow_policy lossguide parameters`
- `XGBoost official documentation tree_method hist max_leaves performance deep trees`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/treemethod.html

Information obtained:

- `max_cached_hist_node` exists for histogram cache behavior in deep trees.
- It was not an obvious AUC lever and therefore was not tried.
- `max_leaves` remained the main actionable lossguide control.

Used for:

- Experiments 71-80.

### Monotone / Interaction Constraints And DART Reconsideration

Queries:

- `XGBoost official documentation monotone_constraints interaction_constraints categorical features`
- `XGBoost official documentation DART booster parameters sample_type normalize_type rate_drop`
- `XGBoost official documentation booster dart dropout prediction buffer slower`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/release_2.0.0/tutorials/dart.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

Information obtained:

- DART remains plausible as regularization but likely too slow and too different for the tight timeout.
- Interaction constraints can restrict which features interact, but there was no safe domain hypothesis strong enough to impose constraints.
- Monotone constraints were not used because the relationship between features and delay probability was not clearly monotonic.

Used for:

- Experiment 80, depth cap instead of DART/constraints.

### Histogram Bin Count And Categorical Runtime

Queries:

- `XGBoost official documentation max_bin categorical hist accuracy speed`
- `XGBoost official documentation enable_categorical QuantileDMatrix categorical memory`
- `XGBoost official documentation eval_metric auc default binary logistic`

URLs used:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/treemethod.html

Information obtained:

- Smaller `max_bin` can speed histogram building but may lose split resolution.
- Categorical support remains tied to dataframe/categorical handling and `enable_categorical=True`.
- Explicit `eval_metric="auc"` was already appropriate.

Used for:

- Experiment 81.

### Early Stopping And Fixed-Round CV

Queries:

- `XGBoost official documentation sklearn estimator callbacks early stopping cross validation`
- `XGBoost official documentation cv early stopping auc xgboost.cv`
- `XGBoost official documentation early stopping sklearn API`

URLs used:

- https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html
- https://xgboost.readthedocs.io/en/stable/python/python_intro.html

Information obtained:

- Early stopping is available in the sklearn API.
- Using early stopping inside the existing `cross_val_score` harness would require a custom CV loop or fold-specific validation handling.
- That would be a larger structural change and would reduce comparability with the previous fixed-round experiments.

Used for:

- Deciding not to add early stopping during this run.
- Experiments 88-90 stayed with fixed-round schedule tuning.

### Final Post-90 Search

Queries:

- `scikit-learn official cyclical feature engineering time of day sine cosine one hot`
- `XGBoost official documentation feature interaction constraints high cardinality categorical`
- `paper flight delay prediction time of day origin destination distance features`

URLs used:

- https://scikit-learn.org/1.5/auto_examples/applications/plot_cyclical_feature_engineering.html
- https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

Search summaries only:

- Flight-delay feature searches again reinforced the same feature families: time-of-day, distance, carrier, origin, destination, and calendar.

Information obtained:

- No new post-90 experiment was run from this search.
- It informed the final synthesis that local HPO was mostly exhausted and future progress likely needs a genuinely new low-cost feature or training idea.

## Retrospective Documentation Source Check

During the later detailed-log writing pass, I also fetched:

- https://xgboost.readthedocs.io/en/stable/parameter.html

Purpose:

- Reconfirm the official XGBoost parameter documentation URL and source framing used in the detailed retrospective.

No new experiment was based on this retrospective fetch.
