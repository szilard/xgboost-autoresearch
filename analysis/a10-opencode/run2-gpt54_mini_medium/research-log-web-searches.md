# Web Search Log

## 1. XGBoost parameter tuning

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html`
- Why: first non-baseline experiment; needed guidance on tuning XGBoost for binary classification.
- Info used:
  - `subsample` and `colsample_bytree` help reduce overfitting.
  - Lower `eta` often needs more boosting rounds.
  - `scale_pos_weight` is useful for imbalanced data, but the dataset here is already balanced by construction.
  - `min_child_weight`, `gamma`, `max_depth`, and `reg_lambda` are the main bias/variance knobs.

## 2. XGBoost parameter reference

- URL: `https://xgboost.readthedocs.io/en/stable/parameter.html`
- Why: to verify exact meanings and valid ranges for tree and categorical parameters.
- Info used:
  - `tree_method="hist"` is the faster histogram tree builder.
  - `grow_policy="lossguide"` and `max_leaves` control leaf-wise growth.
  - `max_cat_to_onehot` and `max_cat_threshold` affect categorical split behavior.
  - `max_bin` affects histogram granularity.
  - `gamma`, `min_child_weight`, `lambda`, and `alpha` are split/regularization knobs.

## 3. XGBoost early stopping

- URL: `https://xgboost.readthedocs.io/en/stable/python/python_intro.html#early-stopping`
- Why: the sklearn wrapper in this environment rejected `early_stopping_rounds`, so I checked the native API.
- Info used:
  - Native `xgb.train()` supports `early_stopping_rounds`.
  - `best_iteration` should be used when predicting after early stopping.
  - Native training was the path that made fold-wise early stopping workable.

## 4. XGBoost sklearn estimator docs

- URL: `https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html`
- Why: to see whether early stopping could be used from `XGBClassifier.fit()`.
- Info used:
  - The sklearn wrapper relies on `eval_set` for early stopping.
  - In this environment, `fit()` rejected both `early_stopping_rounds` and `callbacks`, which pushed the experiment to the native API.

## 5. Cyclical feature engineering

- URL: `https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html`
- Why: to justify sine/cosine encoding of periodic time features.
- Info used:
  - Periodic variables can be represented with sine/cosine pairs.
  - This is a standard way to preserve wraparound structure for month/day/hour-like variables.
  - The default scaling uses the observed max value unless manually overridden.

## 6. Scikit-learn time-related feature engineering

- URL: `https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html`
- Why: second source on periodic/time features and a sanity check that cyclic encoding is mainstream.
- Info used:
  - Cyclical encoding is a standard feature-engineering technique for periodic signals.
  - Time features often benefit from explicit periodic treatment rather than raw ordinal values.

## 7. Feature interaction constraints

- URL: `https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html`
- Why: to test whether limiting interactions could improve generalization.
- Info used:
  - Interaction constraints can restrict which features may co-occur on a path.
  - This can reduce spurious interactions, but in this experiment it overconstrained the model.

## 8. Airline-delay / flight-delay search: route and interaction ideas

- Search URL: `https://duckduckgo.com/html/?q=airline+delay+feature+engineering+route+origin+dest+xgboost`
- Search URL: `https://duckduckgo.com/html/?q=flight+delay+prediction+route+feature+engineering`
- Why: to look for domain-specific feature ideas before trying route-level interactions.
- Info obtained from search results:
  - Common suggestions included route/origin/destination composites.
  - Several results referenced flight-delay prediction projects and papers that used route or airport-pair style features.
  - This informed the brief `Route` / `CarrierRoute` experiment, which then timed out.

## 9. Kaggle search

- Search URL: `https://www.kaggle.com/search?q=airline+delay+feature+engineering+route+origin+dest+xgboost`
- Why: to find practical notebooks or solution writeups for similar tabular flight-delay problems.
- Info obtained:
  - Kaggle search results page was reachable and confirmed the general topic area, but no specific notebook was adopted directly.

## Notes

- I did not use web results blindly; they were used to justify specific experiments and parameter choices.
- The most useful external guidance was the XGBoost docs on early stopping, histogram trees, lossguide growth, and categorical split thresholds, plus the cyclical feature-engineering examples.
