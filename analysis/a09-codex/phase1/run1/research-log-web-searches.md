# Web Search Log

This file records the web searches performed during the experiment run and during the later detailed-log writeup. It includes the search queries and the URLs from which I received search-result summary/snippet information or otherwise used research information.

## Search Batch 1: Before First Non-Baseline Experiment

Purpose: collect general XGBoost tuning guidance and flight-delay feature ideas before modifying the baseline.

Queries:

- `XGBoost documentation parameters binary classification tree booster max_depth min_child_weight subsample colsample_bytree eta`
- `XGBoost parameter tuning guide binary classification AUC early stopping`
- `feature engineering airline delay prediction departure delay machine learning`

Information used:

- XGBoost parameter/tuning concepts: `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `learning_rate`/eta, regularization.
- Flight-delay domain concepts: carrier, origin, destination, distance, day/date, and flight/departure time are natural predictors.

URLs represented in later source summaries and used for these ideas:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10897135/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/

## Search Batch 2: After 10 Experiments

Purpose: refresh research after the first synthesis point; look for additional regularization, categorical handling, and time-feature ideas.

Queries:

- `XGBoost subsample colsample_bytree regularization min_child_weight gamma tuning official documentation`
- `XGBoost categorical features parameters max_cat_to_onehot max_cat_threshold documentation`
- `XGBoost feature engineering time cyclic features hour of day tabular airline delay prediction`

Information used:

- `gamma` as minimum split-loss reduction.
- `subsample` and `colsample_bytree` as stochastic regularizers.
- `max_cat_to_onehot` and categorical partitioning/one-hot controls.
- Flight-delay time-of-day and route/carrier/airport feature relevance.

URLs represented in later source summaries and used for these ideas:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10897135/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/

## Search Batch 3: After 30 Experiments

Purpose: research next ideas after feature engineering failed and deep-tree tuning was working; focus on depth, categorical split controls, and grow policy.

Queries:

- `XGBoost deep trees max_depth min_child_weight regularization overfitting AUC tuning`
- `XGBoost categorical split max_cat_threshold max_cat_to_onehot tuning high cardinality categorical`
- `XGBoost grow_policy lossguide max_leaves depthwise documentation`

Information used:

- Deep trees are more complex and can overfit or become memory/runtime heavy.
- `min_child_weight` and depth should be tuned together.
- `max_cat_to_onehot` changes categorical split strategy.
- `grow_policy="lossguide"` and `max_leaves` are possible future directions for selective deep growth.

URLs represented in later source summaries and used for these ideas:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

## Search Batch 4: Detailed Log Source Reconstruction

Purpose: while writing `research-log-after-details.md`, I re-ran targeted searches to recover explicit URLs and source snippets for the research claims.

Queries:

- `XGBoost parameters max_depth min_child_weight subsample colsample_bytree gamma documentation`
- `XGBoost categorical data max_cat_to_onehot max_cat_threshold documentation`
- `XGBoost grow_policy lossguide max_leaves documentation`
- `airline delay prediction feature engineering departure time origin destination carrier machine learning`

Domains filter:

- The first three queries were restricted to `xgboost.readthedocs.io`.
- The flight-delay query was restricted to `pmc.ncbi.nlm.nih.gov`.

URLs returned with summary/snippet information:

- https://xgboost.readthedocs.io/en/stable/parameter.html?highlight=gblinear
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10897135/
- https://xgboost.readthedocs.io/en/release_1.6.0/dev/categorical_8h.html
- https://xgboost.readthedocs.io/en/release_1.6.0/gpu/
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html
- https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/rf.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/
- https://xgboost.readthedocs.io/_/downloads/en/release_3.0.0/pdf/
- https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/slicing_model.html
- https://xgboost.readthedocs.io/en/release_1.3.0/dev/classxgboost_1_1common_1_1ColumnSampler.html
- https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/saving_model.html
- https://xgboost.readthedocs.io/en/release_3.2.0/R-package/index_base.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/
- https://xgboost.readthedocs.io/_/downloads/en/release_1.6.0/pdf/
- https://xgboost.readthedocs.io/_/downloads/en/latest/pdf/
- https://xgboost.readthedocs.io/_/downloads/en/stable/pdf/

Notes on information extracted:

- From the XGBoost parameter docs/snippets: definitions and roles of `gamma`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, `grow_policy`, `max_leaves`, and categorical parameters.
- From tree-method docs/snippets: `grow_policy` and `max_leaves` support with `hist`/`approx` methods.
- From categorical docs/snippets: `max_cat_to_onehot` controls one-hot versus partition-based categorical splits.
- From flight-delay papers/snippets: relevant predictors include airline/carrier, departure/arrival airports, flight day/time, distance, origin/destination, and delay-related operational factors.

## Search Batch 5: Categorical Documentation Follow-Up

Purpose: get a direct stable XGBoost categorical tutorial URL and related categorical examples.

Queries:

- `site:xgboost.readthedocs.io/en/stable categorical data max_cat_to_onehot XGBoost categorical tutorial`
- `XGBoost categorical data tutorial max_cat_to_onehot max_cat_threshold stable`

URLs returned with summary/snippet information:

- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/release_3.1.0/python/examples/categorical.html
- https://xgboost.readthedocs.io/en/stable/index.html
- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/release_1.6.0/dev/categorical_8h.html
- https://xgboost.readthedocs.io/en/stable/tutorials/
- https://xgboost.readthedocs.io/en/release_3.2.0/python/examples/cat_in_the_dat.html
- https://xgboost.readthedocs.io/en/release_3.2.0/python/examples/cat_pipeline.html
- https://xgboost.readthedocs.io/en/release_3.0.0/
- https://xgboost.readthedocs.io/en/release_0.80/R-package/discoverYourData.html
- https://xgboost.readthedocs.io/_/downloads/en/stable/pdf/
- https://xgboost.readthedocs.io/_/downloads/en/release_1.6.0/pdf/
- https://xgboost.readthedocs.io/_/downloads/en/release_3.0.0/pdf/

Reddit URLs returned with summary/snippet information in this search batch:

- https://www.reddit.com/r/MachineLearning/comments/fpnu63
- https://www.reddit.com/r/datascience/comments/145d04o
- https://www.reddit.com/r/MLQuestions/comments/1fxzj62
- https://www.reddit.com/r/learnmachinelearning/comments/zx4jk9
- https://www.reddit.com/r/askdatascience/comments/1p2qcn0/handling_high_missingness_and_high_cardinality_in/
- https://www.reddit.com/r/learnmachinelearning/comments/17ojwdf
- https://www.reddit.com/r/learnmachinelearning/comments/19ea3fe
- https://www.reddit.com/r/learnmachinelearning/comments/16b613p
- https://www.reddit.com/r/MachineLearning/comments/cbirf7
- https://www.reddit.com/r/MachineLearning/comments/16mthsy
- https://www.reddit.com/r/learnmachinelearning/comments/11ny13t
- https://www.reddit.com/r/learnmachinelearning/comments/qinh58

Notes on information extracted:

- The stable categorical tutorial was the main source used from this batch.
- The Reddit results were returned as search snippets, but I did not base the experiment decisions on Reddit content. They are listed here because summary/snippet information was returned by the search tool.

## URLs Actually Cited In `research-log-after-details.md`

These are the source URLs explicitly cited in the detailed research log:

- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10897135/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/

## Caveats

- I did not open and read full pages for every URL listed above. Some URLs were used through search-result summaries/snippets only.
- The first three search batches were conducted during the experiment loop. Their detailed search result lists were not preserved verbatim in the local repository at the time, so this log reconstructs their used source categories and URLs from the later targeted searches plus the written experiment notes.
- The later source-reconstruction searches did return explicit URL lists, and those are recorded above.
