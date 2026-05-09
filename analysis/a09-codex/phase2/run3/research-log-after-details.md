# Detailed Research Log After Run

Generated after the May 8 experiment run on branch `may8`.

This file is a more explicit companion to `research-log.md` and `results.tsv`. It is written from the actual experiment artifacts: the TSV outcome ledger, the per-experiment research log, the current `train.py` commit path, and diagnostic commands run during the experiment. It does not use held-out ground-truth results.

## Executive Summary

- Dataset/evaluation used during the loop: `data-cache/2005-slice1-100k.csv` with 5-fold stratified CV and AUC.
- Baseline: commit `7d0b368`, CV AUC 0.7445 +/- 0.0043.
- Current valid best: commit `ad96f34`, CV AUC 0.8240, description: learning_rate 0.0348 with DepHourCat.
- Best retained model family: native XGBoost categorical handling, engineered departure-time features, one-hot behavior for low-cardinality categories via `max_cat_to_onehot=32`, partition support for airport categories via `max_cat_threshold=192`, deep hist trees, no L2 leaf penalty, row/column sampling, and a tightly tuned 575-tree boosting schedule.
- Most important improvement blocks: deep-tree capacity plus `min_child_weight`; explicit departure-time features; tree-level column sampling; removal of L2; longer boosting schedule; categorical split tuning; and finally `DepHourCat` plus a small learning-rate increase.
- Most informative near miss: `subsample=0.975` reached 0.8241 twice, but both runs exceeded the practical 60s CV cap, so the valid best remains 0.8240.

## Experimental Methodology

The loop followed `program.md`:

- Only `train.py` was changed for candidate experiments.
- `prepare.py`, `check_groundtruth.py`, `run_groundtruth_all.sh`, `analysis/`, `docs/`, and previous result artifacts were not used for model selection.
- Every candidate was committed before running, then evaluated with `python3 train.py > run.log 2>&1`.
- The decision rule was strict: keep only a candidate that improved printed CV AUC and stayed within the runtime cap; reset to the previous best otherwise.
- Equal printed AUC was treated as a discard unless the change was clearly a simplification. In practice, ties here were not retained because they either added complexity or exceeded runtime.
- Runtime mattered. The printed CV time around 60s was used as the practical cap. A few experiments produced higher AUC but were discarded because they were too slow.
- Research was used at the start, at 10-experiment synthesis points, and at plateaus.

## Data Facts Observed During The Run

- Training slice was class-balanced: 50,000 positive and 50,000 negative labels. That made `scale_pos_weight` unattractive.
- Core categorical cardinalities observed in the training slice: Month 12, DayofMonth 31, DayOfWeek 7, UniqueCarrier 20, Origin 282, Dest 282.
- This cardinality structure explained the biggest categorical breakthrough: `max_cat_to_onehot=32` makes month/day/carrier style fields eligible for one-hot categorical splits, while Origin/Dest remain partition-based.
- Later feature-importance diagnostics on the current-best model showed raw `DepTime`, `Distance`, `Dest`, `Origin`, `DepTimeSin`, and `DepMinute` still had high total gain. That explained why removing raw time or replacing distance with bins did not help.

## Research Source Registry

These are the sources consulted or used as rationale during the experiment, with what each contributed to the search strategy.

- XGBoost parameter documentation: https://xgboost.readthedocs.io/en/stable/parameter.html and https://xgboost.readthedocs.io/en/latest/parameter.html
  - Used for `learning_rate`/`eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`, `gamma`, `reg_lambda`, `reg_alpha`, `max_bin`, `grow_policy`, `max_leaves`, `max_cat_to_onehot`, and `max_cat_threshold`.
  - Key impact: kept the search centered on documented XGBoost levers and clarified which knobs increase conservatism, runtime, or categorical split behavior.
- XGBoost categorical tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
  - Used to understand one-hot vs partition-based categorical split behavior and why low-cardinality and high-cardinality categoricals should be treated differently.
  - Key impact: motivated the `max_cat_to_onehot=32` breakthrough and later `max_cat_threshold` tuning.
- XGBoost categorical sklearn example: https://xgboost.readthedocs.io/en/stable/python/examples/categorical.html
  - Used to confirm that pandas categorical columns with `enable_categorical=True` and `tree_method="hist"` are the intended sklearn-interface route.
- XGBoost sklearn estimator docs: https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html and https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
  - Used to think through early stopping. The conclusion was that early stopping would require restructuring the CV loop and final-model logic, so it was deprioritized.
- XGBoost prediction docs: https://xgboost.readthedocs.io/en/stable/prediction.html
  - Used with early-stopping research to understand `best_iteration` handling in prediction.
- XGBoost tree method docs: https://xgboost.readthedocs.io/en/stable/treemethod.html
  - Used late in the run to confirm that `hist` was already the right speed-oriented tree method and that changing tree methods was unlikely to solve the runtime/accuracy tradeoff.
- XGBoost interaction constraints tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
  - Used to consider whether constraining interactions might reduce noise. Because the dataset lacked strong obvious safe interaction groups, broad constraints were considered risky.
- Chen and Guestrin XGBoost paper: https://arxiv.org/abs/1603.02754
  - Used as conceptual backing for shrinkage, regularized tree boosting, and column subsampling as core generalization mechanisms.
- scikit-learn cyclical feature engineering example: https://scikit-learn.org/1.3/auto_examples/applications/plot_cyclical_feature_engineering.html
  - Used to justify sine/cosine encodings for departure time and calendar fields.
- scikit-learn `cross_val_score` docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
  - Used when considering runtime/parallelism. The loop kept the established CV evaluation and did not change the harness.
- scikit-learn `TargetEncoder` docs: https://sklearn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
  - Used at a plateau. Target encoding was rejected because correct fold-safe usage would require a more complex pipeline and careful ground-truth consistency.
- Flight delay review: https://arxiv.org/abs/1703.06118
  - Used to confirm that flight-delay prediction is commonly treated as a data-driven tabular ML problem.
- Scientific Reports flight delay study: https://www.nature.com/articles/s41598-020-62871-6
  - Used as domain context for carrier/airport delay propagation and why carrier/airport features might matter.
- Flight delay XGBoost study: https://www.sciencedirect.com/science/article/pii/S2772415822000050
  - Used to reinforce departure time and carrier as useful flight-factor signals.
- Berkeley feature-engineering project page: https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches
  - Used as domain inspiration for timing, airport/location, carrier, and distance feature categories.
- Hybrid flight-delay paper feature table: https://engj.org/index.php/ej/article/download/4376/1156
  - Used as another source confirming scheduled departure time and distance as common continuous predictors.
- Kapoor and Perrone tuning paper: https://arxiv.org/abs/2111.06924
  - Used as broad support for the practical cost of XGBoost tuning and the relevance of subsampling-related knobs.
- AWS SageMaker XGBoost tuning guide: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
  - Used to cross-check that eta, min_child_weight, subsample, and colsample parameters are high-impact tuning knobs.
- XGBoost R package docs: https://xgboost.readthedocs.io/en/latest/r_docs/R-package/docs/reference/xgboost.html
  - Used as another formulation of how boosting rounds and eta interact.

## What Worked, In Blocks

### 1. Initial capacity and regularization worked

The starter model was underfit. Increasing rounds, lowering eta, and increasing depth produced steady gains. The biggest early jump came from `min_child_weight`: deep trees wanted capacity, but not tiny leaves. That established the main pattern of the run: high tree capacity was useful only when paired with regularization that made sparse categorical interactions less noisy.

### 2. Departure-time feature engineering worked

Raw HHMM `DepTime` was useful but awkward. Adding `DepHour`, `DepMinute`, `DepMinutes`, and cyclic sine/cosine features gave a large gain. Later ablations confirmed the engineered time stack was not just decorative: removing cyclic features hurt, removing raw `DepTime` hurt, and `DepHourCat` later helped on top.

### 3. Tree-level column sampling was a major HPO win

`colsample_bytree` from 0.9 down to 0.6 improved AUC substantially. The interpretation is that the model had many correlated views of time/calendar/categorical effects, and per-tree feature randomness improved generalization. More aggressive or differently placed column sampling (`0.5`, `bylevel`, `bynode`) hurt or tied.

### 4. Removing L2 worked after sampling was in place

Increasing L2 hurt. Reducing it from default-like settings down to zero improved. Once row sampling, column sampling, and child-weight regularization were doing the main regularization work, explicit L2 leaf-weight penalty became unnecessary and slightly constraining.

### 5. Categorical split tuning was the largest late breakthrough

`max_cat_to_onehot=32` produced the major jump from roughly 0.8088 to 0.8201. The observed category counts explain why: Month, DayofMonth, DayOfWeek, and UniqueCarrier became one-hot-split candidates, while airport features stayed partition-based. Increasing `max_cat_threshold` to 192 then improved airport partitioning. Going too low, too high, or one-hotting airports was harmful.

### 6. Local schedule retuning repeatedly mattered

After each structural change, the best eta/tree-count balance moved. The final valid best came from increasing `learning_rate` from 0.0345 to 0.0348 at the 575-tree child-weight-4 schedule after adding `DepHourCat`.

## What Did Not Work, In Blocks

### 1. Blindly adding correlated feature transforms usually hurt

Calendar cyclic features, distance transforms, distance bins, route category, departure-period bins, quarter-hour category, and carrier-weekday interaction all failed. The successful feature work was not generic feature expansion; it was a targeted representation of departure time that matched known model limitations.

### 2. More conservative regularization often underfit

Higher `min_child_weight` eventually hurt, `gamma` hurt, L1 hurt, and extra column sampling at level/node granularity hurt. The final model needed high capacity, not blanket pruning.

### 3. Some higher-AUC settings were invalid due to runtime

Several candidates printed better or tied AUC but exceeded the practical cap: higher `max_cat_threshold`, lower child weight without enough schedule trimming, higher colsample, high subsample, and lossguide. These were not kept even when the AUC looked attractive.

### 4. Structural alternatives were poor

`grow_policy="lossguide"` was tried twice at different stages and failed both times. Depthwise growth with deep max depth was the stable structure for this dataset and parameter family.

## Phase-By-Phase Interpretation

### Baseline and Experiments 1-10

The model was underfit. More boosting rounds, lower eta, deeper trees, and moderate `min_child_weight` all helped. Depth 4 underfit, depth 14 helped, and `min_child_weight=10` was better than 5 but 20 was too conservative. The best after this block was 0.7747.

### Experiments 11-20

Initial categorical split tweaks did not help, but feature engineering did. Departure-time derived features gave the first strong FE gain. Route, weekend, and calendar cyclic additions were not useful. Deeper trees and a slower schedule then improved modestly.

### Experiments 21-30

Sampling became the strongest HPO direction. Row sampling at 0.95 beat both 0.9 and 0.8, and tree-level column sampling improved steadily down to 0.6. Lower than 0.6 or backing off to 0.65 did not help.

### Experiments 31-40

Extra granular column sampling, gamma, and higher L2 were worse. The useful discovery was in the other direction: lower L2, then no L2, then deeper max depth to 24. This showed the existing sampling/child-weight controls were better regularizers than leaf-weight penalties.

### Experiments 41-50

Histogram bin changes, depth 26, distance transforms, and coarse time buckets failed. Longer boosting schedules worked: 400, 500, 600, and 700 trees with lower eta improved until runtime became tight. Then fixed 700-tree eta retuning started to climb.

### Experiments 51-70

The fixed-700-tree learning-rate curve kept improving up to 0.026, then plateaued around 0.027-0.028. Lowering `min_child_weight` from 10 to 8 to 6 helped strongly, but 4 timed out. Runtime became the active constraint.

### Experiments 71-80

Runtime-aware schedule reductions failed to beat the 700-tree child-weight-6 best. The breakthrough came from categorical handling: `max_cat_to_onehot=32` made low/medium categorical splits much better and produced the largest single late gain. Threshold 24 showed day-of-month one-hot mattered; threshold 512 showed airport one-hot was harmful. `max_cat_threshold=128` added a smaller valid gain.

### Experiments 81-90

Airport partition threshold tuning found 192 as the best valid value. Higher/lower thresholds were worse or too slow. Child weight 5 improved AUC but was initially too slow; cutting to 650 trees and raising eta made it valid, then eta 0.029 became the new best.

### Experiments 91-100

Fine eta retunes around 0.029 did not improve. Higher colsample gave attractive but too-slow AUC. A 625-tree, `colsample_bytree=0.625`, eta 0.031 schedule became a small valid improvement. Attempts to push eta or recover higher-colsample behavior failed.

### Experiments 101-110

L1 regularization failed. Child weight 4 was promising but too slow at 600 trees; a 575-tree schedule with eta 0.0345 made it valid and became the best. Further eta, tree-count, threshold, row-sampling, colsample, and depth tweaks failed or tied while too slow.

### Experiments 111-120

`DepHourCat` was the final feature-engineering win, improving to 0.8238. Quarter-hour categories and time ablations failed. Small gamma failed. A precise eta bump to 0.0348 improved to 0.8240 and became the final valid best. Nearby eta values, longer smoother schedule, and level-wise column sampling failed.

### Experiments 121-130

Structural and feature additions mostly failed: lossguide, depth 23, distance bins, carrier-weekday interaction. The row-sampling band above 0.95 was the only promising direction: `subsample=0.975` produced 0.8241 but missed runtime. Values 0.97 and 0.965 lost the gain and remained too slow. Reducing trees to 570 did not save enough time. The best valid model remained experiment 116.

## Experiment Outcome Index

This table is generated from `results.tsv` and gives one-line coverage for every iteration. The full notes below contain the hypothesis/change/result/decision details.

| Iteration | Commit | CV AUC | Status | Description | Outcome interpretation |
| --- | --- | ---: | --- | --- | --- |
| Baseline | `7d0b368` | 0.7445 | keep | baseline unchanged starter model | established baseline best 0.7445 |
| Exp 1 | `826568d` | 0.7513 | keep | 120 trees lower learning rate explicit hist auc metric | advanced best from 0.7445 to 0.7513 |
| Exp 2 | `f76c3b0` | 0.7517 | keep | 200 trees at 0.03 learning rate | advanced best from 0.7513 to 0.7517 |
| Exp 3 | `8fc0379` | 0.7462 | discard | reduce max_depth to 4 | not retained because it reduced validation AUC |
| Exp 4 | `cc49859` | 0.7552 | keep | increase max_depth to 8 | advanced best from 0.7517 to 0.7552 |
| Exp 5 | `11f1073` | 0.7595 | keep | increase max_depth to 10 | advanced best from 0.7552 to 0.7595 |
| Exp 6 | `3c01149` | 0.7628 | keep | increase max_depth to 12 | advanced best from 0.7595 to 0.7628 |
| Exp 7 | `45c7db5` | 0.7657 | keep | increase max_depth to 14 | advanced best from 0.7628 to 0.7657 |
| Exp 8 | `77ade59` | 0.7731 | keep | add min_child_weight 5 to deep trees | advanced best from 0.7657 to 0.7731 |
| Exp 9 | `6c34e12` | 0.7747 | keep | raise min_child_weight to 10 | advanced best from 0.7731 to 0.7747 |
| Exp 10 | `4aa4e92` | 0.7727 | discard | raise min_child_weight to 20 | not retained because it reduced validation AUC |
| Exp 11 | `7637437` | 0.7746 | discard | max_cat_to_onehot 16 for small categoricals | not retained because it reduced validation AUC |
| Exp 12 | `2ed0c2d` | 0.7745 | discard | max_cat_threshold 32 for categorical regularization | not retained because it reduced validation AUC |
| Exp 13 | `0fac8f8` | 0.7804 | keep | add departure time derived numeric and cyclic features | advanced best from 0.7747 to 0.7804 |
| Exp 14 | `841d3a5` | 0.7793 | discard | add high-cardinality route categorical feature | not retained because it reduced validation AUC |
| Exp 15 | `60714b8` | 0.7802 | discard | add cyclic calendar and weekend features | not retained because it reduced validation AUC |
| Exp 16 | `7ed1b2a` | 0.7818 | keep | raise max_depth to 16 with time features | advanced best from 0.7804 to 0.7818 |
| Exp 17 | `819cae3` | 0.7825 | keep | raise max_depth to 18 with time features | advanced best from 0.7818 to 0.7825 |
| Exp 18 | `97192af` | 0.7827 | keep | raise max_depth to 20 with time features | advanced best from 0.7825 to 0.7827 |
| Exp 19 | `ce852e8` | 0.7807 | discard | raise min_child_weight to 15 | not retained because it reduced validation AUC |
| Exp 20 | `8f4d69d` | 0.7831 | keep | 300 trees at 0.02 learning rate | advanced best from 0.7827 to 0.7831 |
| Exp 21 | `9869075` | 0.7838 | keep | add subsample 0.9 | advanced best from 0.7831 to 0.7838 |
| Exp 22 | `e9844a0` | 0.7826 | discard | lower subsample to 0.8 | not retained because it reduced validation AUC |
| Exp 23 | `b768081` | 0.7840 | keep | light subsample 0.95 | advanced best from 0.7838 to 0.7840 |
| Exp 24 | `798390e` | 0.7870 | keep | add colsample_bytree 0.9 | advanced best from 0.7840 to 0.7870 |
| Exp 25 | `847e7a3` | 0.7897 | keep | lower colsample_bytree to 0.8 | advanced best from 0.7870 to 0.7897 |
| Exp 26 | `89c886f` | 0.7907 | keep | lower colsample_bytree to 0.7 | advanced best from 0.7897 to 0.7907 |
| Exp 27 | `a15efee` | 0.7935 | keep | lower colsample_bytree to 0.6 | advanced best from 0.7907 to 0.7935 |
| Exp 28 | `3642af5` | 0.7926 | discard | lower colsample_bytree to 0.5 | not retained because it reduced validation AUC |
| Exp 29 | `31a6b74` | 0.7935 | discard | colsample_bytree 0.55 tied best | not retained because it only tied the current best at printed precision |
| Exp 30 | `4766104` | 0.7919 | discard | colsample_bytree 0.65 | not retained because it reduced validation AUC |
| Exp 31 | `cb64df2` | 0.7931 | discard | add colsample_bynode 0.9 | not retained because it reduced validation AUC |
| Exp 32 | `badf2cc` | 0.7935 | discard | add gamma 0.1 split regularization | not retained because it only tied the current best at printed precision |
| Exp 33 | `81514b9` | 0.7920 | discard | raise reg_lambda to 2 | not retained because it reduced validation AUC |
| Exp 34 | `9f82898` | 0.7893 | discard | grow_policy lossguide max_leaves 512 | not retained because it reduced validation AUC |
| Exp 35 | `172aec3` | 0.7921 | discard | remove cyclic departure time features | not retained because it reduced validation AUC |
| Exp 36 | `3c09a6b` | 0.7938 | keep | lower reg_lambda to 0.5 | advanced best from 0.7935 to 0.7938 |
| Exp 37 | `9f9a739` | 0.7939 | keep | lower reg_lambda to 0.25 | advanced best from 0.7938 to 0.7939 |
| Exp 38 | `5274663` | 0.7945 | keep | remove reg_lambda | advanced best from 0.7939 to 0.7945 |
| Exp 39 | `35a3c3c` | 0.7947 | keep | raise max_depth to 22 with no l2 | advanced best from 0.7945 to 0.7947 |
| Exp 40 | `c231b8f` | 0.7950 | keep | raise max_depth to 24 with no l2 | advanced best from 0.7947 to 0.7950 |
| Exp 41 | `ce57e9a` | 0.7945 | discard | max_bin 512 | not retained because it reduced validation AUC |
| Exp 42 | `00ba05f` | 0.7945 | discard | max_bin 128 | not retained because it reduced validation AUC |
| Exp 43 | `88d1710` | 0.7946 | discard | raise max_depth to 26 | not retained because it reduced validation AUC |
| Exp 44 | `d81c3a3` | 0.7935 | discard | add log and sqrt distance transforms | not retained because it reduced validation AUC |
| Exp 45 | `ef147ce` | 0.7937 | discard | add 3-hour departure period categorical | not retained because it reduced validation AUC |
| Exp 46 | `7b45d65` | 0.7957 | keep | 400 trees at 0.015 learning rate | advanced best from 0.7950 to 0.7957 |
| Exp 47 | `df5ce5b` | 0.7958 | keep | 500 trees at 0.012 learning rate | advanced best from 0.7957 to 0.7958 |
| Exp 48 | `2b7c718` | 0.7969 | keep | 600 trees at 0.01 learning rate | advanced best from 0.7958 to 0.7969 |
| Exp 49 | `dbb8154` | 0.7970 | keep | 700 trees at 0.0085 learning rate | advanced best from 0.7969 to 0.7970 |
| Exp 50 | `0efcc89` | 0.7974 | keep | 700 trees at 0.009 learning rate | advanced best from 0.7970 to 0.7974 |
| Exp 51 | `f67e930` | 0.7986 | keep | 700 trees at 0.0095 learning rate | advanced best from 0.7974 to 0.7986 |
| Exp 52 | `eb524e5` | 0.7992 | keep | 700 trees at 0.01 learning rate | advanced best from 0.7986 to 0.7992 |
| Exp 53 | `47dcfbe` | 0.8003 | keep | 700 trees at 0.011 learning rate | advanced best from 0.7992 to 0.8003 |
| Exp 54 | `6ef12b1` | 0.8012 | keep | 700 trees at 0.012 learning rate | advanced best from 0.8003 to 0.8012 |
| Exp 55 | `71510ad` | 0.8024 | keep | 700 trees at 0.013 learning rate | advanced best from 0.8012 to 0.8024 |
| Exp 56 | `d423a3e` | 0.8029 | keep | 700 trees at 0.014 learning rate | advanced best from 0.8024 to 0.8029 |
| Exp 57 | `0d7e7d8` | 0.8037 | keep | 700 trees at 0.015 learning rate | advanced best from 0.8029 to 0.8037 |
| Exp 58 | `6fbd86b` | 0.8042 | keep | 700 trees at 0.016 learning rate | advanced best from 0.8037 to 0.8042 |
| Exp 59 | `31eed32` | 0.8046 | keep | 700 trees at 0.017 learning rate | advanced best from 0.8042 to 0.8046 |
| Exp 60 | `293d2eb` | 0.8050 | keep | 700 trees at 0.018 learning rate | advanced best from 0.8046 to 0.8050 |
| Exp 61 | `89e4b20` | 0.8053 | keep | 700 trees at 0.019 learning rate | advanced best from 0.8050 to 0.8053 |
| Exp 62 | `77fe758` | 0.8057 | keep | 700 trees at 0.02 learning rate | advanced best from 0.8053 to 0.8057 |
| Exp 63 | `18875f1` | 0.8061 | keep | 700 trees at 0.022 learning rate | advanced best from 0.8057 to 0.8061 |
| Exp 64 | `ca7ade5` | 0.8065 | keep | 700 trees at 0.024 learning rate | advanced best from 0.8061 to 0.8065 |
| Exp 65 | `0f89874` | 0.8069 | keep | 700 trees at 0.026 learning rate | advanced best from 0.8065 to 0.8069 |
| Exp 66 | `e4e996f` | 0.8069 | discard | 700 trees at 0.028 learning rate tied best | not retained because it only tied the current best at printed precision |
| Exp 67 | `5d04a51` | 0.8068 | discard | 700 trees at 0.027 learning rate | not retained because it reduced validation AUC |
| Exp 68 | `7a30d08` | 0.8077 | keep | lower min_child_weight to 8 at eta peak | advanced best from 0.8069 to 0.8077 |
| Exp 69 | `b23af85` | 0.8088 | keep | lower min_child_weight to 6 | advanced best from 0.8077 to 0.8088 |
| Exp 70 | `6c8f200` | 0.0000 | crash | min_child_weight 4 timed out | crashed or timed out before a valid metric; idea was not retained |
| Exp 71 | `36fbc5f` | 0.8084 | discard | 650 trees at 0.028 learning rate | not retained because it reduced validation AUC |
| Exp 72 | `b4dc670` | 0.8079 | discard | 650 trees at 0.03 learning rate | not retained because it reduced validation AUC |
| Exp 73 | `90817d6` | 0.8084 | discard | reduce max_depth to 22 at best settings | not retained because it reduced validation AUC |
| Exp 74 | `9825fb5` | 0.8088 | discard | colsample_bytree 0.55 tied best | not retained because it only tied the current best at printed precision |
| Exp 75 | `60424a9` | 0.8082 | discard | remove row subsampling | not retained because it reduced validation AUC |
| Exp 76 | `8634ad3` | 0.8082 | discard | lower subsample to 0.9 | not retained because it reduced validation AUC |
| Exp 77 | `9661382` | 0.8201 | keep | max_cat_to_onehot 32 | advanced best from 0.8088 to 0.8201 |
| Exp 78 | `633d32a` | 0.8112 | discard | max_cat_to_onehot 24 | not retained because it reduced validation AUC |
| Exp 79 | `86bf145` | 0.7996 | discard | max_cat_to_onehot 512 | not retained because it reduced validation AUC |
| Exp 80 | `0cde539` | 0.8205 | keep | max_cat_threshold 128 | advanced best from 0.8201 to 0.8205 |
| Exp 81 | `4f9c89e` | 0.8206 | discard | max_cat_threshold 256 too slow | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 82 | `5cdb54d` | 0.8210 | keep | max_cat_threshold 192 | advanced best from 0.8205 to 0.8210 |
| Exp 83 | `65bcc64` | 0.8206 | discard | max_cat_threshold 224 | not retained because it reduced validation AUC |
| Exp 84 | `4b59664` | 0.8205 | discard | max_cat_threshold 208 | not retained because it reduced validation AUC |
| Exp 85 | `4db273f` | 0.8206 | discard | max_cat_threshold 176 | not retained because it reduced validation AUC |
| Exp 86 | `0e2522e` | 0.8208 | discard | learning_rate 0.028 after categorical retune | not retained because it reduced validation AUC |
| Exp 87 | `c1043d5` | 0.8210 | discard | learning_rate 0.025 tied best | not retained because it only tied the current best at printed precision |
| Exp 88 | `9a47a2e` | 0.8217 | discard | min_child_weight 5 too slow | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 89 | `4348ec4` | 0.8213 | keep | min_child_weight 5 with 650 trees at 0.028 | advanced best from 0.8210 to 0.8213 |
| Exp 90 | `288a65a` | 0.8220 | keep | min_child_weight 5 with 650 trees at 0.029 | advanced best from 0.8213 to 0.8220 |
| Exp 91 | `64e6baf` | 0.8212 | discard | min_child_weight 5 with 650 trees at 0.03 | not retained because it reduced validation AUC |
| Exp 92 | `0eb7d08` | 0.8214 | discard | min_child_weight 5 with 650 trees at 0.0295 | not retained because it reduced validation AUC |
| Exp 93 | `6844416` | 0.8214 | discard | max_cat_threshold 224 after schedule retune | not retained because it reduced validation AUC |
| Exp 94 | `8d046f2` | 0.8224 | discard | colsample_bytree 0.65 too slow | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 95 | `caee7f9` | 0.8224 | discard | colsample_bytree 0.625 too slow | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 96 | `f657d61` | 0.8220 | discard | 625 trees 0.03 eta colsample 0.625 tied best | not retained because it only tied the current best at printed precision |
| Exp 97 | `7f25d47` | 0.8222 | keep | 625 trees 0.031 eta colsample 0.625 | advanced best from 0.8220 to 0.8222 |
| Exp 98 | `aa69176` | 0.8216 | discard | 625 trees 0.032 eta colsample 0.625 | not retained because it reduced validation AUC |
| Exp 99 | `d50ebcf` | 0.8221 | discard | 625 trees 0.0315 eta colsample 0.625 | not retained because it reduced validation AUC |
| Exp 100 | `ee795c9` | 0.8219 | discard | 600 trees 0.033 eta colsample 0.65 | not retained because it reduced validation AUC |
| Exp 101 | `9f7fc42` | 0.8221 | discard | add reg_alpha 0.1 | not retained because it reduced validation AUC |
| Exp 102 | `e6d7405` | 0.8233 | discard | min_child_weight 4 with 600 trees too slow | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 103 | `5e7303b` | 0.8233 | keep | min_child_weight 4 with 575 trees at 0.0345 | advanced best from 0.8222 to 0.8233 |
| Exp 104 | `8099ed5` | 0.8226 | discard | min_child_weight 4 with 575 trees at 0.0355 | not retained because it reduced validation AUC |
| Exp 105 | `782cb30` | 0.8230 | discard | min_child_weight 4 with 575 trees at 0.034 | not retained because it reduced validation AUC |
| Exp 106 | `680e55a` | 0.8227 | discard | min_child_weight 4 with 550 trees at 0.036 | not retained because it reduced validation AUC |
| Exp 107 | `731dacd` | 0.8230 | discard | max_cat_threshold 224 with child_weight 4 | not retained because it reduced validation AUC |
| Exp 108 | `7c2f90a` | 0.8224 | discard | subsample 0.9 with child_weight 4 | not retained because it reduced validation AUC |
| Exp 109 | `cf33d9a` | 0.8233 | discard | colsample_bytree 0.65 tied best | not retained because it only tied the current best at printed precision |
| Exp 110 | `c1ce7cb` | 0.8233 | discard | max_depth 26 tied best too slow | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 111 | `4e88375` | 0.8238 | keep | add categorical departure hour | advanced best from 0.8233 to 0.8238 |
| Exp 112 | `0c48fdb` | 0.8236 | discard | add categorical departure quarter | not retained because it reduced validation AUC |
| Exp 113 | `f0f408c` | 0.8238 | discard | remove raw departure hour tied best | not retained because it only tied the current best at printed precision |
| Exp 114 | `11631fa` | 0.8229 | discard | remove raw departure time | not retained because it reduced validation AUC |
| Exp 115 | `0f11483` | 0.8234 | discard | gamma 0.05 split-loss regularization | not retained because it reduced validation AUC |
| Exp 116 | `ad96f34` | 0.8240 | keep | learning_rate 0.0348 with DepHourCat | advanced best from 0.8238 to 0.8240 |
| Exp 117 | `8f82f0f` | 0.8237 | discard | 585 trees at learning_rate 0.0342 | not retained because it reduced validation AUC |
| Exp 118 | `08a837d` | 0.8233 | discard | learning_rate 0.0351 | not retained because it reduced validation AUC |
| Exp 119 | `a04c88b` | 0.8234 | discard | learning_rate 0.0347 | not retained because it reduced validation AUC |
| Exp 120 | `06f30d9` | 0.8234 | discard | colsample_bylevel 0.9 | not retained because it reduced validation AUC |
| Exp 121 | `3712049` | 0.8158 | discard | lossguide growth with max_leaves 512 | not retained because it reduced validation AUC |
| Exp 122 | `b40ee6d` | 0.8231 | discard | max_depth 23 | not retained because it reduced validation AUC |
| Exp 123 | `5ae4698` | 0.8236 | discard | distance quantile bins | not retained because it reduced validation AUC |
| Exp 124 | `3edc009` | 0.8047 | discard | carrier weekday categorical interaction | not retained because it reduced validation AUC |
| Exp 125 | `aa6a093` | 0.8235 | discard | 570 trees at learning_rate 0.035 | not retained because it reduced validation AUC |
| Exp 126 | `422eacb` | 0.8241 | discard | subsample 0.975 over runtime cap | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 127 | `63834f3` | 0.8238 | discard | subsample 0.97 over runtime cap | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 128 | `5eef80d` | 0.8238 | discard | subsample 0.965 over runtime cap | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 129 | `2137b2e` | 0.8240 | discard | colsample_bytree 0.6 over runtime cap | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |
| Exp 130 | `e5726e3` | 0.8241 | discard | 570 trees with subsample 0.975 over runtime cap | not retained because runtime violated the practical cap or the printed AUC did not strictly improve |

## Full Per-Iteration Notes

The sections below preserve the detailed experiment-by-experiment research notes, including hypotheses, searched references, changes, results, and decisions. They are retained here so this file is directly traceable to `results.tsv` and the corresponding git commits.

# Research Log

## Baseline - 7d0b368

- Result: CV AUC 0.7445 +/- 0.0043.
- Notes: Starter model uses raw `DepTime` and `Distance`, native categorical handling for Month/Day/Carrier/Origin/Dest, and a small 30-tree XGBoost classifier.

## Initial research

- XGBoost parameter docs: `learning_rate`/`eta` shrinks tree contributions and makes boosting more conservative; `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, and `reg_alpha` are documented complexity/regularization controls. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost categorical example: sklearn interface accepts pandas categorical columns with `enable_categorical=True`; `tree_method="hist"` is the documented path for categorical support examples. Source: https://xgboost.readthedocs.io/en/stable/python/examples/categorical.html
- XGBoost sklearn estimator docs: early stopping requires explicit validation data and can be awkward inside CV because folds can produce different tree counts; better to tune hyperparameters and retrain. Source: https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html

## Experiment 1 - 826568d

- Classification: follow-up to baseline.
- Hypothesis: 30 trees likely underfit; a smaller learning rate with 120 trees should improve ranking without making the model too complex.
- Change: `n_estimators=120`, `learning_rate=0.05`, explicit `tree_method="hist"`, `eval_metric="auc"`.
- Result: CV AUC 0.7513 +/- 0.0048.
- Decision: keep. This is a clear +0.0068 improvement with small complexity cost.

## Experiment 2 - f76c3b0

- Classification: follow-up to experiment 1.
- Hypothesis: If the model is still underfitting, a slower boosting schedule should add a small ranking gain.
- Change: `n_estimators=200`, `learning_rate=0.03`.
- Result: CV AUC 0.7517 +/- 0.0050.
- Decision: keep. Improvement is modest but the change is simple and runtime remains low.

## Experiment 3 - 8fc0379

- Classification: ablation/simplification of experiment 2.
- Hypothesis: Reducing `max_depth` from 6 to 4 might improve generalization if deeper interaction splits are overfitting categorical combinations.
- Change: `max_depth=4`.
- Result: CV AUC 0.7462 +/- 0.0046.
- Decision: discard. Shallower trees underfit badly here.

## Experiment 4 - cc49859

- Classification: follow-up to experiment 3 and current best.
- Hypothesis: Since depth 4 underfit, richer categorical interactions may help; try deeper trees while keeping the slower boosting schedule.
- Change: `max_depth=8`.
- Result: CV AUC 0.7552 +/- 0.0047.
- Decision: keep. This is the strongest gain since the first schedule change, and runtime is still acceptable.

## Experiment 5 - 11f1073

- Classification: follow-up to experiment 4.
- Hypothesis: If depth 8 helped, depth 10 may capture additional route/time/carrier interactions before overfitting dominates.
- Change: `max_depth=10`.
- Result: CV AUC 0.7595 +/- 0.0049.
- Decision: keep. Strong +0.0043 over depth 8, still under the experiment timeout.

## Experiment 6 - 3c01149

- Classification: follow-up to experiment 5.
- Hypothesis: Depth trend may continue because route/carrier/time interactions are high-cardinality and not represented explicitly.
- Change: `max_depth=12`.
- Result: CV AUC 0.7628 +/- 0.0052.
- Decision: keep. Good gain, but runtime is rising and should be watched.

## Experiment 7 - 45c7db5

- Classification: follow-up to experiment 6.
- Hypothesis: One more depth increase may capture additional sparse interactions; timeout risk is acceptable if monitored.
- Change: `max_depth=14`.
- Result: CV AUC 0.7657 +/- 0.0053.
- Decision: keep. Improvement remains meaningful, CV time 32.7s is still under the one-minute limit.

## Experiment 8 - 77ade59

- Classification: follow-up/regularization of experiment 7.
- Hypothesis: Deep trees are useful, but a higher minimum child weight can remove noisy tiny leaves and improve generalization.
- Change: `min_child_weight=5`.
- Result: CV AUC 0.7731 +/- 0.0050.
- Decision: keep. Large gain and runtime improved substantially.

## Experiment 9 - 6c34e12

- Classification: follow-up to experiment 8.
- Hypothesis: More conservative leaf growth may still help by pruning sparse categorical interactions.
- Change: `min_child_weight=10`.
- Result: CV AUC 0.7747 +/- 0.0053.
- Decision: keep. Improvement continues, though smaller than the jump from 1 to 5.

## Experiment 10 - 4aa4e92

- Classification: follow-up to experiment 9.
- Hypothesis: If `min_child_weight=10` helps, `20` might further reduce noisy sparse leaves.
- Change: `min_child_weight=20`.
- Result: CV AUC 0.7727 +/- 0.0062.
- Decision: discard. Too conservative; current best remains `min_child_weight=10`.

## Research after 10 experiments

- XGBoost categorical tutorial: categorical splits can use one-hot encoding or optimal partitioning; `max_cat_to_onehot` controls which path is used for lower-cardinality features. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost parameter docs: `max_cat_threshold` limits categories considered for partition-based splits and is documented as preventing overfitting. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- Chen and Guestrin's XGBoost paper emphasizes regularized tree boosting, shrinkage, and column subsampling as core generalization tools. Source: https://arxiv.org/abs/1603.02754

## Experiment 11 - 7637437

- Classification: exploration of categorical handling.
- Hypothesis: One-hot splits for low-cardinality calendar fields may be cleaner than partitioning while preserving partitioning for airports.
- Change: `max_cat_to_onehot=16`.
- Result: CV AUC 0.7746 +/- 0.0050.
- Decision: discard. Essentially tied but slightly below current best; no complexity justified.

## Experiment 12 - 2ed0c2d

- Classification: exploration of categorical regularization.
- Hypothesis: Limiting category candidates per partition split may reduce overfitting on Origin/Dest.
- Change: `max_cat_threshold=32`.
- Result: CV AUC 0.7745 +/- 0.0052.
- Decision: discard. Slightly worse than current best.

## Feature engineering research

- scikit-learn time-feature example: cyclical sine/cosine transforms encode periodic time without a discontinuity between the first and last value. Source: https://scikit-learn.org/1.3/auto_examples/applications/plot_cyclical_feature_engineering.html
- Review of flight delay prediction: the problem is complex and commonly approached with data-driven/ML models using flight data attributes. Source: https://arxiv.org/abs/1703.06118
- Scientific Reports flight-delay study: delays are affected by propagation through carriers/airports and BTS-style flight records support broad delay modeling. Source: https://www.nature.com/articles/s41598-020-62871-6

## Experiment 13 - 0fac8f8

- Classification: exploration of feature engineering.
- Hypothesis: Raw HHMM departure time is awkward for trees around midnight; explicit hour/minute/minutes and cyclic features should expose time-of-day structure.
- Change: add `DepHour`, `DepMinute`, `DepMinutes`, `DepTimeSin`, `DepTimeCos` inside `prepare(df)`.
- Result: CV AUC 0.7804 +/- 0.0048.
- Decision: keep. Strong +0.0057 over current best.

## Experiment 14 - 841d3a5

- Classification: exploration of route interaction feature.
- Hypothesis: Origin and destination separately may miss route-specific delay behavior.
- Change: add categorical `Route = Origin + "_" + Dest`.
- Result: CV AUC 0.7793 +/- 0.0046.
- Decision: discard. Worse and much slower.

## Experiment 15 - 60714b8

- Classification: exploration of calendar feature engineering.
- Hypothesis: Ordered/cyclic calendar structure and weekend effects may add signal beyond categorical partitions.
- Change: add `IsWeekend`, month/day-of-month/day-of-week sine/cosine features.
- Result: CV AUC 0.7802 +/- 0.0050.
- Decision: discard. Slightly worse than time-feature best and adds complexity.

## Experiment 16 - 7ed1b2a

- Classification: follow-up to experiment 13 and earlier depth sweep.
- Hypothesis: After adding useful time features, the model may still benefit from deeper interactions.
- Change: `max_depth=16`.
- Result: CV AUC 0.7818 +/- 0.0050.
- Decision: keep. Clear +0.0014 improvement with acceptable runtime.

## Experiment 17 - 819cae3

- Classification: follow-up to experiment 16.
- Hypothesis: `min_child_weight=10` may allow still deeper trees to capture useful interactions without exploding tiny leaves.
- Change: `max_depth=18`.
- Result: CV AUC 0.7825 +/- 0.0050.
- Decision: keep. Smaller but positive gain; runtime remains acceptable.

## Experiment 18 - 97192af

- Classification: follow-up to experiment 17.
- Hypothesis: One more depth increase may still add useful interaction splits, but gains should be smaller.
- Change: `max_depth=20`.
- Result: CV AUC 0.7827 +/- 0.0041.
- Decision: keep. Tiny but positive; stop direct depth sweep here.

## Experiment 19 - ce852e8

- Classification: follow-up/regularization of experiment 18.
- Hypothesis: Depth 20 might benefit from a slightly larger `min_child_weight`.
- Change: `min_child_weight=15`.
- Result: CV AUC 0.7807 +/- 0.0049.
- Decision: discard. Over-regularized; keep `min_child_weight=10`.

## Experiment 20 - 8f4d69d

- Classification: follow-up to current best boosting schedule.
- Hypothesis: Keeping roughly the same shrinkage budget with more smaller steps may improve ranking.
- Change: `n_estimators=300`, `learning_rate=0.02`.
- Result: CV AUC 0.7831 +/- 0.0044.
- Decision: keep. Small but positive improvement, runtime 29.6s remains under timeout.

## Research after 20 experiments

- XGBoost parameter docs: `subsample` samples rows per boosting iteration and can help prevent overfitting; uniform sampling typically uses values at least 0.5. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost parameter docs: `colsample_bytree`, `colsample_bylevel`, and `colsample_bynode` subsample columns at different points and compound with each other. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost parameter docs: `gamma`/`min_split_loss`, `reg_lambda`, and `reg_alpha` make split creation or leaf weights more conservative. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- Kapoor and Perrone discuss XGBoost tuning as expensive and use subsampling as a practical tuning accelerator; the direct training-data subsampling idea is not allowed here, but it reinforces evaluating subsampling-related knobs deliberately. Source: https://arxiv.org/abs/2111.06924

## Experiment 21 - 9869075

- Classification: follow-up/regularization of current best.
- Hypothesis: Slight row subsampling can reduce variance in the deep ensemble without losing much signal.
- Change: `subsample=0.9`.
- Result: CV AUC 0.7838 +/- 0.0048.
- Decision: keep. Useful +0.0007 gain and runtime stayed under limit.

## Experiment 22 - e9844a0

- Classification: follow-up to experiment 21.
- Hypothesis: Stronger row subsampling may regularize the high-capacity model further.
- Change: `subsample=0.8`.
- Result: CV AUC 0.7826 +/- 0.0050.
- Decision: discard. Too much row sampling; keep `subsample=0.9`.

## Experiment 23 - b768081

- Classification: follow-up to experiment 21.
- Hypothesis: A lighter row-sampling value may retain more signal while preserving some variance reduction.
- Change: `subsample=0.95`.
- Result: CV AUC 0.7840 +/- 0.0045.
- Decision: keep. Small positive gain over `subsample=0.9`.

## Experiment 24 - 798390e

- Classification: follow-up/regularization with column sampling.
- Hypothesis: Light feature sampling can reduce repeated correlated splits in deep trees without hiding too much signal.
- Change: `colsample_bytree=0.9`.
- Result: CV AUC 0.7870 +/- 0.0049.
- Decision: keep. Strong regularization gain.

## Experiment 25 - 847e7a3

- Classification: follow-up to experiment 24.
- Hypothesis: Stronger feature sampling may improve generalization further.
- Change: `colsample_bytree=0.8`.
- Result: CV AUC 0.7897 +/- 0.0049.
- Decision: keep. Large gain; continue bracketing column sampling.

## Experiment 26 - 89c886f

- Classification: follow-up to experiment 25.
- Hypothesis: Deeper trees may still see enough signal with 70% feature sampling while gaining more randomness.
- Change: `colsample_bytree=0.7`.
- Result: CV AUC 0.7907 +/- 0.0051.
- Decision: keep. Continued improvement.

## Experiment 27 - a15efee

- Classification: follow-up to experiment 26.
- Hypothesis: Stronger feature sampling may still work because multiple features are redundant views of time/calendar/categorical effects.
- Change: `colsample_bytree=0.6`.
- Result: CV AUC 0.7935 +/- 0.0047.
- Decision: keep. Strong improvement and faster runtime.

## Experiment 28 - 3642af5

- Classification: follow-up to experiment 27.
- Hypothesis: Very strong feature sampling may continue the regularization gain.
- Change: `colsample_bytree=0.5`.
- Result: CV AUC 0.7926 +/- 0.0049.
- Decision: discard. Too aggressive; current best remains `0.6`.

## Experiment 29 - 31a6b74

- Classification: follow-up to experiment 28.
- Hypothesis: A value between 0.5 and 0.6 might recover the benefit while adding a touch more regularization.
- Change: `colsample_bytree=0.55`.
- Result: CV AUC 0.7935 +/- 0.0047.
- Decision: discard. Tied current best at printed precision but did not improve.

## Experiment 30 - 4766104

- Classification: follow-up to experiment 29.
- Hypothesis: A slightly less aggressive column sample than 0.6 may preserve more key predictors while retaining randomness.
- Change: `colsample_bytree=0.65`.
- Result: CV AUC 0.7919 +/- 0.0051.
- Decision: discard. Worse; keep `colsample_bytree=0.6`.

## Research after 30 experiments

- XGBoost parameter docs: tree, level, and node column sampling compound, so adding `colsample_bylevel` or `colsample_bynode` on top of `colsample_bytree=0.6` should be mild. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost parameter docs: `gamma` is the minimum loss reduction required for a split; higher values make the tree more conservative. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost parameter docs: `reg_lambda` and `reg_alpha` are L2 and L1 leaf-weight penalties. Source: https://xgboost.readthedocs.io/en/stable/parameter.html

## Experiment 31 - cb64df2

- Classification: exploration of node-level column sampling.
- Hypothesis: Mild per-node feature randomization may improve generalization on deep trees.
- Change: `colsample_bynode=0.9`.
- Result: CV AUC 0.7931 +/- 0.0038.
- Decision: discard. Worse than tree-level sampling alone.

## Experiment 32 - badf2cc

- Classification: exploration of split-loss regularization.
- Hypothesis: A small `gamma` may remove marginal deep splits.
- Change: `gamma=0.1`.
- Result: CV AUC 0.7935 +/- 0.0048.
- Decision: discard. Tied best at printed precision but did not improve.

## Experiment 33 - 81514b9

- Classification: exploration of L2 regularization.
- Hypothesis: Stronger L2 leaf-weight regularization may stabilize the deep sampled ensemble.
- Change: `reg_lambda=2.0`.
- Result: CV AUC 0.7920 +/- 0.0055.
- Decision: discard. Too much regularization.

## Plateau research after experiment 33

- XGBoost parameter docs: `grow_policy` supports `depthwise` and `lossguide`; `lossguide` splits nodes with highest loss change and `max_leaves` caps leaves. Source: https://xgboost.readthedocs.io/en/stable/parameter.html

## Experiment 34 - 9f82898

- Classification: exploration of tree growth policy.
- Hypothesis: Leaf-wise growth may spend capacity more efficiently on sparse categorical interactions.
- Change: `grow_policy="lossguide"`, `max_leaves=512`.
- Result: CV AUC 0.7893 +/- 0.0048.
- Decision: discard. Worse and slower; depthwise growth remains better.

## Experiment 35 - 172aec3

- Classification: ablation/simplification of time features.
- Hypothesis: Direct hour/minute/minutes features may carry the time signal without sine/cosine redundancy.
- Change: remove `DepTimeSin` and `DepTimeCos`.
- Result: CV AUC 0.7921 +/- 0.0045.
- Decision: discard. Cyclic features are useful.

## Experiment 36 - 3c09a6b

- Classification: follow-up to L2 regularization result.
- Hypothesis: Since higher L2 hurt, lower L2 may let the sampled ensemble use stronger leaf weights.
- Change: `reg_lambda=0.5`.
- Result: CV AUC 0.7938 +/- 0.0050.
- Decision: keep. Small but positive improvement.

## Experiment 37 - 9f9a739

- Classification: follow-up to experiment 36.
- Hypothesis: The sampled model may still be over-regularized by L2, so a lower penalty can help.
- Change: `reg_lambda=0.25`.
- Result: CV AUC 0.7939 +/- 0.0045.
- Decision: keep. Very small but positive gain.

## Experiment 38 - 5274663

- Classification: follow-up to experiment 37.
- Hypothesis: With row/column sampling and child-weight regularization, explicit L2 may be unnecessary.
- Change: `reg_lambda=0.0`.
- Result: CV AUC 0.7945 +/- 0.0041.
- Decision: keep. Removing L2 is a simple improvement.

## Experiment 39 - 35a3c3c

- Classification: follow-up to experiment 38.
- Hypothesis: Without L2, a little more depth may recover useful sparse interactions.
- Change: `max_depth=22`.
- Result: CV AUC 0.7947 +/- 0.0044.
- Decision: keep. Small positive gain.

## Experiment 40 - c231b8f

- Classification: follow-up to experiment 39.
- Hypothesis: Depth gains may continue slightly under no-L2 plus strong column sampling.
- Change: `max_depth=24`.
- Result: CV AUC 0.7950 +/- 0.0051.
- Decision: keep. Small positive gain; depth still not hurting runtime.

## Research after 40 experiments

- XGBoost parameter docs: `max_bin` controls the maximum discrete bins for `hist`; larger values can improve split precision at a computational cost. Source: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost interaction constraints tutorial: constraints can restrict feature interactions by feature names in Python, but this is a structural modeling choice and should be tried only with a clear domain grouping. Source: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
- XGBoost parameter docs: `scale_pos_weight` is for unbalanced positive/negative classes; this training slice is balanced by construction, so changing it is not a good next lever. Source: https://xgboost.readthedocs.io/en/stable/parameter.html

## Experiment 41 - ce57e9a

- Classification: exploration of histogram split precision.
- Hypothesis: More bins may improve continuous feature splits.
- Change: `max_bin=512`.
- Result: CV AUC 0.7945 +/- 0.0052.
- Decision: discard. Worse and slower.

## Experiment 42 - 00ba05f

- Classification: exploration of histogram regularization.
- Hypothesis: Coarser histograms may regularize noisy splits and speed training.
- Change: `max_bin=128`.
- Result: CV AUC 0.7945 +/- 0.0047.
- Decision: discard. Default binning remains better.

## Experiment 43 - 88d1710

- Classification: follow-up to depth sweep.
- Hypothesis: Depth 26 may continue the small no-L2 depth gains.
- Change: `max_depth=26`.
- Result: CV AUC 0.7946 +/- 0.0043.
- Decision: discard. Depth 24 is the current ceiling.

## Plateau research after experiment 43

- Flight-delay XGBoost study: departure time and carrier are important flight-factor signals; weather is not available in this dataset. Source: https://www.sciencedirect.com/science/article/pii/S2772415822000050
- Flight-delay feature-engineering article: timing, airport/location, aircraft/carrier, and distance-style features are common feature categories. Source: https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches
- Hybrid flight-delay paper feature table includes scheduled departure time, elapsed time, and distance as continuous flight predictors. Source: https://engj.org/index.php/ej/article/download/4376/1156

## Experiment 44 - d81c3a3

- Classification: exploration of distance transformations.
- Hypothesis: Log/sqrt distance may expose short/medium/long-haul structure more smoothly than raw miles.
- Change: add `LogDistance` and `SqrtDistance`.
- Result: CV AUC 0.7935 +/- 0.0050.
- Decision: discard. Raw distance is enough; transforms add noise.

## Experiment 45 - ef147ce

- Classification: exploration of time bucket feature.
- Hypothesis: A low-cardinality time-of-day categorical may capture departure regimes beyond numeric/cyclic time.
- Change: add 3-hour `DepPeriod` categorical.
- Result: CV AUC 0.7937 +/- 0.0049.
- Decision: discard. Time buckets add noise.

## Experiment 46 - 7b45d65

- Classification: follow-up to boosting schedule.
- Hypothesis: More smaller boosting steps may improve ranking with the current regularized deep model.
- Change: `n_estimators=400`, `learning_rate=0.015`.
- Result: CV AUC 0.7957 +/- 0.0043.
- Decision: keep. Good +0.0007 gain, runtime still under timeout.

## Experiment 47 - df5ce5b

- Classification: follow-up to experiment 46.
- Hypothesis: Finer boosting steps with the same rough total budget may eke out a small ranking gain.
- Change: `n_estimators=500`, `learning_rate=0.012`.
- Result: CV AUC 0.7958 +/- 0.0049.
- Decision: keep. Tiny positive gain; runtime is getting high.

## Experiment 48 - 2b7c718

- Classification: follow-up to experiment 47.
- Hypothesis: The slower schedule may continue helping if runtime stays below the cap.
- Change: `n_estimators=600`, `learning_rate=0.01`.
- Result: CV AUC 0.7969 +/- 0.0046.
- Decision: keep. Strong gain, but runtime 45.8s is near the practical limit.

## Experiment 49 - dbb8154

- Classification: follow-up to experiment 48.
- Hypothesis: A 700-tree schedule with a matching smaller learning rate may slightly improve ranking while staying under timeout.
- Change: `n_estimators=700`, `learning_rate=0.0085`.
- Result: CV AUC 0.7970 +/- 0.0048.
- Decision: keep. Tiny gain; CV time 56.6s is near the hard limit, so do not increase tree count further.

## Experiment 50 - 0efcc89

- Classification: follow-up to experiment 49.
- Hypothesis: Keeping 700 trees but slightly increasing learning rate may improve ranking at the same runtime.
- Change: `learning_rate=0.009`.
- Result: CV AUC 0.7974 +/- 0.0046.
- Decision: keep. New best; runtime 55.0s leaves little room for more trees.

## Research after 50 experiments

- XGBoost sklearn estimator docs: early stopping requires validation data through `eval_set`; this is not directly compatible with the current `cross_val_score` one-liner without changing the CV loop. Source: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
- XGBoost prediction docs: sklearn predictions automatically use `best_iteration` when early stopping is used, but final-model handling still needs care. Source: https://xgboost.readthedocs.io/en/stable/prediction.html
- XGBoost parameter docs: further tree-count increases are risky because current CV runtime is already near the one-minute experiment limit; tune same-runtime parameters first. Source: https://xgboost.readthedocs.io/en/stable/parameter.html

## Experiment 51 - f67e930

- Classification: follow-up to experiment 50.
- Hypothesis: Since 0.009 beat 0.0085, a slightly larger learning rate may continue improving at the same runtime.
- Change: `learning_rate=0.0095`.
- Result: CV AUC 0.7986 +/- 0.0046.
- Decision: keep. Strong same-runtime gain.

## Experiment 52 - eb524e5

- Classification: follow-up to experiment 51.
- Hypothesis: Same 700-tree runtime with a larger learning-rate budget may keep improving.
- Change: `learning_rate=0.01`.
- Result: CV AUC 0.7992 +/- 0.0046.
- Decision: keep. New best.

## Experiment 53 - 47dcfbe

- Classification: follow-up to experiment 52.
- Hypothesis: The 700-tree model still benefits from a larger learning-rate budget.
- Change: `learning_rate=0.011`.
- Result: CV AUC 0.8003 +/- 0.0044.
- Decision: keep. Strong same-runtime improvement.

## Experiment 54 - 6ef12b1

- Classification: follow-up to experiment 53.
- Hypothesis: The learning-rate curve has not peaked yet.
- Change: `learning_rate=0.012`.
- Result: CV AUC 0.8012 +/- 0.0046.
- Decision: keep. Strong improvement at same runtime.

## Experiment 55 - 71510ad

- Classification: follow-up to experiment 54.
- Hypothesis: A larger same-runtime learning-rate budget may continue improving before overfit appears.
- Change: `learning_rate=0.013`.
- Result: CV AUC 0.8024 +/- 0.0043.
- Decision: keep. Strong gain.

## Experiment 56 - d423a3e

- Classification: follow-up to experiment 55.
- Hypothesis: Same-runtime learning-rate increases may still help.
- Change: `learning_rate=0.014`.
- Result: CV AUC 0.8029 +/- 0.0046.
- Decision: keep. Improved, but CV time 57.1s is close to the limit.

## Experiment 57 - 0d7e7d8

- Classification: follow-up to experiment 56.
- Hypothesis: The learning-rate peak is still higher; same runtime, higher step budget.
- Change: `learning_rate=0.015`.
- Result: CV AUC 0.8037 +/- 0.0041.
- Decision: keep. Strong gain, runtime still under cap.

## Experiment 58 - 6fbd86b

- Classification: follow-up to experiment 57.
- Hypothesis: Same-runtime learning-rate increase may still improve before overfit.
- Change: `learning_rate=0.016`.
- Result: CV AUC 0.8042 +/- 0.0041.
- Decision: keep. Improvement continues.

## Experiment 59 - 31eed32

- Classification: follow-up to experiment 58.
- Hypothesis: The 700-tree learning-rate peak is still higher.
- Change: `learning_rate=0.017`.
- Result: CV AUC 0.8046 +/- 0.0041.
- Decision: keep. Improvement continues.

## Experiment 60 - 293d2eb

- Classification: follow-up to experiment 59.
- Hypothesis: Learning-rate peak may still be higher at fixed tree count.
- Change: `learning_rate=0.018`.
- Result: CV AUC 0.8050 +/- 0.0045.
- Decision: keep. Improvement continues with acceptable runtime.

## Research after 60 experiments

- XGBoost parameter docs: `learning_rate`/`eta` shrinks each boosting update to make boosting more conservative; the optimal number of rounds depends on the problem and is not automatic. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost R-package docs: number of boosting rounds varies by problem, and `eta` controls step-size shrinkage. Source: https://xgboost.readthedocs.io/en/latest/r_docs/R-package/docs/reference/xgboost.html

## Experiment 61 - 89e4b20

- Classification: follow-up to experiment 60.
- Hypothesis: The fixed-700-tree learning-rate curve is still increasing.
- Change: `learning_rate=0.019`.
- Result: CV AUC 0.8053 +/- 0.0042.
- Decision: keep. Improvement continues.

## Experiment 62 - 77fe758

- Classification: follow-up to experiment 61.
- Hypothesis: The learning-rate peak remains above 0.019.
- Change: `learning_rate=0.02`.
- Result: CV AUC 0.8057 +/- 0.0046.
- Decision: keep. Improvement continues.

## Experiment 63 - 18875f1

- Classification: follow-up to experiment 62.
- Hypothesis: The fixed-tree learning-rate curve is still rising.
- Change: `learning_rate=0.022`.
- Result: CV AUC 0.8061 +/- 0.0037.
- Decision: keep. Improvement continues.

## Experiment 64 - ca7ade5

- Classification: follow-up to experiment 63.
- Hypothesis: The learning-rate peak is close but not yet reached.
- Change: `learning_rate=0.024`.
- Result: CV AUC 0.8065 +/- 0.0043.
- Decision: keep. Improvement continues.

## Experiment 65 - 0f89874

- Classification: follow-up to experiment 64.
- Hypothesis: The same-runtime learning-rate curve may still be rising.
- Change: `learning_rate=0.026`.
- Result: CV AUC 0.8069 +/- 0.0040.
- Decision: keep. Improvement continues.

## Experiment 66 - e4e996f

- Classification: follow-up to experiment 65.
- Hypothesis: Learning rate 0.028 may improve beyond 0.026.
- Change: `learning_rate=0.028`.
- Result: CV AUC 0.8069 +/- 0.0041.
- Decision: discard. Tied best at printed precision but did not improve.

## Experiment 67 - 5d04a51

- Classification: follow-up to experiment 66.
- Hypothesis: The midpoint 0.027 may beat both 0.026 and 0.028.
- Change: `learning_rate=0.027`.
- Result: CV AUC 0.8068 +/- 0.0036.
- Decision: discard. Local best remains 0.026.

## Experiment 68 - 7a30d08

- Classification: follow-up/regularization retune after eta peak.
- Hypothesis: With the retuned learning rate, slightly lower child-weight regularization may recover useful splits.
- Change: `min_child_weight=8`.
- Result: CV AUC 0.8077 +/- 0.0040.
- Decision: keep. Strong gain.

## Experiment 69 - b23af85

- Classification: follow-up to experiment 68.
- Hypothesis: Current best may still be over-regularized by child weight.
- Change: `min_child_weight=6`.
- Result: CV AUC 0.8088 +/- 0.0043.
- Decision: keep. Strong gain, but CV time 59.0s is at the limit.

## Experiment 70 - 6c8f200

- Classification: follow-up to experiment 69.
- Hypothesis: Lower child weight may improve AUC further if runtime allows.
- Change: `min_child_weight=4`.
- Result: timeout before metric.
- Decision: crash/discard. Revert to `min_child_weight=6`.

## Research after 70 experiments

- XGBoost parameter docs: `hist` is the faster histogram-based approximate tree method, and it is already in use. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost parameter docs: deep trees can be expensive; `max_depth`, boosting rounds, and child-weight regularization are the practical runtime levers here. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- scikit-learn `cross_val_score` docs: CV supports `n_jobs`, but changing outer/inner parallelism is a runtime-only experiment and needs care to avoid oversubscription. Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

## Experiment 71 - 36fbc5f

- Classification: runtime-aware follow-up to experiment 69.
- Hypothesis: 650 trees with higher eta may preserve boosting budget while reducing runtime.
- Change: `n_estimators=650`, `learning_rate=0.028`.
- Result: CV AUC 0.8084 +/- 0.0034.
- Decision: discard. Close but worse than current best.

## Experiment 72 - b4dc670

- Classification: follow-up to experiment 71.
- Hypothesis: A larger eta may let the 650-tree model catch the 700-tree best.
- Change: `n_estimators=650`, `learning_rate=0.03`.
- Result: CV AUC 0.8079 +/- 0.0034.
- Decision: discard. Worse than 650 at 0.028 and worse than best.

## Experiment 73 - 90817d6

- Classification: runtime-aware simplification of current best.
- Hypothesis: Depth 22 may preserve AUC or reduce overfit/runtime after child-weight retune.
- Change: `max_depth=22`.
- Result: CV AUC 0.8084 +/- 0.0039.
- Decision: discard. Worse and not meaningfully faster.

## Experiment 74 - 9825fb5

- Classification: follow-up retune of column sampling.
- Hypothesis: Child-weight and eta changes may shift the column-sampling sweet spot lower.
- Change: `colsample_bytree=0.55`.
- Result: CV AUC 0.8088 +/- 0.0043.
- Decision: discard. Tied best at printed precision but did not improve.

## Experiment 75 - 60424a9

- Classification: follow-up retune of row sampling.
- Hypothesis: With lower child weight, row sampling may no longer be needed.
- Change: `subsample=1.0`.
- Result: CV AUC 0.8082 +/- 0.0041.
- Decision: discard. Worse and slower; keep row sampling.

## Experiment 76 - 8634ad3

- Classification: follow-up to experiment 75.
- Hypothesis: Stronger row sampling may regularize lower child-weight trees and improve runtime.
- Change: `subsample=0.9`.
- Result: CV AUC 0.8082 +/- 0.0037.
- Decision: discard. Worse; current best remains `subsample=0.95`.

## Plateau research after experiment 76

- XGBoost categorical tutorial: categorical splits can use one-hot or partitioning, controlled by `max_cat_to_onehot`; auto-recoding helps keep dataframe category handling consistent. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- scikit-learn `TargetEncoder` docs: target encoding is cross-fitted internally in `fit_transform`; using it correctly inside this repo would require restructuring the model pipeline and careful ground-truth consistency. Source: https://sklearn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- XGBoost interaction constraints docs: constraints can be set by feature names, but need strong domain priors; no obvious safe grouping emerged from the failed route/calendar experiments. Source: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

## Experiment 77 - 9661382

- Classification: follow-up to categorical handling research.
- Hypothesis: Under the stronger model settings, one-hot splits for low/medium-cardinality categorical fields may outperform partitioning.
- Change: `max_cat_to_onehot=32`.
- Result: CV AUC 0.8201 +/- 0.0039.
- Decision: keep. Major gain.

## Experiment 78 - 633d32a

- Classification: ablation of categorical one-hot threshold.
- Hypothesis: Threshold 24 may keep carrier/month/day-of-week one-hot while avoiding day-of-month.
- Change: `max_cat_to_onehot=24`.
- Result: CV AUC 0.8112 +/- 0.0037.
- Decision: discard. Day-of-month one-hot appears important.

## Experiment 79 - 86bf145

- Classification: exploration of high-cardinality categorical one-hot splits.
- Hypothesis: Airport one-hot-style categorical splits may outperform partitioning.
- Change: `max_cat_to_onehot=512`.
- Result: CV AUC 0.7996 +/- 0.0040.
- Decision: discard. Airport categoricals should stay partition-based.

## Experiment 80 - 0cde539

- Classification: follow-up to airport categorical partitioning.
- Hypothesis: With airports kept partition-based, a larger `max_cat_threshold` may improve airport splits by considering more categories.
- Change: `max_cat_threshold=128`.
- Result: CV AUC 0.8205 +/- 0.0038.
- Decision: keep. Small positive gain.

## Research after 80 experiments

- XGBoost categorical tutorial: partition-based categorical splits group categories with similar leaf values using sorted partitions, rather than enumerating all category permutations. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost parameter docs: `max_cat_threshold` is specifically for partition-based categorical splits and prevents overfitting by limiting categories considered per split. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- Category counts in this training slice: Month 12, DayofMonth 31, DayOfWeek 7, UniqueCarrier 20, Origin/Dest 282. This supports `max_cat_to_onehot=32` for calendar/carrier while leaving airports partitioned.

## Experiment 81 - 4f9c89e

- Classification: follow-up to experiment 80.
- Hypothesis: Raising `max_cat_threshold` to 256 may improve airport partitioning further.
- Change: `max_cat_threshold=256`.
- Result: CV AUC 0.8206 +/- 0.0033, but CV time 60.2s.
- Decision: discard. Slight AUC gain is over the runtime cap.

## Experiment 82 - 5cdb54d

- Classification: follow-up to experiment 81.
- Hypothesis: `max_cat_threshold=192` may recover more airport partitioning signal while staying under the cap.
- Change: `max_cat_threshold=192`.
- Result: CV AUC 0.8210 +/- 0.0036, CV time 58.8s.
- Decision: keep. Better than 128 and under the reported CV cap.

## Experiment 83 - 65bcc64

- Classification: follow-up to experiment 82.
- Hypothesis: A threshold of 224 may improve airport partitioning further.
- Change: `max_cat_threshold=224`.
- Result: CV AUC 0.8206 +/- 0.0031.
- Decision: discard. Worse than 192.

## Experiment 84 - 4b59664

- Classification: follow-up to experiment 83.
- Hypothesis: Threshold 208 may sit between the 192 keeper and 224 discard.
- Change: `max_cat_threshold=208`.
- Result: CV AUC 0.8205 +/- 0.0033.
- Decision: discard. Worse than 192.

## Experiment 85 - 4db273f

- Classification: follow-up to experiment 84.
- Hypothesis: Threshold 176 may beat 192 from below.
- Change: `max_cat_threshold=176`.
- Result: CV AUC 0.8206 +/- 0.0037.
- Decision: discard. Worse than 192.

## Experiment 86 - 0e2522e

- Classification: follow-up retune after categorical improvements.
- Hypothesis: The better categorical split settings may shift the optimal learning rate upward.
- Change: `learning_rate=0.028`.
- Result: CV AUC 0.8208 +/- 0.0031.
- Decision: discard. Worse than current best 0.026.

## Experiment 87 - c1043d5

- Classification: follow-up to experiment 86.
- Hypothesis: Learning rate 0.025 may beat 0.026 from below.
- Change: `learning_rate=0.025`.
- Result: CV AUC 0.8210 +/- 0.0032.
- Decision: discard. Tied current best at printed precision but did not improve.

## Experiment 88 - 9a47a2e

- Classification: follow-up to child-weight retune.
- Hypothesis: `min_child_weight=5` may improve AUC after categorical tuning without timing out.
- Change: `min_child_weight=5`.
- Result: CV AUC 0.8217 +/- 0.0031, CV time 63.7s.
- Decision: discard for runtime despite improved AUC.

## Experiment 89 - 4348ec4

- Classification: runtime-aware follow-up to experiment 88.
- Hypothesis: Fewer trees with a compensating eta can keep the child-weight 5 gain under the cap.
- Change: `n_estimators=650`, `learning_rate=0.028`, `min_child_weight=5`.
- Result: CV AUC 0.8213 +/- 0.0033, CV time 59.6s.
- Decision: keep. New valid best.

## Experiment 90 - 288a65a

- Classification: follow-up to experiment 89.
- Hypothesis: The 650-tree child-weight 5 schedule may still benefit from a slightly larger eta.
- Change: `learning_rate=0.029`.
- Result: CV AUC 0.8220 +/- 0.0034, CV time 58.8s.
- Decision: keep. New best.

## Research after 90 experiments

- XGBoost parameter docs confirm that `max_cat_to_onehot` uses one-hot splits only when the category count is lower than the threshold; `max_cat_threshold` controls partition-based categorical split search. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost categorical tutorial explains that partitioning sorts categories by leaf value and groups similar categories, matching why Origin/Dest should remain partition-based. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost tuning guidance from managed tuning docs lists eta, min_child_weight, subsample, and colsample parameters among high-impact knobs; these match the strongest empirical levers here. Source: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html

## Experiment 91 - 64e6baf

- Classification: follow-up to experiment 90.
- Hypothesis: Eta 0.030 may improve over 0.029 for the 650-tree child-weight 5 schedule.
- Change: `learning_rate=0.030`.
- Result: CV AUC 0.8212 +/- 0.0032.
- Decision: discard. Too high; current best remains 0.029.

## Experiment 92 - 0eb7d08

- Classification: follow-up to experiment 91.
- Hypothesis: Eta 0.0295 may improve between the 0.029 keeper and 0.030 discard.
- Change: `learning_rate=0.0295`.
- Result: CV AUC 0.8214 +/- 0.0034.
- Decision: discard. Worse than 0.029.

## Experiment 93 - 6844416

- Classification: categorical threshold retune after schedule change.
- Hypothesis: `max_cat_threshold=224` may work better under the 650-tree child-weight 5 schedule.
- Change: `max_cat_threshold=224`.
- Result: CV AUC 0.8214 +/- 0.0032.
- Decision: discard. Worse than 192.

## Experiment 94 - 8d046f2

- Classification: column-sampling retune after categorical changes.
- Hypothesis: The model may need more features per tree after one-hot categorical splits.
- Change: `colsample_bytree=0.65`.
- Result: CV AUC 0.8224 +/- 0.0032, CV time 61.6s.
- Decision: discard for runtime despite AUC gain.

## Experiment 95 - caee7f9

- Classification: runtime-aware follow-up to experiment 94.
- Hypothesis: `colsample_bytree=0.625` may recover the AUC gain while staying under runtime cap.
- Change: `colsample_bytree=0.625`.
- Result: CV AUC 0.8224 +/- 0.0032, CV time 61.5s.
- Decision: discard for runtime.

## Experiment 96 - f657d61

- Classification: runtime-aware combination after experiment 95.
- Hypothesis: Fewer trees can make `colsample_bytree=0.625` valid while preserving AUC.
- Change: `n_estimators=625`, `learning_rate=0.030`, `colsample_bytree=0.625`.
- Result: CV AUC 0.8220 +/- 0.0034, CV time 59.3s.
- Decision: discard. Tied best at printed precision but did not improve.

## Experiment 97 - 7f25d47

- Classification: follow-up to experiment 96.
- Hypothesis: The 625-tree colsample 0.625 model was slightly under-boosted.
- Change: `learning_rate=0.031`.
- Result: CV AUC 0.8222 +/- 0.0032, CV time 59.1s.
- Decision: keep. New valid best.

## Experiment 98 - aa69176

- Classification: follow-up to experiment 97.
- Hypothesis: Eta 0.032 may improve the 625-tree colsample 0.625 schedule further.
- Change: `learning_rate=0.032`.
- Result: CV AUC 0.8216 +/- 0.0036.
- Decision: discard. Too high.

## Experiment 99 - d50ebcf

- Classification: follow-up to experiment 98.
- Hypothesis: Eta 0.0315 may beat the 0.031 keeper.
- Change: `learning_rate=0.0315`.
- Result: CV AUC 0.8221 +/- 0.0032.
- Decision: discard. Worse than 0.031.

## Experiment 100 - ee795c9

- Classification: runtime-aware feature-sampling tradeoff.
- Hypothesis: 600 trees with higher eta and `colsample_bytree=0.65` may recover the higher-colsample AUC under the cap.
- Change: `n_estimators=600`, `learning_rate=0.033`, `colsample_bytree=0.65`.
- Result: CV AUC 0.8219 +/- 0.0037, CV time 56.2s.
- Decision: discard. Faster but worse than current best.

## Research after 100 experiments

- XGBoost docs list `reg_alpha` as L1 regularization on leaf weights; unlike tree count or categorical thresholds, it should not materially increase runtime. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost tuning docs continue to identify eta, min_child_weight, subsample, and column sampling as high-impact knobs, matching the strongest retained changes. Source: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html

## Experiment 101 - 9f7fc42

- Classification: exploration of L1 regularization.
- Hypothesis: Small L1 may trim weak leaf weights in the richer categorical model.
- Change: `reg_alpha=0.1`.
- Result: CV AUC 0.8221 +/- 0.0032.
- Decision: discard. Slightly worse than current best.

## Experiment 102 - e6d7405

- Classification: child-weight/runtime retune.
- Hypothesis: Child weight 4 can add signal if a smaller schedule controls runtime.
- Change: `n_estimators=600`, `learning_rate=0.033`, `min_child_weight=4`.
- Result: CV AUC 0.8233 +/- 0.0029, CV time 61.8s.
- Decision: discard for runtime despite AUC gain.

## Experiment 103 - 5e7303b

- Classification: runtime-aware follow-up to experiment 102.
- Hypothesis: 575 trees with larger eta may preserve child-weight 4 AUC under the cap.
- Change: `n_estimators=575`, `learning_rate=0.0345`, `min_child_weight=4`.
- Result: CV AUC 0.8233 +/- 0.0031, CV time 59.3s.
- Decision: keep. New valid best.

## Experiment 104 - 8099ed5

- Classification: follow-up to experiment 103.
- Hypothesis: The 575-tree child-weight 4 schedule may still be under-boosted.
- Change: `learning_rate=0.0355`.
- Result: CV AUC 0.8226 +/- 0.0033.
- Decision: discard. Too high.

## Experiment 105 - 782cb30

- Classification: follow-up to experiment 104.
- Hypothesis: Eta 0.034 may beat 0.0345 from below.
- Change: `learning_rate=0.034`.
- Result: CV AUC 0.8230 +/- 0.0031.
- Decision: discard. Worse than 0.0345.

## Experiment 106 - 680e55a

- Classification: runtime-aware schedule simplification.
- Hypothesis: 550 trees with compensating eta may preserve AUC and free runtime.
- Change: `n_estimators=550`, `learning_rate=0.036`.
- Result: CV AUC 0.8227 +/- 0.0033.
- Decision: discard. Faster but worse.

## Experiment 107 - 731dacd

- Classification: categorical threshold retune under child-weight 4.
- Hypothesis: Larger airport partition threshold may improve the current child-weight 4 model.
- Change: `max_cat_threshold=224`.
- Result: CV AUC 0.8230 +/- 0.0033, CV time 60.0s.
- Decision: discard. Worse than 192.

## Experiment 108 - 7c2f90a

- Classification: row-sampling retune under child-weight 4.
- Hypothesis: Stronger row sampling may regularize lower child weight and reduce runtime.
- Change: `subsample=0.9`.
- Result: CV AUC 0.8224 +/- 0.0030.
- Decision: discard. Worse than current best.

## Experiment 109 - cf33d9a

- Classification: column-sampling retune under child-weight 4.
- Hypothesis: Smaller tree count may make `colsample_bytree=0.65` valid and improve AUC.
- Change: `colsample_bytree=0.65`.
- Result: CV AUC 0.8233 +/- 0.0031.
- Decision: discard. Tied current best at printed precision but did not improve.

## Experiment 110 - c1ce7cb

- Classification: capacity retune under current best.
- Hypothesis: Extra depth may help after categorical and child-weight changes.
- Change: `max_depth=26`.
- Result: CV AUC 0.8233 +/- 0.0033, CV time 61.4s.
- Decision: discard. Tied best and too slow.

## Research after 110 experiments

- XGBoost tree-method docs: `hist` is the fastest tree method and `approx` can sometimes help accuracy but is slower; switching tree methods is unlikely to solve the current runtime constraint. Source: https://xgboost.readthedocs.io/en/stable/treemethod.html
- XGBoost parameter docs: `max_bin` can improve split optimality at higher compute cost; earlier direct max-bin tests did not help here. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost categorical docs: low-cardinality categoricals can use one-hot split behavior; a categorical departure-hour feature is a low-risk extension of the successful time features. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

## Experiment 111 - 4e88375

- Classification: feature-engineering follow-up to successful time features.
- Hypothesis: A one-hot categorical departure-hour feature can capture hour regimes beyond numeric/cyclic time.
- Change: add `DepHourCat`.
- Result: CV AUC 0.8238 +/- 0.0032, CV time 58.8s.
- Decision: keep. Clear gain.

## Experiment 112 - 0c48fdb

- Classification: feature-engineering follow-up to experiment 111.
- Hypothesis: A categorical departure-quarter feature can capture within-hour timing regimes beyond the numeric minute field.
- Change: add `DepQuarterCat`.
- Result: CV AUC 0.8236 +/- 0.0034, CV time 59.8s.
- Decision: discard. Worse than current best.

## Experiment 113 - f0f408c

- Classification: feature ablation after experiment 111.
- Hypothesis: With `DepHourCat` present, the raw integer `DepHour` may be redundant noise.
- Change: remove `DepHour`.
- Result: CV AUC 0.8238 +/- 0.0033, CV time 59.0s.
- Decision: discard. Tied current best at printed precision but did not strictly improve.

## Experiment 114 - 11631fa

- Classification: feature ablation after experiment 111.
- Hypothesis: Raw `DepTime` may be a noisy HHMM duplicate once `DepMinutes`, cyclic time, and `DepHourCat` are present.
- Change: remove `DepTime` from model inputs while still using the source column to derive engineered time features.
- Result: CV AUC 0.8229 +/- 0.0036, CV time 57.9s.
- Decision: discard. Faster but materially worse than current best.

## Research after three discards

- XGBoost categorical docs: `max_cat_to_onehot` chooses between one-hot and partitioning for categorical features; the current winning setup likely works because small calendar/hour categories are one-hot while high-cardinality airport categories use partitioning. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost parameter docs: `gamma` / `min_split_loss` requires a minimum loss reduction before a split and makes the model more conservative; this is a plausible next test for the very deep current trees. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost interaction-constraints docs: deep trees can capture spurious feature interactions, and constraints can improve generalization when domain-valid interactions are known. Broad constraints are risky here, so a small split-loss penalty is the lower-risk regularization test. Source: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html

## Experiment 115 - 0f11483

- Classification: regularization follow-up after research.
- Hypothesis: A small split-loss penalty can prune marginal deep splits while preserving the useful time and categorical interactions.
- Change: add `gamma=0.05`.
- Result: CV AUC 0.8234 +/- 0.0033, CV time 58.0s.
- Decision: discard. Regularization reduced AUC.

## Experiment 116 - ad96f34

- Classification: schedule retune after experiment 111.
- Hypothesis: The added categorical hour signal may benefit from a slightly larger update at the same tree count.
- Change: `learning_rate=0.0348`.
- Result: CV AUC 0.8240 +/- 0.0036, CV time 58.4s.
- Decision: keep. New valid best.

## Experiment 117 - 8f82f0f

- Classification: schedule retune after experiment 116.
- Hypothesis: A slightly longer, lower-eta schedule may smooth the new best while staying under the runtime cap.
- Change: `n_estimators=585`, `learning_rate=0.0342`.
- Result: CV AUC 0.8237 +/- 0.0032, CV time 59.4s.
- Decision: discard. Under the time cap but worse than current best.

## Experiment 118 - 08a837d

- Classification: schedule retune after experiment 116.
- Hypothesis: The local optimum may be above `learning_rate=0.0348` while keeping the same tree count.
- Change: `learning_rate=0.0351`.
- Result: CV AUC 0.8233 +/- 0.0031, CV time 58.2s.
- Decision: discard. The higher rate overshot the useful range.

## Experiment 119 - a04c88b

- Classification: schedule retune after experiment 116.
- Hypothesis: A slightly lower rate than `0.0348` may keep the AUC gain with less fold variance.
- Change: `learning_rate=0.0347`.
- Result: CV AUC 0.8234 +/- 0.0035, CV time 58.4s.
- Decision: discard. Worse than current best.

## Research after experiment 119

- XGBoost parameter-tuning docs frame `subsample` and `colsample_bytree` as randomness-based overfitting controls, and also recommend pairing smaller step sizes with more rounds when reducing `eta`. The local schedule tests suggest `0.0348` is already the useful point for 575 trees. Source: https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
- XGBoost parameter docs describe `colsample_bytree`, `colsample_bylevel`, and `colsample_bynode` as cumulative column-sampling controls. Since `colsample_bytree=0.625` helped earlier, a mild `colsample_bylevel` test is a plausible next randomness lever. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- The local training slice is class-balanced: 50,000 positive and 50,000 negative labels. That makes `scale_pos_weight` unattractive despite the docs suggesting it for imbalanced AUC tasks.

## Experiment 120 - 06f30d9

- Classification: column-sampling follow-up after research.
- Hypothesis: Mild level-wise column sampling can reduce noisy deep interactions while preserving most per-tree feature diversity.
- Change: add `colsample_bylevel=0.9`.
- Result: CV AUC 0.8234 +/- 0.0036, CV time 56.4s.
- Decision: discard. Faster but worse than current best.

## Synthesis after 120 experiments

- Current best: experiment 116 (`ad96f34`), CV AUC 0.8240 +/- 0.0036, CV time 58.4s.
- The strongest recent win was `DepHourCat` followed by a slight learning-rate increase. This confirms that a low-cardinality categorical time regime is useful, but finer correlated time bins (`DepQuarterCat`) and time ablations did not help.
- The local schedule appears sharp: `learning_rate=0.0348` beat `0.0345`, while `0.0347`, `0.0351`, and a 585-tree lower-eta schedule were worse.
- Regularization via `gamma` and extra level-wise column sampling reduced AUC. The model still wants high-capacity trees, but only with the existing `min_child_weight`, categorical partitioning, and per-tree column sampling balance.
- XGBoost 3.2.0 supports `grow_policy="lossguide"` with `tree_method="hist"` and `max_leaves`; this is the next structural test because it changes how deep-tree capacity is allocated rather than adding another feature or small scalar regularizer. Source: https://xgboost.readthedocs.io/en/latest/parameter.html

## Experiment 121 - 3712049

- Classification: structural growth-policy test after 120-experiment synthesis.
- Hypothesis: Leaf-wise growth may spend deep-tree capacity on the highest-gain nodes under an explicit leaf cap.
- Change: add `grow_policy="lossguide"` and `max_leaves=512`.
- Result: CV AUC 0.8158 +/- 0.0036, CV time 77.9s.
- Decision: discard. Much worse and far over the runtime cap.

## Experiment 122 - b40ee6d

- Classification: depth-cap retune after experiment 116.
- Hypothesis: Reducing maximum depth by one may trim marginal deep interactions without changing other regularization.
- Change: `max_depth=23`.
- Result: CV AUC 0.8231 +/- 0.0035, CV time 57.3s.
- Decision: discard. Faster but worse than current best.

## Experiment 123 - 5ae4698

- Classification: feature-engineering follow-up to categorical regime features.
- Hypothesis: A coarse quantile-based categorical distance regime may complement raw numeric `Distance`.
- Change: add five-category `DistanceBin` while keeping `Distance`.
- Result: CV AUC 0.8236 +/- 0.0039, CV time 59.2s.
- Decision: discard. Worse than current best.

## Research after experiment 123

- Current-best one-model gain diagnostic: highest average-gain features were `Month`, `DepTimeSin`, `UniqueCarrier`, `DayofMonth`, `DepTime`, `Dest`, and `Origin`; highest total-gain features were `Distance`, `DepTime`, `Dest`, `Origin`, `DepTimeSin`, and `DepMinute`.
- The diagnostic supports keeping raw time and distance features. Failed ablations of `DepTime`, `DepHour`, and distance binning are consistent with the model already using those raw/engineered signals effectively.
- Candidate interaction counts from the training slice: `CarrierDayOfWeek` has 140 levels, `CarrierMonth` has 236 levels, while origin/destination weekday interactions exceed 1,800 levels. Given the runtime cap and XGBoost categorical docs, `CarrierDayOfWeek` is the lowest-risk interaction feature to try next. Source: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

## Experiment 124 - 3edc009

- Classification: categorical interaction feature test.
- Hypothesis: Carrier delay patterns may vary by weekday, and a 140-level joint categorical can expose that interaction in one split.
- Change: add `CarrierDayOfWeek`.
- Result: CV AUC 0.8047 +/- 0.0036, CV time 67.8s.
- Decision: discard. Much worse and over the runtime cap.

## Experiment 125 - aa6a093

- Classification: schedule retune after experiment 116.
- Hypothesis: A few fewer trees with slightly larger steps may preserve AUC while freeing runtime.
- Change: `n_estimators=570`, `learning_rate=0.035`.
- Result: CV AUC 0.8235 +/- 0.0035, CV time 59.0s.
- Decision: discard. Worse than current best.

## Experiment 126 - 422eacb

- Classification: row-sampling retune after experiment 116.
- Hypothesis: Slightly more row signal than `subsample=0.95` may improve AUC without going all the way to full-row training.
- Change: `subsample=0.975`.
- Result: CV AUC 0.8241 +/- 0.0038, CV time 60.2s.
- Decision: discard for runtime. AUC improved but exceeded the 60s cap.

## Research after experiment 126

- XGBoost parameter docs describe `subsample` as row sampling before growing each tree, with range `(0, 1]`. Source: https://xgboost.readthedocs.io/en/latest/parameter.html
- XGBoost tuning notes list `subsample` and `colsample_bytree` as randomness controls for robustness to noise. The failed `subsample=0.9`, failed `1.0`, and promising-but-slow `0.975` suggest a narrow row-sampling window just above `0.95`. Source: https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

## Experiment 127 - 63834f3

- Classification: row-sampling follow-up to experiment 126.
- Hypothesis: `subsample=0.97` may keep most of the AUC lift from `0.975` while coming under the runtime cap.
- Change: `subsample=0.97`.
- Result: CV AUC 0.8238 +/- 0.0033, CV time 60.1s.
- Decision: discard. Worse than current best and still over the cap.

## Experiment 128 - 5eef80d

- Classification: row-sampling follow-up to experiment 127.
- Hypothesis: `subsample=0.965` may capture the useful effect of `0.975` with enough runtime savings.
- Change: `subsample=0.965`.
- Result: CV AUC 0.8238 +/- 0.0033, CV time 60.9s.
- Decision: discard. Worse than current best and over the cap.

## Experiment 129 - 2137b2e

- Classification: column-sampling retune after experiment 116.
- Hypothesis: Slightly stronger per-tree feature sampling may regularize and reduce runtime.
- Change: `colsample_bytree=0.6`.
- Result: CV AUC 0.8240 +/- 0.0036, CV time 61.8s.
- Decision: discard. Tied current best at printed precision and exceeded the runtime cap.

## Research after experiment 129

- The only post-116 setting that beat the current best AUC was `subsample=0.975`, but it missed the cap by 0.2s. Smaller subsample values did not preserve the gain.
- XGBoost tuning docs recommend reducing rounds when changing sampling/regularization if runtime is the active constraint. The highest-value exp130 candidate is therefore to keep `subsample=0.975` and reduce tree count slightly rather than continue lowering subsample. Source: https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

## Experiment 130 - e5726e3

- Classification: runtime-aware follow-up to experiment 126.
- Hypothesis: A small tree-count reduction may preserve the `subsample=0.975` AUC lift while bringing CV time under the cap.
- Change: `n_estimators=570`, `subsample=0.975`.
- Result: CV AUC 0.8241 +/- 0.0037, CV time 61.2s.
- Decision: discard for runtime. AUC improved but remained over the cap.

## Synthesis after 130 experiments

- Current valid best remains experiment 116 (`ad96f34`): CV AUC 0.8240 +/- 0.0036, CV time 58.4s.
- The most informative near miss is the high-subsample band: `subsample=0.975` produced 0.8241 twice, but both valid attempts exceeded the runtime cap. Reducing subsample to 0.97/0.965 lost the AUC lift, and reducing trees to 570 did not save enough time.
- Structural alternatives after the best were poor: `lossguide` collapsed AUC and ran slowly, depth 23 underfit, extra column sampling did not improve, and the carrier-weekday interaction was strongly harmful.
- The best next direction, if continuing, is a joint runtime/AUC compromise around high subsample with a larger runtime cut, such as fewer trees with a slightly compensating learning rate, but the recent evidence shows the valid improvement margin is very narrow.
