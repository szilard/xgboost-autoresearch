# Research Log

Run tag: may7

Setup notes:
- Branch created from the current HEAD.
- Baseline not run yet.
- Training uses 5-fold cross-validation on `data-cache/2005-slice1-100k.csv`.

## Baseline

commit: 7d0b368
CV AUC: 0.7445 +/- 0.0043
Notes: untouched starter `train.py`; 30 trees, depth 6, learning rate 0.1, native categorical support.

## Research Before Experiment 1

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost categorical data tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

Relevant notes:
- `learning_rate` shrinks each boosting update; lower values are commonly paired with more trees.
- `max_depth`, `min_child_weight`, `gamma`, row sampling, and column sampling all control tree complexity/overfitting.
- Native categorical training expects pandas categorical dtype with `enable_categorical=True`; `tree_method="hist"` is supported for categorical features.

## Experiment 1

Hypothesis: Baseline with 30 trees is likely low-capacity. More rounds with lower learning rate plus row/column sampling and higher `min_child_weight` should improve ranking while controlling overfit.
Type: exploration
Change: 200 trees, depth 4, learning rate 0.05, min_child_weight 5, subsample/colsample_bytree 0.85, explicit hist/auc.
Result: ba8f26d CV AUC 0.7487 +/- 0.0046. Keep. More rounds plus regularization improved over 0.7445 baseline.

## Experiment 2

Hypothesis: Follow up on ba8f26d by lowering learning rate and increasing boosting rounds. If the model is still underfit, 400 rounds at 0.03 should improve AUC; otherwise it should be neutral/worse.
Type: follow-up
Change: n_estimators 200 -> 400, learning_rate 0.05 -> 0.03; keep depth 4 and sampling regularization.
Result: 2ec6b62 CV AUC 0.7496 +/- 0.0048. Keep. More rounds/lower rate gave +0.0009.

## Experiment 3

Hypothesis: Follow up on 2ec6b62 by allowing deeper interaction structure. Depth 5 may capture carrier/origin/destination/time interactions, while increasing `min_child_weight` to 10 should limit fragile splits.
Type: follow-up
Change: max_depth 4 -> 5, min_child_weight 5 -> 10.
Result: d84eab8 CV AUC 0.7579 +/- 0.0048. Keep. Strong evidence that deeper interactions matter.

## Experiment 4

Hypothesis: If carrier/route/time interactions still need more depth, depth 6 may help, but only with stronger `min_child_weight` to avoid sparse leaf overfit.
Type: follow-up
Change: max_depth 5 -> 6, min_child_weight 10 -> 20.
Result: 49206b8 CV AUC 0.7657 +/- 0.0051. Keep. Deeper interactions still help substantially.

## Experiment 5

Hypothesis: Continue the successful depth/regularization sweep. Depth 7 may capture still richer route/time interactions; `min_child_weight=40` should prevent tiny category leaves from dominating.
Type: follow-up
Change: max_depth 6 -> 7, min_child_weight 20 -> 40.
Result: 23da8fc CV AUC 0.7691 +/- 0.0053. Keep. Gain is smaller than depth 6 but still meaningful.

## Experiment 6

Hypothesis: At depth 7, `min_child_weight=40` may be over-regularizing useful route/time/category splits. Reducing it to 20 tests whether more leaf flexibility improves AUC.
Type: follow-up
Change: min_child_weight 40 -> 20 at depth 7.
Result: dfd242a CV AUC 0.7702 +/- 0.0052. Keep. Relaxing child weight helped, so useful splits were being constrained.

## Experiment 7

Hypothesis: Since reducing `min_child_weight` from 40 to 20 helped, lowering to 10 may allow additional useful interactions. If overfit begins, CV should drop.
Type: follow-up
Change: min_child_weight 20 -> 10 at depth 7.
Result: 922a158 CV AUC 0.7686 +/- 0.0046. Discard. Too much leaf flexibility hurt versus dfd242a.

## Experiment 8

Hypothesis: The child-weight optimum at depth 7 may lie between 10 and 20. Test `min_child_weight=15` as a narrow interpolation after 20 kept and 10 discarded.
Type: follow-up
Change: min_child_weight 20 -> 15 at depth 7.
Result: 408922b CV AUC 0.7704 +/- 0.0049. Keep. Small gain; best child-weight region appears near 15-20.

## Experiment 9

Hypothesis: Depth 8 might capture additional interactions if paired with stronger leaf-mass regularization. Use `min_child_weight=30` rather than 15 to avoid sparse deep leaves.
Type: follow-up
Change: max_depth 7 -> 8, min_child_weight 15 -> 30.
Result: 9566bf1 CV AUC 0.7736 +/- 0.0052. Keep. Another depth increase helped when regularized.

## Experiment 10

Hypothesis: Continue depth/regularization sweep. Depth 9 may help only if leaf mass is strongly constrained, so increase `min_child_weight` to 60.
Type: follow-up
Change: max_depth 8 -> 9, min_child_weight 30 -> 60.
Result: 1a8092b CV AUC 0.7722 +/- 0.0052. Discard. Depth 9/60 is worse than depth 8/30; useful depth boundary likely around 8.

## Research Before Experiment 11

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost categorical data tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost tree methods documentation: https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

Relevant notes:
- `max_cat_to_onehot` controls when categorical features use one-hot splits versus partitioned categorical splits.
- `max_cat_threshold` limits categories considered per split and can act as categorical overfit control.
- `grow_policy="lossguide"` with `max_leaves` is available for hist trees and can change how capacity is allocated.
- `reg_lambda`, `reg_alpha`, and `gamma` make the model more conservative through leaf-weight and split regularization.

## Experiment 11

Hypothesis: Low-cardinality categoricals such as Month, DayOfWeek, DayofMonth, and carrier may benefit from one-hot categorical splits, while high-cardinality Origin/Dest should still use partitioning. `max_cat_to_onehot=32` should make that split explicit.
Type: exploration
Change: add `max_cat_to_onehot=32` to current best depth 8/min_child_weight 30 model.
Result: bbaf29f CV AUC 0.7620 +/- 0.0052. Discard. Native partitioning is much better than forcing one-hot for these categoricals.

## Experiment 12

Hypothesis: Since partition-based categorical splitting is better than one-hot here, allowing more categories to be considered per split may help high-cardinality Origin/Dest features. `max_cat_threshold=256` relaxes the default threshold.
Type: exploration
Change: add `max_cat_threshold=256` to current best depth 8/min_child_weight 30 model.
Result: 06ab09b CV AUC 0.7735 +/- 0.0053. Discard. Near-neutral but not better than 9566bf1.

## Experiment 13

Hypothesis: A small `gamma` may prune weak deep splits in the current depth 8 model without changing the useful interaction capacity.
Type: exploration
Change: add `gamma=0.1`.
Result: 8e13809 CV AUC 0.7736 +/- 0.0053. Discard. Equal rounded AUC; added complexity without improvement.

## Research Before Experiment 14

Sources:
- scikit-learn time-related feature engineering example: https://scikit-learn.org/1.5/auto_examples/applications/plot_cyclical_feature_engineering.html
- Flight-delay literature search surfaced temporal/spatial features as common predictors, including month, day of week, departure time, origin, destination, and distance.

Relevant notes:
- Cyclic sine/cosine encodings represent periodic variables without a discontinuity between the end and start of the period.
- Flight delay studies commonly use temporal and spatial schedule features; this dataset already has those raw inputs, so engineered time-of-day features are a low-risk first feature change.

## Experiment 14

Hypothesis: Raw HHMM `DepTime` is awkward for trees because 959 and 1000 are close in time but numerically far, and midnight wraps. Adding departure hour/minute/minutes plus cyclic time-of-day features may expose useful delay patterns while keeping original features.
Type: exploration
Change: add `DepHour`, `DepMinute`, `DepMinutes`, `DepTimeSin`, and `DepTimeCos` inside `prepare(df)`.
Result: f02d374 CV AUC 0.7784 +/- 0.0054. Keep. Time-of-day feature engineering produced a large gain.

## Experiment 15

Hypothesis: Raw HHMM `DepTime` may now be redundant or slightly harmful because engineered minute/cyclic features represent time more cleanly. Removing raw `DepTime` is a simplification ablation.
Type: ablation/simplification
Change: remove `DepTime` from `num_cols`; still derive engineered time features from `df["DepTime"]`.
Result: 09c2a1c CV AUC 0.7787 +/- 0.0052. Keep. Simpler and slightly better; raw HHMM was not needed.

## Experiment 16

Hypothesis: Calendar fields are currently categorical only. Adding numeric and cyclic encodings for Month, DayOfMonth, and DayOfWeek plus an `IsWeekend` indicator may expose seasonal/weekly adjacency patterns while preserving categorical splits.
Type: follow-up
Change: add calendar numeric/cyclic features and weekend indicator inside `prepare(df)`.
Result: ed3e2f1 CV AUC 0.7784 +/- 0.0052. Discard. More calendar features added complexity and underperformed the simpler time-only feature model.

## Experiment 17

Hypothesis: Delay propensity may be route-specific. Adding an Origin-Dest categorical interaction should let the model split directly on routes instead of relying on deeper combinations of separate Origin and Dest splits.
Type: exploration
Change: add `Route` categorical feature derived from Origin and Dest.
Result: 93377f1 CV AUC 0.7783 +/- 0.0054. Discard. Route category added runtime and did not improve AUC.

## Experiment 18

Hypothesis: After adding engineered time features, the model may need less tree depth to capture interactions. Re-test depth 7/min_child_weight 15 for a simpler and faster model.
Type: ablation/simplification
Change: max_depth 8 -> 7 and min_child_weight 30 -> 15 on the time-feature model.
Result: 1bdc8a4 CV AUC 0.7742 +/- 0.0047. Discard. Depth 8 remains important even after time feature engineering.

## Research Before Experiment 19

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost tree methods documentation: https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html
- XGBoost DART tutorial: https://xgboost.readthedocs.io/en/release_2.0.0/tutorials/dart.html

Relevant notes:
- With `hist`, higher `max_bin` can sometimes improve accuracy while preserving performance.
- `subsample` and `colsample_*` are standard overfit controls, and `colsample_*` parameters compound.
- DART can regularize by dropping trees but is slower because dropout prevents prediction-buffer reuse.

## Experiment 19

Hypothesis: The engineered time features create more meaningful continuous split points. Raising `max_bin` from default 256 to 512 may improve histogram split resolution without too much runtime cost.
Type: exploration
Change: add `max_bin=512`.
Result: 8031221 CV AUC 0.7783 +/- 0.0052. Discard. Higher histogram resolution did not improve AUC.

## Experiment 20

Hypothesis: After adding time features, the model may still benefit from a slower boosting schedule. Increase trees to 600 and lower learning rate to 0.02, mirroring the earlier successful 200->400 schedule change.
Type: follow-up
Change: n_estimators 400 -> 600, learning_rate 0.03 -> 0.02.
Result: a81641c CV AUC 0.7795 +/- 0.0050. Keep. Slower schedule improved again, runtime still acceptable.

## Research Before Experiment 21

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost feature interaction constraints tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
- Flight-delay feature literature search surfaced time of day, day of week, season, flight distance, airline, origin, and destination as recurring predictors.

Relevant notes:
- Lower learning rates paired with more boosting rounds can reduce step size while preserving capacity.
- Regularization controls worth revisiting include `reg_lambda`, `reg_alpha`, `subsample`, and column sampling.
- Flight-delay studies repeatedly emphasize temporal/spatial/schedule features; route interaction did not help in this dataset, but distance/time transformations remain plausible.

## Experiment 21

Hypothesis: The 600-tree/0.02 schedule improved over 400/0.03. Try another proportional step to 800 trees and 0.015 learning rate to see whether smoother boosting still improves AUC.
Type: follow-up
Change: n_estimators 600 -> 800, learning_rate 0.02 -> 0.015.
Result: 6ff2d6c CV AUC 0.7797 +/- 0.0051. Keep. Small gain; runtime remains acceptable but diminishing returns are visible.

## Experiment 22

Hypothesis: One more lower-rate step may give a small AUC gain, but returns are diminishing. Try 1000 trees at learning rate 0.012 while staying below the runtime cap.
Type: follow-up
Change: n_estimators 800 -> 1000, learning_rate 0.015 -> 0.012.
Result: 63104e6 CV AUC 0.7799 +/- 0.0053. Keep. Small gain; CV time ~24s leaves some budget but schedule returns are diminishing.

## Experiment 23

Hypothesis: With 1000 trees, slightly stronger leaf-mass regularization may reduce overfit. Test `min_child_weight=40` around the current depth 8 schedule.
Type: follow-up
Change: min_child_weight 30 -> 40.
Result: f2efbc1 CV AUC 0.7784 +/- 0.0053. Discard. Stronger child weight was too conservative.

## Experiment 24

Hypothesis: Since `min_child_weight=40` was too conservative, test whether 20 improves by allowing more useful deep splits under the slower 1000-tree schedule.
Type: follow-up
Change: min_child_weight 30 -> 20.
Result: 066e231 CV AUC 0.7805 +/- 0.0050. Keep. Lower child weight helps under the slower schedule.

## Experiment 25

Hypothesis: Since lowering child weight from 30 to 20 helped, try 15 as a nearby point. It may allow more useful splits, but risks the overfit seen with too-low values earlier.
Type: follow-up
Change: min_child_weight 20 -> 15.
Result: bedc3db CV AUC 0.7808 +/- 0.0050. Keep. More leaf flexibility helps at this schedule.

## Experiment 26

Hypothesis: Test the lower boundary of leaf flexibility under the 1000-tree model. `min_child_weight=10` may improve further, but prior shallower/smaller models overfit at 10.
Type: follow-up
Change: min_child_weight 15 -> 10.
Result: ccde93f CV AUC 0.7797 +/- 0.0048. Discard. Too much leaf flexibility hurts; best child-weight is around 15.

## Experiment 27

Hypothesis: The optimum may be between 10 and 15. Test `min_child_weight=12` as a narrow interpolation around the current best.
Type: follow-up
Change: min_child_weight 15 -> 12.
Result: 1315f01 CV AUC 0.7802 +/- 0.0045. Discard. Worse than min_child_weight 15.

## Experiment 28

Hypothesis: Test the upper side near the current best. `min_child_weight=18` may split the difference between best 15 and weaker 20.
Type: follow-up
Change: min_child_weight 15 -> 18.
Result: a370e42 CV AUC 0.7809 +/- 0.0049. Keep. Slightly better than 15 and 20.

## Experiment 29

Hypothesis: Fine-tune around the current peak by testing `min_child_weight=17`, one step below the best 18.
Type: follow-up
Change: min_child_weight 18 -> 17.
Result: d10606c CV AUC 0.7806 +/- 0.0050. Discard. Worse than 18.

## Experiment 30

Hypothesis: Test the upper immediate neighbor of the current best. `min_child_weight=19` may match or beat 18 if the optimum is slightly higher.
Type: follow-up
Change: min_child_weight 18 -> 19.
Result: a3b746f CV AUC 0.7808 +/- 0.0052. Discard. Close but not better than 18.

## Research Before Experiment 31

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost random forest tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/rf.html

Relevant notes:
- `colsample_bytree`, `colsample_bylevel`, and `colsample_bynode` control random feature subsampling and compound when used together.
- Row subsampling and column subsampling can reduce overfit, but with few curated features they may also block useful interactions.
- L1/L2 regularization remain available if removing column randomness overfits.

## Experiment 31

Hypothesis: The current model has a small feature set with important interactions. Setting `colsample_bytree=1.0` may help each tree see all time, distance, and categorical predictors instead of randomly hiding some.
Type: exploration
Change: colsample_bytree 0.85 -> 1.0.
Result: ce7470c CV AUC 0.7792 +/- 0.0051. Discard. Column sampling is beneficial; all-columns overfits or reduces useful diversity.

## Experiment 32

Hypothesis: Since removing column sampling hurt, stronger column randomness may improve generalization. Test `colsample_bytree=0.75`.
Type: follow-up
Change: colsample_bytree 0.85 -> 0.75.
Result: b4e34e4 CV AUC 0.7811 +/- 0.0051. Keep. More column sampling improved generalization.

## Experiment 33

Hypothesis: Continue column-sampling sweep. `colsample_bytree=0.65` may improve diversity further, but could hide too many key predictors.
Type: follow-up
Change: colsample_bytree 0.75 -> 0.65.
Result: 86087a4 CV AUC 0.7819 +/- 0.0050. Keep. Stronger column sampling improves and speeds training slightly.

## Experiment 34

Hypothesis: Continue column sampling lower to 0.55. If diversity is the main gain, this may improve; if key features are hidden too often, AUC will drop.
Type: follow-up
Change: colsample_bytree 0.65 -> 0.55.
Result: d0c6fb5 CV AUC 0.7816 +/- 0.0052. Discard. Too much column sampling loses signal.

## Experiment 35

Hypothesis: The column-sampling optimum may be between 0.55 and 0.65. Test `colsample_bytree=0.60`.
Type: follow-up
Change: colsample_bytree 0.65 -> 0.60.
Result: 3b5621f CV AUC 0.7819 +/- 0.0050. Discard. Equal rounded AUC; keep existing 0.65.

## Experiment 36

Hypothesis: Test the upper side near the column-sampling optimum. `colsample_bytree=0.70` may beat 0.65 while staying more regularized than 0.75.
Type: follow-up
Change: colsample_bytree 0.65 -> 0.70.
Result: a5e1541 CV AUC 0.7816 +/- 0.0050. Discard. Worse than 0.65; column-sampling optimum appears around 0.65.

## Research Before Experiment 37

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- Friedman, "Stochastic Gradient Boosting" bibliographic page: https://ideas.repec.org/a/eee/csdana/v38y2002i4p367-378.html

Relevant notes:
- `subsample` samples rows before growing each tree and is a standard overfit/variance control.
- Stochastic gradient boosting motivates row subsampling as a way to decorrelate trees and improve generalization.

## Experiment 37

Hypothesis: Column sampling improved materially. More row-level stochasticity may complement it, so reduce `subsample` from 0.85 to 0.75.
Type: exploration
Change: subsample 0.85 -> 0.75.
Result: 2a91b2c CV AUC 0.7812 +/- 0.0051. Discard. More row sampling regularization hurt.

## Experiment 38

Hypothesis: Since lower subsample hurt, try less row sampling regularization. `subsample=0.95` may preserve signal while keeping slight stochasticity.
Type: follow-up
Change: subsample 0.85 -> 0.95.
Result: e1cbaf9 CV AUC 0.7818 +/- 0.0050. Discard. Close but not better than 0.85.

## Experiment 39

Hypothesis: Interpolate row subsampling between best 0.85 and close 0.95. `subsample=0.90` may recover the small gap.
Type: follow-up
Change: subsample 0.85 -> 0.90.
Result: d9d5c93 CV AUC 0.7821 +/- 0.0053. Keep. Row subsampling optimum appears near 0.90.

## Experiment 40

Hypothesis: Fine-tune around the row-sampling optimum by testing `subsample=0.88`, slightly below the best 0.90.
Type: follow-up
Change: subsample 0.90 -> 0.88.
Result: 64b2866 CV AUC 0.7818 +/- 0.0053. Discard. Worse than 0.90.

## Research Before Experiment 41

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost tree methods documentation: https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

Relevant notes:
- For `tree_method="hist"`, `grow_policy="lossguide"` can grow the node with highest loss change instead of depthwise level growth.
- `max_leaves` caps capacity for lossguide trees.
- This is a different capacity allocation strategy, not just another depth tweak.

## Experiment 41

Hypothesis: Loss-guided growth with a leaf cap may allocate splits more efficiently than fixed depthwise depth 8. Use `max_depth=0`, `grow_policy="lossguide"`, and `max_leaves=128`.
Type: exploration
Change: switch from depthwise max_depth 8 to lossguide max_leaves 128.
Result: 1b4c93d CV AUC 0.7859 +/- 0.0049. Discard. Large improvement, but total printed training time is about 68s, over the experiment timeout.

## Experiment 42

Hypothesis: Lossguide was promising but too slow at 128 leaves. A 64-leaf cap may preserve some gain while keeping total runtime under the timeout.
Type: follow-up
Change: lossguide with `max_leaves=64` and `max_depth=0`.
Result: 70f193c CV AUC 0.7769 +/- 0.0051. Discard. Runtime was acceptable but 64 leaves underfit badly.

## Experiment 43

Hypothesis: Lossguide may need more than 64 leaves but fewer than the too-slow 128. Test `max_leaves=96` with an explicit timeout.
Type: follow-up
Change: lossguide with `max_leaves=96` and `max_depth=0`.
Result: 9610702 CV AUC 0.7825 +/- 0.0050. Keep. Total printed training time ~56.5s, under the timeout, and better than depthwise best.

## Experiment 44

Hypothesis: The too-slow 128-leaf model suggests lossguide leaf capacity helps. Trade fewer rounds for more leaves: 850 trees at learning rate 0.014 with 112 leaves, keeping roughly the same boosting budget but more per-tree capacity.
Type: follow-up
Change: n_estimators 1000 -> 850, learning_rate 0.012 -> 0.014, max_leaves 96 -> 112.
Result: 03d5217 CV AUC 0.7840 +/- 0.0051. Keep. Total printed time ~52.7s, and leaf capacity gain helped.

## Experiment 45

Hypothesis: The 128-leaf model had the best AUC but was too slow at 1000 trees. Use 750 trees and learning rate 0.016 to keep the boosting budget similar while making 128 leaves fit under the timeout.
Type: follow-up
Change: n_estimators 850 -> 750, learning_rate 0.014 -> 0.016, max_leaves 112 -> 128.
Result: 007f281 CV AUC 0.7855 +/- 0.0052. Keep. Total printed time ~51.0s, close to invalid 128-leaf AUC but valid runtime.

## Experiment 46

Hypothesis: Push the leaves-for-rounds trade further. 650 trees at learning rate 0.0185 with 144 leaves keeps a similar total shrinkage budget while increasing per-tree capacity.
Type: follow-up
Change: n_estimators 750 -> 650, learning_rate 0.016 -> 0.0185, max_leaves 128 -> 144.
Result: a1c2b6b CV AUC 0.7872 +/- 0.0051. Keep. Total printed time ~48.2s; larger trees with fewer rounds help.

## Experiment 47

Hypothesis: Continue leaves-for-rounds trade. 550 trees at learning rate 0.022 with 160 leaves may improve if larger trees are carrying useful route/time interactions.
Type: follow-up
Change: n_estimators 650 -> 550, learning_rate 0.0185 -> 0.022, max_leaves 144 -> 160.
Result: 65a84fa CV AUC 0.7885 +/- 0.0051. Keep. Total printed time ~43.9s; trend still positive.

## Experiment 48

Hypothesis: Larger lossguide trees still seem valuable. Try 450 trees at learning rate 0.027 with 192 leaves to keep total shrinkage similar while increasing per-tree capacity.
Type: follow-up
Change: n_estimators 550 -> 450, learning_rate 0.022 -> 0.027, max_leaves 160 -> 192.
Result: a28fa57 CV AUC 0.7906 +/- 0.0046. Keep. Total printed time ~40.1s; larger lossguide trees are strongly helping.

## Experiment 49

Hypothesis: Continue the leaves-for-rounds trajectory. 350 trees at learning rate 0.034 with 256 leaves may capture richer interactions while staying within runtime.
Type: follow-up
Change: n_estimators 450 -> 350, learning_rate 0.027 -> 0.034, max_leaves 192 -> 256.
Result: a18be2e CV AUC 0.7918 +/- 0.0050. Keep. Total printed time ~38.4s; larger leaves still improve.

## Experiment 50

Hypothesis: Push larger lossguide trees further. 250 trees at learning rate 0.048 with 384 leaves keeps a similar aggregate shrinkage budget but may start overfitting.
Type: follow-up
Change: n_estimators 350 -> 250, learning_rate 0.034 -> 0.048, max_leaves 256 -> 384.
Result: 2d47dfd CV AUC 0.7939 +/- 0.0050. Keep. Total printed time ~36.0s; larger lossguide trees continue to improve.

## Research Before Experiment 51

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost tree methods documentation: https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html

Relevant notes:
- `max_leaves` is the explicit capacity cap for lossguide hist trees.
- `min_child_weight`, `gamma`, `reg_lambda`, and `reg_alpha` are the main levers to make large trees more conservative.
- Learning rate shrinkage still matters; when reducing rounds, keep the product of rounds and learning rate in the same rough range before tuning further.

## Experiment 51

Hypothesis: Continue the successful larger-tree/fewer-round trajectory. 200 trees at learning rate 0.06 with 512 leaves keeps aggregate shrinkage similar and may capture even richer high-order interactions.
Type: follow-up
Change: n_estimators 250 -> 200, learning_rate 0.048 -> 0.06, max_leaves 384 -> 512.
Result: 6085269 CV AUC 0.7939 +/- 0.0046. Discard. Equal rounded AUC; keep simpler 384-leaf model.

## Experiment 52

Hypothesis: The 512-leaf/fewer-round variant tied but did not improve. Keep 384 leaves and try smoother boosting with 300 trees at learning rate 0.04.
Type: follow-up
Change: n_estimators 250 -> 300, learning_rate 0.048 -> 0.04.
Result: 75d7f37 CV AUC 0.7943 +/- 0.0051. Keep. Smoother boosting improves at 384 leaves.

## Experiment 53

Hypothesis: Continue smoother boosting at fixed 384 leaves. 350 trees at learning rate 0.034 may improve further if the previous 250-tree model was too coarse.
Type: follow-up
Change: n_estimators 300 -> 350, learning_rate 0.04 -> 0.034.
Result: 38316c9 CV AUC 0.7946 +/- 0.0050. Keep. More rounds at 384 leaves improved further; total printed time ~50.8s.

## Experiment 54

Hypothesis: Continue smoother boosting at fixed 384 leaves. 400 trees at learning rate 0.03 may improve, but it is close to the timeout.
Type: follow-up
Change: n_estimators 350 -> 400, learning_rate 0.034 -> 0.03.
Result: 568aca6 timeout before CV AUC. Crash/timeout. Revert to 38316c9.

## Experiment 55

Hypothesis: Interpolate between valid 350 trees and timed-out 400 trees. 375 trees at learning rate 0.032 may improve while still fitting under the timeout.
Type: follow-up
Change: n_estimators 350 -> 375, learning_rate 0.034 -> 0.032.
Result: e86b1ac CV AUC 0.7952 +/- 0.0049. Keep. Total printed time ~54.7s; near timeout but valid and best so far.

## Experiment 56

Hypothesis: The 375-tree model improved and 400 timed out. Try 390 trees at learning rate 0.031 to squeeze a bit more smoother boosting under the limit.
Type: follow-up
Change: n_estimators 375 -> 390, learning_rate 0.032 -> 0.031.
Result: a290479 CV AUC 0.7958 +/- 0.0047. Keep. Total printed time ~56.6s; near timeout but valid and new best.

## Experiment 57

Hypothesis: Try one last narrow tree-count increase before the timeout boundary. 395 trees at learning rate 0.0305 may improve if it completes in time.
Type: follow-up
Change: n_estimators 390 -> 395, learning_rate 0.031 -> 0.0305.
Result: 37f0f3e CV AUC 0.7953 +/- 0.0048. Discard. More trees/lower rate underperformed 390.

## Experiment 58

Hypothesis: The current lossguide model has much larger trees than earlier models. Stronger leaf-mass regularization (`min_child_weight=25`) may reduce overfit without changing the tuned leaf/round budget.
Type: follow-up
Change: min_child_weight 18 -> 25.
Result: 6489367 CV AUC 0.7927 +/- 0.0050. Discard. Stronger child weight over-regularized the high-leaf model.

## Experiment 59

Hypothesis: Since higher child weight hurt, try more leaf flexibility. `min_child_weight=12` may let large lossguide trees use their capacity better.
Type: follow-up
Change: min_child_weight 18 -> 12.
Result: 7d563ed CV AUC 0.7964 +/- 0.0047. Keep. Total printed time ~57.8s; more leaf flexibility helped.

## Experiment 60

Hypothesis: Continue the child-weight direction. `min_child_weight=8` may improve if the high-leaf lossguide model still benefits from more flexible leaves, but overfit risk is increasing.
Type: follow-up
Change: min_child_weight 12 -> 8.
Result: 57c3219 CV AUC 0.7957 +/- 0.0048. Discard. Too much leaf flexibility hurt; best child weight is around 12.

## Research Before Experiment 61

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html

Relevant notes:
- `gamma`/`min_split_loss` requires a minimum loss reduction for a split, making trees more conservative.
- `reg_lambda` and `reg_alpha` regularize leaf weights, while `gamma` targets split creation directly.

## Experiment 61

Hypothesis: The high-leaf lossguide model may benefit from pruning weak splits. Add a small `gamma=0.05` while keeping the current best schedule and child weight.
Type: exploration
Change: add `gamma=0.05`.
Result: dd71459 CV AUC 0.7964 +/- 0.0047. Discard. Equal rounded AUC with extra complexity.

## Experiment 62

Hypothesis: More leaf flexibility helped. Lower L2 leaf-weight regularization (`reg_lambda=0.5`) may improve if the default is too conservative for this balanced dataset.
Type: exploration
Change: add `reg_lambda=0.5`.
Result: 7f28f86 CV AUC 0.7959 +/- 0.0048. Discard. Less L2 regularization hurt.

## Experiment 63

Hypothesis: Since reducing L2 regularization hurt, increasing `reg_lambda` to 2.0 may stabilize leaf weights in the high-capacity model.
Type: follow-up
Change: add `reg_lambda=2.0`.
Result: 8482685 CV AUC 0.7952 +/- 0.0044. Discard. More L2 regularization also hurt.

## Research Before Experiment 64

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html

Relevant notes:
- Column sampling can be applied by tree, by level, and by node, and the effects compound.
- The current model benefits from tree-level column sampling, so node-level sampling is a plausible high-capacity lossguide regularizer.
- `max_delta_step` is mainly recommended for extremely imbalanced logistic problems, which does not match the balanced slices here.

## Experiment 64

Hypothesis: High-leaf lossguide trees may overuse the same strong predictors at many nodes. Add `colsample_bynode=0.8` to introduce split-level feature randomness while retaining the tuned `colsample_bytree=0.65`.
Type: exploration
Change: add `colsample_bynode=0.8`.
Result: 942a0cf CV AUC 0.7944 +/- 0.0049. Discard. Node-level feature sampling hurt.

## Experiment 65

Hypothesis: Distance may have nonlinear effects such as short-haul congestion or long-haul schedule differences. Add log and square-root distance transforms while keeping raw distance.
Type: exploration
Change: add `DistanceLog` and `DistanceSqrt` inside `prepare(df)`.
Result: 7247d86 timeout before CV AUC. Crash/timeout. Extra distance features pushed the near-limit model over runtime.

## Experiment 66

Hypothesis: The current best is near the timeout. Try 385 trees at learning rate 0.0315, between 375 and 390, to see whether it preserves AUC with a little runtime headroom.
Type: follow-up
Change: n_estimators 390 -> 385, learning_rate 0.031 -> 0.0315.
Result: 1070d81 CV AUC 0.7961 +/- 0.0046. Discard. Slightly worse than 390-tree best.

## Research Before Experiment 67

Sources:
- XGBoost categorical data tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html

Relevant notes:
- Native categorical partitioning splits on category sets.
- `max_cat_threshold` caps the number of categories considered per split and can act as overfit control.
- Earlier `max_cat_threshold=256` was neutral on the depthwise model, but lossguide/high-leaf capacity may interact differently.

## Experiment 67

Hypothesis: High-leaf lossguide trees can use richer Origin/Dest categorical partitions than the earlier depthwise model. Retry `max_cat_threshold=256` on the current best.
Type: exploration
Change: add `max_cat_threshold=256`.
Result: 431f9f7 timeout before CV AUC. Crash/timeout. Larger categorical threshold is too expensive here.

## Experiment 68

Hypothesis: Instead of allowing larger categorical partitions, lower `max_cat_threshold` to 32. This may regularize and speed high-cardinality Origin/Dest splits.
Type: follow-up
Change: add `max_cat_threshold=32`.
Result: c7e711e CV AUC 0.7962 +/- 0.0045. Discard. Slightly worse than default threshold.

## Experiment 69

Hypothesis: Try another leaves-for-rounds point after categorical threshold attempts failed. 330 trees at learning rate 0.0365 with 448 leaves may improve if extra leaf capacity still helps.
Type: follow-up
Change: n_estimators 390 -> 330, learning_rate 0.031 -> 0.0365, max_leaves 384 -> 448.
Result: 6e3f9c3 CV AUC 0.7966 +/- 0.0041. Keep. Total printed time ~53.2s; new best with more leaf capacity.

## Experiment 70

Hypothesis: Continue the leaf-capacity direction. 300 trees at learning rate 0.04 with 512 leaves may improve if high-capacity lossguide trees still generalize.
Type: follow-up
Change: n_estimators 330 -> 300, learning_rate 0.0365 -> 0.04, max_leaves 448 -> 512.
Result: 918f195 CV AUC 0.7973 +/- 0.0044. Keep. Total printed time ~53.8s; larger leaves still improve.

## Research Before Experiment 71

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost tree methods documentation: https://xgboost.readthedocs.io/en/stable/treemethod.html

Relevant notes:
- `max_leaves` directly controls lossguide tree capacity.
- `hist` is the fastest tree method and remains the right method here.
- `max_cached_hist_node` exists for deep trees, but the default is high and not an obvious accuracy lever.

## Experiment 71

Hypothesis: Continue leaf/round trade beyond 512 leaves. 250 trees at learning rate 0.048 with 640 leaves may improve if larger trees are still carrying useful interactions.
Type: follow-up
Change: n_estimators 300 -> 250, learning_rate 0.04 -> 0.048, max_leaves 512 -> 640.
Result: a80412e CV AUC 0.7979 +/- 0.0045. Keep. Total printed time ~52.8s; larger leaf cap continues to help.

## Experiment 72

Hypothesis: Continue larger leaves with fewer rounds. 200 trees at learning rate 0.06 with 768 leaves may improve if per-tree interactions remain the main driver.
Type: follow-up
Change: n_estimators 250 -> 200, learning_rate 0.048 -> 0.06, max_leaves 640 -> 768.
Result: a2d1397 CV AUC 0.7962 +/- 0.0038. Discard. Too few rounds / too much per-tree capacity.

## Experiment 73

Hypothesis: The 640-leaf model may benefit from slightly smoother boosting. Try 275 trees at learning rate 0.044 while keeping 640 leaves.
Type: follow-up
Change: n_estimators 250 -> 275, learning_rate 0.048 -> 0.044.
Result: c4573f5 CV AUC 0.7983 +/- 0.0045. Keep. Total printed time ~58.9s; new best but very close to timeout.

## Experiment 74

Hypothesis: More leaves may offset fewer rounds. Test 250 trees at learning rate 0.048 with 704 leaves against the current 275-tree/640-leaf best.
Type: follow-up
Change: n_estimators 275 -> 250, learning_rate 0.044 -> 0.048, max_leaves 640 -> 704.
Result: a79aed2 CV AUC 0.7984 +/- 0.0040. Keep. Total printed time ~56.0s; slight improvement.

## Experiment 75

Hypothesis: Revisit 768 leaves with more rounds than the failed 200-tree variant. 230 trees at learning rate 0.052 may avoid underfitting while using the larger leaf cap.
Type: follow-up
Change: n_estimators 250 -> 230, learning_rate 0.048 -> 0.052, max_leaves 704 -> 768.
Result: 6239b57 CV AUC 0.7976 +/- 0.0047. Discard. 768 leaves underperformed again.

## Experiment 76

Hypothesis: At fixed 704 leaves, a slightly smoother schedule may improve over 250/0.048. Try 260 trees at learning rate 0.0465.
Type: follow-up
Change: n_estimators 250 -> 260, learning_rate 0.048 -> 0.0465.
Result: 00d0b0f CV AUC 0.7989 +/- 0.0051. Keep. Total printed time ~57.9s; new best.

## Experiment 77

Hypothesis: Try a very small increase in smoothing at fixed 704 leaves. 265 trees at learning rate 0.0455 may improve if it remains under the timeout.
Type: follow-up
Change: n_estimators 260 -> 265, learning_rate 0.0465 -> 0.0455.
Result: d0aca0e timeout before CV AUC. Crash/timeout. Revert to 00d0b0f.

## Experiment 78

Hypothesis: The current 704-leaf model may prefer a slightly lower child weight than 12. Test `min_child_weight=10`.
Type: follow-up
Change: min_child_weight 12 -> 10.
Result: 76302ee timeout before CV AUC. Crash/timeout. Lower child weight is too expensive at this capacity.

## Experiment 79

Hypothesis: Test slightly stronger child weight at the current 704-leaf model. `min_child_weight=14` may regularize with lower runtime, without the underfit seen at 25.
Type: follow-up
Change: min_child_weight 12 -> 14.
Result: 7ce5a0d CV AUC 0.7973 +/- 0.0045. Discard. Stronger child weight reduced AUC.

## Research Before Experiment 80

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost DART tutorial: https://xgboost.readthedocs.io/en/release_2.0.0/tutorials/dart.html

Relevant notes:
- DART is a possible tree-dropout regularizer, but documentation notes prediction/training behavior is different and it tends to be slower, so it is risky under the timeout.
- `max_depth` can still constrain lossguide trees; using it with `max_leaves` may prevent very deep sparse paths.

## Experiment 80

Hypothesis: Unlimited-depth lossguide with 704 leaves may create very deep sparse paths. Set `max_depth=16` while keeping `max_leaves=704` to regularize and possibly speed up without changing the schedule.
Type: exploration
Change: max_depth 0 -> 16.
Result: 2a70be4 CV AUC 0.7968 +/- 0.0040. Discard. Depth cap hurt AUC.

## Research Before Experiment 81

Sources:
- XGBoost tree methods documentation: https://xgboost.readthedocs.io/en/stable/treemethod.html
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html

Relevant notes:
- `hist` uses binned features; `max_bin` trades split resolution for speed/memory.
- Earlier larger bins did not help. A smaller bin count may be worth testing now because the model is timeout-constrained.

## Experiment 81

Hypothesis: Lowering `max_bin` from the default to 128 may reduce runtime enough to help near-timeout high-leaf models, with limited AUC loss.
Type: exploration
Change: add `max_bin=128`.
Result: 55e5e9d CV AUC 0.7973 +/- 0.0044. Discard. AUC fell and runtime did not meaningfully improve.

## Experiment 82

Hypothesis: Retune tree-level column sampling for the current high-leaf model. `colsample_bytree=0.60` may regularize better than 0.65 at 704 leaves.
Type: follow-up
Change: colsample_bytree 0.65 -> 0.60.
Result: 3d49c0c timeout before CV AUC. Crash/timeout. Lower column sample did not finish under cap.

## Experiment 83

Hypothesis: Retune row subsampling for the high-leaf model. `subsample=0.95` may preserve more signal per tree while keeping slight stochasticity.
Type: follow-up
Change: subsample 0.90 -> 0.95.
Result: d50b33c timeout before CV AUC. Crash/timeout.

## Research Before Experiment 84

Sources:
- XGBoost parameters documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost scikit-learn estimator documentation: https://xgboost.readthedocs.io/en/release_3.2.0/python/sklearn_estimator.html

Relevant notes:
- With lossguide, `max_leaves` remains the main capacity control.
- Parallelism settings affect runtime rather than AUC, but the experiment loop advances on AUC, so runtime-only changes are not the right next step.

## Experiment 84

Hypothesis: Fine-tune around the current leaf-cap peak. `max_leaves=672` may beat 704 if 704 is slightly over capacity, while being cheaper.
Type: follow-up
Change: max_leaves 704 -> 672.
Result: 91ebdef CV AUC 0.7979 +/- 0.0039. Discard. Worse than 704 leaves.

## Experiment 85

Hypothesis: Test the upper side near 704 leaves without the failed 768 setup. 250 trees at learning rate 0.048 with 736 leaves may improve over 704/250 and avoid the 768 underperformance.
Type: follow-up
Change: n_estimators 260 -> 250, learning_rate 0.0465 -> 0.048, max_leaves 704 -> 736.
Result: 07012cc CV AUC 0.7987 +/- 0.0041. Discard. Close but below 704-leaf best.

## Experiment 86

Hypothesis: Interpolate the 704-leaf schedule between 250/0.048 and best 260/0.0465. Try 255 trees at learning rate 0.0472.
Type: follow-up
Change: n_estimators 260 -> 255, learning_rate 0.0465 -> 0.0472.
Result: e223999 CV AUC 0.7980 +/- 0.0045. Discard. Worse than 260/0.0465.

## Experiment 87

Hypothesis: `DepMinute` is redundant with `DepMinutes` and may add split noise/runtime. Remove `DepMinute` while keeping hour, absolute minutes, and cyclic encodings.
Type: ablation/simplification
Change: remove `DepMinute`.
Result: 540e89b timeout before CV AUC. Crash/timeout.

## Research Before Experiment 88

Sources:
- XGBoost scikit-learn estimator documentation: https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html
- XGBoost Python package introduction: https://xgboost.readthedocs.io/en/stable/python/python_intro.html

Relevant notes:
- Early stopping is available, but using it inside CV changes the number of trees by fold and is not a clean fit for this experiment loop.
- Continue with fixed-round CV and small schedule/capacity tests.

## Experiment 88

Hypothesis: The current 260-tree model may be slightly under-boosted. Raise learning rate from 0.0465 to 0.048 without changing tree count or leaf cap.
Type: follow-up
Change: learning_rate 0.0465 -> 0.048.
Result: d8044af CV AUC 0.7988 +/- 0.0040. Discard. Slightly worse than 0.0465.

## Experiment 89

Hypothesis: Test the lower learning-rate side at fixed 260 trees. `learning_rate=0.045` may generalize better if 0.0465 is slightly too aggressive.
Type: follow-up
Change: learning_rate 0.0465 -> 0.045.
Result: ce5cf47 CV AUC 0.7983 +/- 0.0043. Discard. Worse than 0.0465.

## Experiment 90

Hypothesis: Fine-tune learning rate between best 0.0465 and worse 0.048. Try 0.047 at the same 260-tree/704-leaf setup.
Type: follow-up
Change: learning_rate 0.0465 -> 0.047.
Result: 3db368e CV AUC 0.7979 +/- 0.0041. Discard. Worse than 0.0465.
