# Research Log

## Baseline

- Commit: `7d0b368`
- Result: `CV AUC 0.7445 ± 0.0043`
- Note: the system Python lacked `pandas`, but the project `.venv` worked.
- Status: baseline established.

## Experiment 1

- Hypothesis: the baseline was underfit / under-regularized, so a larger boosting budget with shallower trees and subsampling would improve AUC.
- Change: `n_estimators=250`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=2.0`, `reg_alpha=0.1`, `gamma=0.1`, `tree_method="hist"`.
- Result: `CV AUC 0.7494 ± 0.0048`.
- Status: keep.

## Experiment 2

- Hypothesis: raw time fields hide periodic structure, so sine/cosine encodings for month, day of week, and departure time should make that pattern easier for XGBoost to learn.
- Change: added cyclical features for `Month`, `DayOfWeek`, and `DepTime` (minute-of-day), while keeping the categorical originals.
- Source: feature-engine cyclical feature guide; periodic variables are often better represented with sine/cosine pairs.
- Result: `CV AUC 0.7495 ± 0.0050`.
- Status: keep.

## Experiment 3

- Hypothesis: the fixed boosting budget was still suboptimal; letting each fold choose its own best iteration with early stopping should reduce overfitting and improve generalization.
- Change: switched from sklearn CV to native `xgb.train()` with `early_stopping_rounds=50`, `num_boost_round=2000`, and fold-specific `best_iteration` tracking.
- Result: `CV AUC 0.7555 ± 0.0058`.
- Status: keep.

## Experiment 4

- Hypothesis: the depthwise tree builder was still leaving signal on the table; a loss-guided growth policy with bounded leaf count should fit the more useful splits first.
- Change: switched to `grow_policy="lossguide"`, set `max_leaves=64`, and relaxed `max_depth` to `0` so leaf budget drives complexity.
- Result: `CV AUC 0.7685 ± 0.0053`.
- Status: keep.

## Experiment 5

- Hypothesis: the raw `DepTime` HHMM code is a poor ordinal input and may be redundant with the cyclic encoding, so removing it should simplify the model without losing signal.
- Change: dropped raw `DepTime` from the feature set and kept only `Distance` plus the cyclic time features.
- Result: `CV AUC 0.7692 ± 0.0051`.
- Status: keep.

## Experiment 6

- Hypothesis: raw Month and DayOfWeek categories may be redundant once cyclic encodings are present, so removing them could simplify the model.
- Change: dropped the raw Month and DayOfWeek categorical columns.
- Result: `CV AUC 0.7690 ± 0.0052`.
- Status: discard.

## Experiment 7

- Hypothesis: low-cardinality categorical fields may work better if XGBoost is allowed to use one-hot splits for them instead of only partition-based splits.
- Change: raised `max_cat_to_onehot` to 32 while keeping the lossguide + early-stopping setup.
- Result: `CV AUC 0.7819 ± 0.0052`.
- Status: keep.

## Experiment 8

- Hypothesis: the lossguide model might be slightly over-capacity, so a smaller leaf budget could preserve most of the gain while reducing runtime.
- Change: reduced `max_leaves` from 64 to 48.
- Result: `CV AUC 0.7781 ± 0.0054`.
- Status: discard.

## Experiment 9

- Hypothesis: additional low-cardinality categories might benefit from one-hot splitting, so raising `max_cat_to_onehot` further could help.
- Change: raised `max_cat_to_onehot` from 32 to 64.
- Result: `CV AUC 0.7819 ± 0.0052`.
- Status: discard (no gain, slower).

## Experiment 10

- Hypothesis: allowing partition-based categorical splits to consider more categories might help the high-cardinality location fields.
- Change: raised `max_cat_threshold` to 128.
- Result: `CV AUC 0.7812 ± 0.0054`.
- Status: discard.

## Experiment 11

- Hypothesis: adding per-node column subsampling could reduce overfitting in the deeper lossguide trees.
- Change: set `colsample_bynode=0.8`.
- Result: `CV AUC 0.7816 ± 0.0054`.
- Status: discard.

## Experiment 12

- Hypothesis: the lossguide trees might still be too conservative, so lowering `min_child_weight` could uncover useful splits.
- Change: lowered `min_child_weight` from 5 to 3.
- Result: `CV AUC 0.7793 ± 0.0049`.
- Status: discard.

## Experiment 13

- Hypothesis: explicit route and carrier-route interactions might help capture airport-pair and carrier-specific effects.
- Change: added `Route` and `CarrierRoute` categorical features.
- Result: timed out before completion.
- Status: crash.

## Experiment 14

- Hypothesis: the current learning rate was a bit too small, so a slightly more aggressive boosting schedule might recover AUC while staying faster.
- Change: increased `eta` to 0.05, reduced `num_boost_round` to 1200, and shortened early stopping to 30 rounds.
- Result: `CV AUC 0.7801 ± 0.0056`.
- Status: discard.

## Experiment 15

- Hypothesis: a middle-ground learning rate and boosting budget might recover the best AUC while still being faster than the baseline schedule.
- Change: set `eta=0.04`, `num_boost_round=1500`, and `early_stopping_rounds=40`.
- Result: `CV AUC 0.7810 ± 0.0050`.
- Status: discard.

## Experiment 16

- Hypothesis: forcing trees to keep time and location interactions separate would reduce spurious splits and improve generalization.
- Change: added feature interaction constraints for time-related and location-related groups.
- Result: `CV AUC 0.7465 ± 0.0050`.
- Status: discard.

## Experiment 17

- Hypothesis: finer histogram binning should let the lossguide trees find more precise splits on the numeric and cyclical features.
- Change: increased `max_bin` to 512.
- Result: `CV AUC 0.7827 ± 0.0053`.
- Status: keep.

## Experiment 18

- Hypothesis: even finer histogram bins may continue to help the numeric and cyclical features.
- Change: increased `max_bin` from 512 to 1024.
- Result: `CV AUC 0.7834 ± 0.0051`.
- Status: keep.

## Experiment 19

- Hypothesis: finer histograms might benefit from a larger leaf budget.
- Change: increased `max_bin` from 1024 to 2048.
- Result: `CV AUC 0.7834 ± 0.0049`.
- Status: discard (no gain, slower).

## Experiment 20

- Hypothesis: a larger leaf budget might let the stronger histogram resolution find more useful splits.
- Change: increased `max_leaves` from 64 to 96.
- Result: timed out before completion.
- Status: crash.

## Experiment 21

- Hypothesis: a moderately larger leaf budget could help the finer histogram splits without tipping into the timeout zone.
- Change: increased `max_leaves` from 64 to 80.
- Result: `CV AUC 0.7873 ± 0.0052`.
- Status: keep.

## Experiment 22

- Hypothesis: a slightly larger leaf budget might still improve AUC if it stays under the timeout.
- Change: increased `max_leaves` from 80 to 88.
- Result: timed out before completion.
- Status: crash.

## Experiment 23

- Hypothesis: a slightly larger leaf budget might still help if it fits within the runtime envelope.
- Change: increased `max_leaves` from 80 to 84.
- Result: timed out before completion.
- Status: crash.

## Experiment 24

- Hypothesis: a wider one-hot threshold might help the categorical splits, especially with the larger leaf budget.
- Change: raised `max_cat_to_onehot` from 32 to 48.
- Result: `CV AUC 0.7873 ± 0.0052`.
- Status: discard (no gain, slower).

## Experiment 25

- Hypothesis: a stronger split penalty could regularize the larger model and improve generalization.
- Change: raised `gamma` from 0.1 to 0.2.
- Result: `CV AUC 0.7873 ± 0.0052`.
- Status: discard.

## Experiment 26

- Hypothesis: slightly less aggressive row/column subsampling might let the larger model fit a bit more signal.
- Change: increased `subsample` and `colsample_bytree` to 0.9.
- Result: `CV AUC 0.7869 ± 0.0049`.
- Status: discard.

## Experiment 27

- Hypothesis: an intermediate subsampling rate might recover some of the lost score.
- Change: increased `subsample` and `colsample_bytree` to 0.85.
- Result: `CV AUC 0.7861 ± 0.0054`.
- Status: discard.

## Experiment 28

- Hypothesis: the larger tree can likely tolerate less L2/L1 regularization.
- Change: lowered `lambda` to 1.0 and `alpha` to 0.0.
- Result: `CV AUC 0.7874 ± 0.0043`.
- Status: keep.

## Experiment 29

- Hypothesis: a bit less L2 regularization than 1.0 could still help.
- Change: lowered `lambda` from 1.0 to 0.5.
- Result: `CV AUC 0.7873 ± 0.0049`.
- Status: discard.

## Experiment 30

- Hypothesis: a slightly less conservative split threshold could help the larger model capture useful structure.
- Change: lowered `min_child_weight` from 5 to 4.
- Result: `CV AUC 0.7855 ± 0.0052`.
- Status: discard.

## Experiment 31

- Hypothesis: a slightly more aggressive learning rate might still be competitive if paired with a longer schedule.
- Change: set `eta=0.035`, `num_boost_round=1700`, and `early_stopping_rounds=40`.
- Result: `CV AUC 0.7870 ± 0.0053`.
- Status: discard.

## Experiment 32

- Hypothesis: a midpoint learning rate might be the sweet spot between 0.03 and 0.035.
- Change: set `eta=0.032`, `num_boost_round=1850`, and `early_stopping_rounds=45`.
- Result: `CV AUC 0.7873 ± 0.0050`.
- Status: discard.

## Experiment 33

- Hypothesis: a tiny amount of L1 regularization might help now that the model has more capacity.
- Change: set `alpha` to 0.02.
- Result: `CV AUC 0.7873 ± 0.0048`.
- Status: discard.

## Experiment 34

- Hypothesis: allowing slightly more splits could help the larger tree use its capacity.
- Change: lowered `gamma` from 0.1 to 0.0.
- Result: `CV AUC 0.7874 ± 0.0043`.
- Status: discard.

## Experiment 35

- Hypothesis: the best leaf budget may be a bit above 80, but below the timeout threshold.
- Change: increased `max_leaves` from 80 to 82.
- Result: `CV AUC 0.7878 ± 0.0046`.
- Status: keep.

## Experiment 36

- Hypothesis: one more leaf might still help if the runtime stays within bounds.
- Change: increased `max_leaves` from 82 to 83.
- Result: `CV AUC 0.7877 ± 0.0049`.
- Status: discard.

## Experiment 37

- Hypothesis: more histogram bins beyond 1024 might let the stronger leaf budget find a few more useful splits.
- Change: increased `max_bin` from 1024 to 1536.
- Result: `CV AUC 0.7874 ± 0.0052`.
- Status: discard.

## Experiment 38

- Hypothesis: dropout-style boosting regularization might improve generalization.
- Change: switched to the DART booster with dropout parameters.
- Result: timed out before completion.
- Status: crash.

## Experiment 39

- Hypothesis: a deeper depthwise tree could capture interactions without the runtime cost of very large leaf budgets.
- Change: switched from lossguide to depthwise with `max_depth=8`.
- Result: `CV AUC 0.7877 ± 0.0047`.
- Status: discard.

## Experiment 40

- Hypothesis: one more level of depthwise capacity could capture stronger interactions without blowing up runtime.
- Change: increased `max_depth` from 8 to 9.
- Result: `CV AUC 0.7926 ± 0.0047`.
- Status: keep.

## Experiment 41

- Hypothesis: a deeper depthwise tree may still extract additional interaction signal.
- Change: increased `max_depth` from 9 to 10.
- Result: `CV AUC 0.7955 ± 0.0030`.
- Status: keep.

## Experiment 42

- Hypothesis: another level of depth could still add useful interaction capacity.
- Change: increased `max_depth` from 10 to 11.
- Result: `CV AUC 0.7987 ± 0.0041`.
- Status: keep.

## Experiment 43

- Hypothesis: an even deeper tree could still help, but the schedule may be too long for the runtime budget.
- Change: increased `max_depth` from 11 to 12.
- Result: timed out before completion.
- Status: crash.

## Experiment 44

- Hypothesis: the deeper tree may work if the boosting schedule is trimmed to fit the runtime budget.
- Change: kept `max_depth=12` but reduced `num_boost_round` to 1200 and `early_stopping_rounds` to 30.
- Result: `CV AUC 0.7978 ± 0.0052`.
- Status: keep.

## Experiment 45

- Hypothesis: one more depth level may still add useful interactions, especially with the shorter schedule.
- Change: increased `max_depth` from 12 to 13.
- Result: `CV AUC 0.7999 ± 0.0049`.
- Status: keep.

## Experiment 46

- Hypothesis: another level of depth may still help a little without hurting runtime too badly.
- Change: increased `max_depth` from 13 to 14.
- Result: `CV AUC 0.8002 ± 0.0037`.
- Status: keep.

## Experiment 47

- Hypothesis: the deeper tree may still have some headroom.
- Change: increased `max_depth` from 14 to 15.
- Result: `CV AUC 0.8025 ± 0.0036`.
- Status: keep.

## Experiment 48

- Hypothesis: another depth increment may still squeeze out a tiny gain.
- Change: increased `max_depth` from 15 to 16.
- Result: `CV AUC 0.8027 ± 0.0042`.
- Status: keep.

## Experiment 49

- Hypothesis: one more depth level may still help a bit.
- Change: increased `max_depth` from 16 to 17.
- Result: `CV AUC 0.8034 ± 0.0033`.
- Status: keep.

## Experiment 50

- Hypothesis: another depth increment could still produce a small gain.
- Change: increased `max_depth` from 17 to 18.
- Result: `CV AUC 0.8041 ± 0.0036`.
- Status: keep.

## Experiment 51

- Hypothesis: one more depth step might still eke out a gain.
- Change: increased `max_depth` from 18 to 19.
- Result: `CV AUC 0.8041 ± 0.0044`.
- Status: discard.

## Experiment 52

- Hypothesis: the deeper tree may need a slightly smaller step size to generalize best.
- Change: increased `max_depth` from 18 to 20.
- Result: `CV AUC 0.8039 ± 0.0042`.
- Status: discard.

## Experiment 53

- Hypothesis: the deeper tree might need a smaller step size, but that could push runtime over the limit.
- Change: set `eta=0.028`, `num_boost_round=2200`, and `early_stopping_rounds=45` at `max_depth=18`.
- Result: timed out before completion.
- Status: crash.

## Experiment 54

- Hypothesis: finer histogram bins may help more on the deeper tree.
- Change: increased `max_bin` from 1024 to 2048 at `max_depth=18`.
- Result: `CV AUC 0.8037 ± 0.0044`.
- Status: discard.

## Experiment 55

- Hypothesis: the deeper tree might still benefit from a little more split freedom.
- Change: lowered `gamma` from 0.1 to 0.0 at `max_depth=18`.
- Result: `CV AUC 0.8038 ± 0.0038`.
- Status: discard.

## Experiment 56

- Hypothesis: the deeper model may still benefit from a little more boosting budget.
- Change: increased `num_boost_round` to 1500 and `early_stopping_rounds` to 40 at `max_depth=18`.
- Result: `CV AUC 0.8042 ± 0.0038`.
- Status: keep.

## Experiment 57

- Hypothesis: pushing the boosting budget too far may be slowing training without helping score.
- Change: increased `num_boost_round` to 1800 and `early_stopping_rounds` to 50 at `max_depth=18`.
- Result: timed out before completion.
- Status: crash.

## Experiment 58

- Hypothesis: a slightly longer schedule might still help, but the added runtime may not be worth it.
- Change: increased `num_boost_round` to 1600 and `early_stopping_rounds` to 45 at `max_depth=18`.
- Result: `CV AUC 0.8042 ± 0.0038`.
- Status: discard (same score, slower).

## Experiment 59

- Hypothesis: a midpoint gamma could be the sweet spot for the deeper tree.
- Change: lowered `gamma` from 0.1 to 0.05 at `max_depth=18`.
- Result: `CV AUC 0.8038 ± 0.0040`.
- Status: discard.

## Experiment 60

- Hypothesis: the deeper depthwise tree might need a bit more boosting budget, but `max_depth=19` may be too slow.
- Change: increased `max_depth` from 18 to 19 with the longer `1500/40` schedule.
- Result: timed out before completion.
- Status: crash.

## Experiment 61

- Hypothesis: a slightly larger `min_child_weight` may help the deep tree, but the long schedule was too slow.
- Change: increased `min_child_weight` from 5 to 6 at `max_depth=18`.
- Result: timed out before completion.
- Status: crash.

## Experiment 62

- Hypothesis: a shorter schedule might make `min_child_weight=6` workable, but it likely needs to be checked.
- Change: kept `min_child_weight=6` and shortened to `num_boost_round=1200` with `early_stopping_rounds=30`.
- Result: `CV AUC 0.8035 ± 0.0042`.
- Status: discard.

## Experiment 63

- Hypothesis: a stronger L2 penalty may help, but the longer schedule could still exceed the runtime budget.
- Change: raised `lambda` to 1.5 at `max_depth=18`.
- Result: timed out before completion.
- Status: crash.

## Experiment 64

- Hypothesis: a stronger L2 penalty might still help if the schedule is shortened.
- Change: kept `lambda=1.5` and shortened to `num_boost_round=1200` with `early_stopping_rounds=30`.
- Result: `CV AUC 0.8031 ± 0.0041`.
- Status: discard.
