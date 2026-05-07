# Detailed Research Log After Experiment Batch

Branch: `may7`

Scope: baseline plus experiments 1-35 from `results.tsv` and `research-log.md`.

Current best commit: `a20ccac`

Current best CV AUC: `0.7875 +/- 0.0049`

Best final model configuration:

```python
xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=13,
    learning_rate=0.01,
    min_child_weight=12,
    subsample=0.85,
    colsample_bytree=0.55,
    reg_lambda=2.0,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
```

## Source Notes

I used the required web research at three decision points: before the first non-baseline experiment, after the first 10 experiments, and after 30 experiments. The searches were not used as recipes to copy blindly; they were used to choose which axes to test deliberately.

Sources consulted:

- XGBoost parameter documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost categorical data tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- XGBoost tree-method documentation: https://xgboost.readthedocs.io/en/release_1.7.0/treemethod.html
- Flight-delay prediction paper, Scientific Reports 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC10897135/
- Flight-delay ML review/paper result from search: https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/

Key source takeaways:

- `max_depth` increases model complexity and can overfit; the XGBoost docs also warn that very deep trees consume memory aggressively. That framed depth probes as useful but increasingly runtime-risky.
- `min_child_weight` is a conservative split control. Larger values require more hessian mass in child nodes, which can prevent small noisy leaves. That motivated bracketing `min_child_weight` after deeper trees started working.
- `gamma` requires a minimum loss reduction for a split. I treated it as a second, more direct split-pruning mechanism after finding a promising `min_child_weight`.
- `subsample` and `colsample_bytree` are regularization controls. The docs specifically describe row sampling as a way to prevent overfitting and column sampling as per-tree feature subsampling. This motivated row/column bracketing rather than only increasing tree capacity.
- `reg_lambda` is L2 leaf-weight regularization. I introduced mild extra L2 regularization in the first non-baseline run and then held it steady while identifying larger effects.
- The categorical tutorial explains native categorical handling, including one-hot versus partition-based categorical splits and the `max_cat_to_onehot` switch. This motivated the later categorical split experiment.
- The tree-method docs mention `grow_policy` and `max_leaves`; I considered these as possible future directions, but did not get to a committed lossguide/max-leaves experiment before this detailed log request.
- Flight-delay papers and search results repeatedly emphasize airline/carrier, origin, destination, flight time, day, and distance as important. That matched the available columns and motivated time-of-day and route-interaction feature engineering attempts.

## Dataset And Baseline Context

The training script uses `data-cache/2005-slice1-100k.csv` with 5-fold stratified cross-validation scored by ROC AUC. The available columns are:

- Categorical: `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`
- Numeric: `DepTime`, `Distance`
- Target: `dep_delayed_15min`

The baseline model was:

```python
n_estimators=30
max_depth=6
learning_rate=0.1
enable_categorical=True
```

The baseline already used pandas categoricals with fixed category levels from the training set. That is important because any feature engineering had to remain inside `prepare(df)` so later ground-truth evaluation could apply the same transformation.

Baseline result:

- Commit `7d0b368`
- CV AUC `0.7445 +/- 0.0043`
- Status `keep`

Interpretation: the baseline was likely underfit. Thirty trees at eta 0.1 was very small for a categorical interaction problem with carrier, airport, day, time, and distance effects.

## Block 1: Establish Capacity And Basic Regularization

Experiments 1-10 focused on the core XGBoost tuning axes from the parameter docs: number of trees, learning rate, depth, child-weight pruning, row sampling, column sampling, and L2 regularization.

The initial theory was that the baseline was underfit because it used only 30 trees. I did not start with feature engineering because the most direct and lower-risk change was to give the existing representation a better learner.

### Experiment 1: `4c2aa50`, Regularized 120-Tree Booster

Change:

- `n_estimators`: 30 -> 120
- `max_depth`: 6 -> 5
- `learning_rate`: 0.1 -> 0.05
- Added `min_child_weight=3`
- Added `subsample=0.85`
- Added `colsample_bytree=0.85`
- Added `reg_lambda=2.0`

Reasoning:

The XGBoost docs describe depth and child weight as complexity controls, and subsample/colsample as regularization. Tuning guides commonly pair a lower learning rate with more trees. I chose a conservative first move: more boosting rounds and smaller eta, but slightly shallower trees and several regularizers to reduce the chance that the improvement would be only overfitting.

Result:

- CV AUC `0.7497 +/- 0.0048`
- Status `keep`
- Delta vs previous best: `+0.0052`

Interpretation:

The large gain confirmed baseline underfit. The lower-rate, more-regularized model was better despite reducing depth from 6 to 5, which suggested boosting schedule and regularization mattered immediately.

### Experiment 2: `f37a992`, Longer Lower-Rate Boosting

Change:

- `n_estimators`: 120 -> 220
- `learning_rate`: 0.05 -> 0.035

Reasoning:

This was a direct follow-up to determine whether the previous gain came from smoother boosting. I kept depth and regularization fixed so the result would isolate the schedule change.

Result:

- CV AUC `0.7511 +/- 0.0047`
- Status `keep`
- Delta: `+0.0014`

Interpretation:

More lower-rate trees continued to help. The gain was smaller than experiment 1 but still real enough to keep. It established that the model still had useful learning capacity left under the conservative depth-5 setup.

### Experiment 3: `dbc1fa8`, Restore Depth 6

Change:

- `max_depth`: 5 -> 6

Reasoning:

With smoother boosting and regularization in place, I tested whether the original depth 6 was useful. The domain has natural interactions: carrier by airport, airport by time, distance by route, day by time. A tree of depth 6 can represent richer conjunctions than depth 5.

Result:

- CV AUC `0.7557 +/- 0.0046`
- Status `keep`
- Delta: `+0.0046`

Interpretation:

Depth mattered. The strong gain showed that the problem was not just underfit by too few trees; it needed richer interaction structure.

### Experiment 4: `9765533`, Depth 7

Change:

- `max_depth`: 6 -> 7

Reasoning:

Depth 6 had improved materially, so a one-step depth probe was justified. Sampling and `min_child_weight=3` remained in place to limit variance.

Result:

- CV AUC `0.7599 +/- 0.0047`
- Status `keep`
- Delta: `+0.0042`

Interpretation:

The positive depth trend continued. This strengthened the theory that high-order categorical interactions are central to this dataset.

### Experiment 5: `0e16eec`, Depth 8

Change:

- `max_depth`: 7 -> 8

Reasoning:

Another controlled depth probe. At this point I expected the gain might flatten, but the previous two depth steps were too strong to stop early.

Result:

- CV AUC `0.7643 +/- 0.0048`
- Status `keep`
- Delta: `+0.0044`

Interpretation:

Depth 8 again improved substantially. The model was still not over the useful interaction-depth boundary.

### Experiment 6: `a0f8e7c`, Depth 9

Change:

- `max_depth`: 8 -> 9

Reasoning:

Depth improvements were monotonic and meaningful. I continued the depth probe while watching fold variance and runtime.

Result:

- CV AUC `0.7676 +/- 0.0052`
- Status `keep`
- Delta: `+0.0033`

Interpretation:

Depth 9 improved, but the fold standard deviation increased to `0.0052`. That was the first warning that variance might be rising. Still, the mean gain was large enough to keep.

### Experiment 7: `d723a6b`, Depth 10

Change:

- `max_depth`: 9 -> 10

Reasoning:

Depth 9 was still better. Depth 10 was a boundary test to see whether the upward trend had one more useful step before regularization needed retuning.

Result:

- CV AUC `0.7718 +/- 0.0048`
- Status `keep`
- Delta: `+0.0042`

Interpretation:

Depth 10 improved strongly and fold spread did not worsen. This made depth 10 the new base for regularization tuning.

### Experiment 8: `414cbb1`, Lower `min_child_weight` To 1

Change:

- `min_child_weight`: 3 -> 1

Reasoning:

Since deeper interactions were useful, I tested whether the child-weight constraint was blocking valuable smaller leaves. The XGBoost docs define `min_child_weight` as the minimum child hessian needed to continue splitting; lower values permit finer partitions.

Result:

- CV AUC `0.7675 +/- 0.0044`
- Status `discard`
- Delta vs best: `-0.0043`

Interpretation:

This was a clear failure. The deeper model needed pruning, not looser leaf creation. The likely issue was noisy small categorical leaves.

### Experiment 9: `6097c86`, Raise `min_child_weight` To 5

Change:

- Reset to depth-10 best.
- `min_child_weight`: 3 -> 5

Reasoning:

Since `min_child_weight=1` lost badly, I tested the opposite direction. Larger child weight should suppress weak deep leaves while retaining the depth-10 interaction structure.

Result:

- CV AUC `0.7738 +/- 0.0047`
- Status `keep`
- Delta: `+0.0020`

Interpretation:

More pruning helped. This validated the theory that deep trees are useful only when constrained enough to avoid tiny noisy leaves.

### Experiment 10: `16ca59b`, Raise `min_child_weight` To 8

Change:

- `min_child_weight`: 5 -> 8

Reasoning:

Because `min_child_weight=5` improved over 3, I continued bracketing. The goal was to find the pruning point where useful interactions remained but noisy leaves were removed.

Result:

- CV AUC `0.7751 +/- 0.0050`
- Status `keep`
- Delta: `+0.0013`

Interpretation:

Stronger pruning still helped. Gains were smaller than depth changes, but still useful.

### Synthesis After 10 Experiments

What worked:

- More trees with lower learning rate.
- Increasing depth from 5 through 10.
- Increasing `min_child_weight` after reaching deep trees.

What failed:

- Lowering `min_child_weight` to 1.

Working theory:

The data has strong high-order interactions among carrier, airport, date, time, and distance. XGBoost can exploit those interactions, but only with enough depth and enough pruning. Deep trees without pruning create noisy leaves; shallow trees underfit.

## Block 2: Regularization Bracketing At Depth 10

Experiments 11-20 refined pruning and sampling around the depth-10 model. This block was guided by XGBoost docs on `min_child_weight`, `gamma`, `subsample`, and `colsample_bytree`.

### Experiment 11: `abbf167`, `min_child_weight=12`

Change:

- `min_child_weight`: 8 -> 12

Reasoning:

Since child-weight pruning helped through 8, I tested a moderately stronger value. I expected smaller returns and a possible underfit boundary.

Result:

- CV AUC `0.7757 +/- 0.0048`
- Status `keep`
- Delta: `+0.0006`

Interpretation:

Still a small improvement. The useful pruning point was above 8, but gains were flattening.

### Experiment 12: `b14ad5b`, `min_child_weight=20`

Change:

- `min_child_weight`: 12 -> 20

Reasoning:

The gains were flattening, so I used a wider step to find the underfit side of the bracket.

Result:

- CV AUC `0.7749 +/- 0.0049`
- Status `discard`
- Delta vs best: `-0.0008`

Interpretation:

This crossed the over-pruning boundary. At depth 10, `min_child_weight=12` was better than 20.

### Experiment 13: `69a4b9a`, `gamma=0.05`

Change:

- Added `gamma=0.05`

Reasoning:

`gamma` requires a minimum loss reduction for a split, so it is another split-pruning control. I tried a small value because child-weight already performed most pruning and I did not want to blunt the model heavily.

Result:

- CV AUC `0.7755 +/- 0.0051`
- Status `discard`
- Delta vs best: `-0.0002`

Interpretation:

No useful gain. The child-weight setting was already doing the useful pruning, and adding `gamma` slightly reduced mean AUC.

### Experiment 14: `8f590da`, Full Row Sampling

Change:

- `subsample`: 0.85 -> 1.0

Reasoning:

After stronger leaf pruning, I tested whether row stochasticity was still needed. Using all rows can improve split estimates if overfitting is controlled elsewhere.

Result:

- CV AUC `0.7752 +/- 0.0060`
- Status `discard`
- Delta vs best: `-0.0005`

Interpretation:

Full rows did not help and increased fold spread. Row stochasticity remained useful.

### Experiment 15: `4e236f2`, Stronger Row Sampling

Change:

- `subsample`: 0.85 -> 0.75

Reasoning:

Since full rows were worse, I tested whether more row sampling would further regularize deep trees.

Result:

- CV AUC `0.7751 +/- 0.0050`
- Status `discard`
- Delta vs best: `-0.0006`

Interpretation:

Too much row sampling also lost. `subsample=0.85` looked like a good local point.

### Experiment 16: `4c4d0ae`, Full Column Sampling

Change:

- `colsample_bytree`: 0.85 -> 1.0

Reasoning:

The feature set is small, so I considered whether withholding columns was unnecessarily starving split decisions. This was a direct test of whether column sampling was helpful or harmful.

Result:

- CV AUC `0.7712 +/- 0.0047`
- Status `discard`
- Delta vs best: `-0.0045`

Interpretation:

Using all columns was much worse. Column stochasticity was important regularization for this model.

### Experiment 17: `d1daf26`, `colsample_bytree=0.70`

Change:

- `colsample_bytree`: 0.85 -> 0.70

Reasoning:

Since full columns hurt badly, I tested stronger feature sampling. With deep trees, different trees seeing different feature subsets may reduce over-reliance on dominant categorical splits.

Result:

- CV AUC `0.7776 +/- 0.0054`
- Status `keep`
- Delta: `+0.0019`

Interpretation:

Stronger column sampling helped. Fold spread rose slightly, but the mean improvement was meaningful.

### Experiment 18: `8ed423e`, `colsample_bytree=0.55`

Change:

- `colsample_bytree`: 0.70 -> 0.55

Reasoning:

The trend favored more column randomness, so I pushed further while staying above an obviously tiny feature subset.

Result:

- CV AUC `0.7791 +/- 0.0050`
- Status `keep`
- Delta: `+0.0015`

Interpretation:

This became the new best. It was an important result: even with only eight raw features, the model benefited from substantial per-tree feature subsampling.

### Experiment 19: `1cf4f3f`, `colsample_bytree=0.40`

Change:

- `colsample_bytree`: 0.55 -> 0.40

Reasoning:

This was an aggressive boundary test. If 0.55 was good, maybe 0.40 would regularize further; if not, it would identify the starvation point.

Result:

- CV AUC `0.7774 +/- 0.0050`
- Status `discard`
- Delta vs best: `-0.0017`

Interpretation:

Too aggressive. Important features were probably unavailable too often. The local optimum was above 0.40.

### Experiment 20: `69d863f`, `colsample_bytree=0.50`

Change:

- `colsample_bytree`: 0.55 -> 0.50

Reasoning:

This refined between the kept 0.55 and discarded 0.40.

Result:

- CV AUC `0.7791 +/- 0.0050`
- Status `discard`
- Delta vs best: tied at four decimals

Interpretation:

The experiment tied the best to the reported precision, so by the experiment rule it was discarded. There was no reason to prefer it over 0.55.

### Synthesis After 20 Experiments

What worked:

- `min_child_weight=12` at depth 10.
- `subsample=0.85` retained row stochasticity without starving the model.
- `colsample_bytree=0.55` strongly improved over both 0.85 and 1.0.

What failed:

- `min_child_weight=20` over-pruned.
- `gamma=0.05` was redundant or harmful.
- `subsample=1.0` and `subsample=0.75` both lost.
- `colsample_bytree=1.0` lost badly; `0.40` was too aggressive.

Working theory:

Deep trees need two forms of regularization here: leaf pruning and feature stochasticity. Row sampling helps, but the optimum is moderate. Column sampling is surprisingly important, likely because otherwise trees repeatedly exploit the same high-cardinality categorical splits too aggressively.

## Block 3: Feature Engineering Attempts

Experiments 21-23 tested feature engineering ideas motivated by flight-delay literature and the available columns. All feature engineering was done inside `prepare(df)` as required.

The domain sources emphasized flight time, airline/carrier, origin, destination, distance, and day. Since carrier/airport/day/distance already existed, the most natural additions were:

- better representation of departure time,
- a direct route interaction.

### Experiment 21: `91942f9`, Numeric Departure Time Features

Change:

Inside `prepare(df)`, added:

- `DepHour`
- `DepMinutes`

Reasoning:

Raw `DepTime` is HHMM encoded, which is not a true continuous scale: 0959 to 1000 is a one-minute difference but numerically jumps by 41. Adding parsed hour/minutes could make time-of-day splits cleaner.

Result:

- CV AUC `0.7771 +/- 0.0053`
- Status `discard`
- Delta vs best: `-0.0020`

Interpretation:

The derived numeric time features hurt. Possible reasons:

- Raw `DepTime` already gave useful threshold splits.
- The extra columns created redundant split choices, especially with column sampling.
- `DepMinutes` may have introduced a different ordinal representation without adding enough signal.

### Experiment 22: `8600faf`, Categorical Departure Hour

Change:

Inside `prepare(df)`, added categorical `DepHour` with categories `h-0` through `h-23`.

Reasoning:

The numeric time features hurt, but a categorical hour might capture time blocks without imposing linear or HHMM ordinality.

Result:

- CV AUC `0.7776 +/- 0.0056`
- Status `discard`
- Delta vs best: `-0.0015`

Interpretation:

Categorical hour was still worse than raw `DepTime` only. It likely added a redundant categorical feature and worsened split competition under column sampling.

### Experiment 23: `04ceb60`, Route Interaction Category

Change:

Inside `prepare(df)`, added categorical `Route = Origin + "_" + Dest`.

Reasoning:

Flight-delay literature and intuition suggest route effects. A direct `Origin_Dest` category might capture route-specific congestion or operational patterns better than separate origin and destination splits. I avoided count/target statistics because the instructions warned against count-derived features on the balanced sample and because target encoding would risk leakage unless carefully nested inside CV.

Result:

- CV AUC `0.7756 +/- 0.0050`
- Status `discard`
- Delta vs best: `-0.0035`
- Runtime also increased.

Interpretation:

The high-cardinality route feature hurt. Separate `Origin` and `Dest` categories were better for this setup. The route interaction probably created sparse categories and noisy splits, and it slowed training.

### Feature Engineering Synthesis

What worked:

- None of the attempted FE improved CV AUC.

What failed:

- Parsed numeric departure time.
- Categorical departure hour.
- High-cardinality route interaction.

Working theory:

XGBoost's deep trees were already able to build useful interactions from raw features. Naive derived features mainly added redundant or sparse split candidates. In a model with strong column sampling, extra redundant features can also change which feature subsets each tree sees, potentially making trees less reliable.

## Block 4: Boosting Schedule Revisited

Experiments 24-28 returned to HPO after feature engineering failed. The working model now had stronger pruning and column sampling, so it was plausible that a longer, lower-rate boosting schedule could help.

### Experiment 24: `92f6aee`, 400 Trees At Eta 0.02

Change:

- `n_estimators`: 220 -> 400
- `learning_rate`: 0.035 -> 0.02

Reasoning:

Earlier, lower-rate boosting helped. After tuning depth/pruning/sampling, I revisited the schedule to see whether smoother additive updates would improve ranking.

Result:

- CV AUC `0.7794 +/- 0.0049`
- Status `keep`
- Delta: `+0.0003`

Interpretation:

Small gain. It was worth keeping, but the effect was much smaller than depth/colsample changes.

### Experiment 25: `81b0471`, 600 Trees At Eta 0.015

Change:

- `n_estimators`: 400 -> 600
- `learning_rate`: 0.02 -> 0.015

Reasoning:

The 400-tree schedule improved slightly, so I continued the smooth-boosting trend.

Result:

- CV AUC `0.7802 +/- 0.0051`
- Status `keep`
- Delta: `+0.0008`

Interpretation:

Another small but real gain. Runtime was still acceptable.

### Experiment 26: `7338924`, 800 Trees At Eta 0.012

Change:

- `n_estimators`: 600 -> 800
- `learning_rate`: 0.015 -> 0.012

Reasoning:

The schedule trend was still positive. I expected diminishing returns but wanted to locate the boundary.

Result:

- CV AUC `0.7813 +/- 0.0050`
- Status `keep`
- Delta: `+0.0011`

Interpretation:

The 800-tree schedule was a meaningful improvement. CV time was `24.2s`, still comfortably inside the one-minute budget.

### Experiment 27: `1b7f12e`, 1000 Trees At Eta 0.01

Change:

- `n_estimators`: 800 -> 1000
- `learning_rate`: 0.012 -> 0.01

Reasoning:

This was the natural continuation of the schedule trend. I expected this to be near the practical limit because runtime was increasing.

Result:

- CV AUC `0.7817 +/- 0.0049`
- Status `keep`
- Delta: `+0.0004`
- CV time `30.2s`

Interpretation:

The result improved slightly. Runtime was now significant but still below the one-minute cap.

### Experiment 28: `191ca6d`, 1200 Trees At Eta 0.008

Change:

- `n_estimators`: 1000 -> 1200
- `learning_rate`: 0.01 -> 0.008

Reasoning:

This tested whether the smoother schedule had any headroom beyond 1000 trees.

Result:

- CV AUC `0.7816 +/- 0.0050`
- Status `discard`
- Delta vs best: `-0.0001`
- CV time `36.1s`

Interpretation:

No improvement and slower. The schedule optimum was around 1000 trees at eta 0.01.

### Schedule Synthesis

What worked:

- Lower learning rate with more trees from 400 through 1000.

What failed:

- Extending to 1200 trees.

Working theory:

The model benefits from smoother boosting up to around 1000 trees, but beyond that the extra rounds mostly add runtime and no measurable AUC.

## Block 5: Depth Reopened Under Better Schedule

Experiments 29-35 tested whether the stronger 1000-tree schedule reopened the depth axis.

### Experiment 29: `92276bc`, Depth 11

Change:

- `max_depth`: 10 -> 11

Reasoning:

After longer lower-rate boosting, stronger child-weight, and stronger column sampling, deeper trees might generalize better than before. This revisited depth after changing the broader training regime.

Result:

- CV AUC `0.7843 +/- 0.0050`
- Status `keep`
- Delta: `+0.0026`
- CV time `35.0s`

Interpretation:

Depth 11 was a substantial improvement. The depth axis was still productive under the tuned schedule.

### Experiment 30: `32281af`, Depth 12

Change:

- `max_depth`: 11 -> 12

Reasoning:

Depth 11 improved materially, so depth 12 was a justified boundary test. Runtime risk was rising.

Result:

- CV AUC `0.7860 +/- 0.0049`
- Status `keep`
- Delta: `+0.0017`
- CV time `39.3s`

Interpretation:

Depth 12 improved again. This confirmed that the problem was still interaction-limited even after many prior depth increases.

### Experiment 31: `a20ccac`, Depth 13

Change:

- `max_depth`: 12 -> 13

Reasoning:

Depth 12 improved, so depth 13 tested the next interaction boundary. The XGBoost docs warn deep trees can consume memory aggressively, so I watched runtime closely.

Result:

- CV AUC `0.7875 +/- 0.0049`
- Status `keep`
- Delta: `+0.0015`
- CV time `43.0s`

Interpretation:

Depth 13 improved and stayed within the timeout. This became the current best. Runtime is close enough to the cap that larger models should be treated carefully.

### Experiment 32: `757d2a4`, Depth 14

Change:

- `max_depth`: 13 -> 14

Reasoning:

Depth 13 improved, so depth 14 was the next boundary. I ran it under an explicit 60-second timeout because of the runtime trend.

Result:

- Timed out after 60s.
- Status `crash`
- Logged AUC `0.0000`

Interpretation:

Depth 14 was not viable under the experiment budget. Even if it might improve AUC eventually, it violated the runtime constraint.

### Experiment 33: `deb23fa`, `min_child_weight=18` At Depth 13

Change:

- Reset to depth-13 best.
- `min_child_weight`: 12 -> 18

Reasoning:

Since depth 14 timed out and depth 13 was expensive, stronger pruning at depth 13 might reduce noisy leaves and speed up training while preserving interaction capacity.

Result:

- CV AUC `0.7849 +/- 0.0050`
- Status `discard`
- Delta vs best: `-0.0026`
- CV time `37.3s`

Interpretation:

This over-pruned. It was faster, but it gave up too much signal. At depth 13, `min_child_weight=12` remained better.

### Experiment 34: `020d67e`, `min_child_weight=8` At Depth 13

Change:

- `min_child_weight`: 12 -> 8

Reasoning:

Since 18 over-pruned, I tested whether less pruning would recover smaller useful leaves at depth 13.

Result:

- Timed out after 60s.
- Status `crash`
- Logged AUC `0.0000`

Interpretation:

Looser pruning was too expensive at depth 13. This brackets the current `min_child_weight=12` from both sides: 18 over-prunes, 8 times out.

### Experiment 35: `2d4bca0`, `max_cat_to_onehot=32`

Change:

- Added `max_cat_to_onehot=32`

Reasoning:

The XGBoost categorical tutorial says `max_cat_to_onehot` controls whether categorical features use one-hot split style or partitioning. I tested whether moderate-cardinality features such as carrier/day/month might benefit from one-hot categorical splits while high-cardinality airport fields remained partitioned.

Result:

- CV AUC `0.7833 +/- 0.0050`
- Status `discard`
- Delta vs best: `-0.0042`
- CV time `31.8s`

Interpretation:

The default categorical strategy was better. For this dataset/model, pushing more categoricals into one-hot split handling degraded AUC.

## Overall Findings

Best result path:

1. Baseline: `0.7445`
2. More lower-rate trees and basic regularization: `0.7497`
3. Longer lower-rate schedule: `0.7511`
4. Depth 6-10: `0.7718`
5. Child-weight pruning to 12: `0.7757`
6. Column sampling to 0.55: `0.7791`
7. 1000-tree eta 0.01 schedule: `0.7817`
8. Depth 11-13: `0.7875`

The dominant pattern:

- The model wants deep trees.
- Deep trees only work with meaningful pruning and stochasticity.
- Column sampling was more important than expected.
- Feature engineering attempts were negative.
- Runtime becomes the binding constraint at depth 14 or looser pruning at depth 13.

## What Worked

### Deeper Trees

Depth improved repeatedly:

- 5 -> 6: `0.7511` to `0.7557`
- 6 -> 7: `0.7557` to `0.7599`
- 7 -> 8: `0.7599` to `0.7643`
- 8 -> 9: `0.7643` to `0.7676`
- 9 -> 10: `0.7676` to `0.7718`
- 10 -> 11 under the later schedule: `0.7817` to `0.7843`
- 11 -> 12: `0.7843` to `0.7860`
- 12 -> 13: `0.7860` to `0.7875`

Interpretation:

The dataset likely contains high-order categorical interactions. Carrier, airport, date, day of week, departure time, and distance all interact. Shallow models cannot represent enough of those conjunctions.

### Stronger Child-Weight Pruning

At depth 10:

- `min_child_weight=1`: `0.7675`, bad
- `min_child_weight=5`: `0.7738`, good
- `min_child_weight=8`: `0.7751`, good
- `min_child_weight=12`: `0.7757`, best at that point
- `min_child_weight=20`: `0.7749`, too much

At depth 13:

- `min_child_weight=18`: `0.7849`, over-pruned
- `min_child_weight=8`: timeout
- `min_child_weight=12`: best viable point

Interpretation:

Deep trees need a middle ground: enough pruning to avoid tiny noisy leaves and runtime explosion, but not so much that useful interactions disappear.

### Column Sampling

Results:

- `colsample_bytree=1.0`: `0.7712`, bad
- `0.85`: depth-10 regularized base `0.7757`
- `0.70`: `0.7776`
- `0.55`: `0.7791`
- `0.50`: tied `0.7791`, discarded by rule
- `0.40`: `0.7774`, too aggressive

Interpretation:

Substantial feature subsampling improved generalization. With deep categorical trees, always exposing all features may let trees repeatedly exploit unstable categorical splits. But too little column availability starves the model.

### Longer Lower-Rate Boosting

Results:

- 400 at 0.02: `0.7794`
- 600 at 0.015: `0.7802`
- 800 at 0.012: `0.7813`
- 1000 at 0.01: `0.7817`
- 1200 at 0.008: `0.7816`, discarded

Interpretation:

Smoother boosting helped until about 1000 trees. Beyond that, returns vanished and runtime rose.

## What Did Not Work

### Naive Time Feature Engineering

Attempts:

- `DepHour` plus `DepMinutes`: `0.7771`
- categorical `DepHour`: `0.7776`

Why likely failed:

- Raw `DepTime` already enabled useful time splits.
- Extra redundant features created split competition.
- Column sampling means adding redundant features changes the feature subsets each tree sees.
- Numeric minute parsing may not add enough new information beyond raw HHMM.

### High-Cardinality Route Category

Attempt:

- `Route = Origin_Dest`: `0.7756`

Why likely failed:

- Sparse high-cardinality categories create noisy leaves.
- Existing `Origin` and `Dest` features already let deep trees form interactions.
- The route feature increased runtime.

### Extra Gamma

Attempt:

- `gamma=0.05`: `0.7755`, worse than `0.7757`

Why likely failed:

- `min_child_weight` already supplied useful pruning.
- Gamma pruned some marginal splits that were still useful.

### Forced Categorical One-Hot Threshold

Attempt:

- `max_cat_to_onehot=32`: `0.7833`

Why likely failed:

- XGBoost's default partitioning behavior was better for these categorical features.
- One-hot style splits may be too granular or less efficient for the relevant categorical interactions.

### Too Much Depth Or Too Loose Pruning

Attempts:

- `max_depth=14`: timeout
- `max_depth=13`, `min_child_weight=8`: timeout

Why likely failed:

- Deep categorical trees become computationally expensive.
- Looser child-weight pruning creates too many candidate splits/leaves.
- The experiment loop has a hard practical runtime budget.

## Detailed Timeline Table

| # | Commit | AUC | Status | Change | Interpretation |
|---:|---|---:|---|---|---|
| 0 | `7d0b368` | 0.7445 | keep | Baseline | Underfit reference point. |
| 1 | `4c2aa50` | 0.7497 | keep | 120 trees, eta 0.05, depth 5, sampling, child-weight, lambda | Baseline was underfit; regularized extra boosting helps. |
| 2 | `f37a992` | 0.7511 | keep | 220 trees, eta 0.035 | Lower-rate longer boosting still helps. |
| 3 | `dbc1fa8` | 0.7557 | keep | Depth 6 | Interactions matter. |
| 4 | `9765533` | 0.7599 | keep | Depth 7 | Depth trend continues. |
| 5 | `0e16eec` | 0.7643 | keep | Depth 8 | Still interaction-limited. |
| 6 | `a0f8e7c` | 0.7676 | keep | Depth 9 | Good gain, slightly higher fold spread. |
| 7 | `d723a6b` | 0.7718 | keep | Depth 10 | Strong gain; tune pruning next. |
| 8 | `414cbb1` | 0.7675 | discard | `min_child_weight=1` | Too loose; noisy leaves. |
| 9 | `6097c86` | 0.7738 | keep | `min_child_weight=5` | Stronger pruning helps. |
| 10 | `16ca59b` | 0.7751 | keep | `min_child_weight=8` | More pruning still helps. |
| 11 | `abbf167` | 0.7757 | keep | `min_child_weight=12` | Small gain; near pruning optimum. |
| 12 | `b14ad5b` | 0.7749 | discard | `min_child_weight=20` | Over-pruned. |
| 13 | `69a4b9a` | 0.7755 | discard | `gamma=0.05` | No additive value beyond child weight. |
| 14 | `8f590da` | 0.7752 | discard | `subsample=1.0` | Row stochasticity needed. |
| 15 | `4e236f2` | 0.7751 | discard | `subsample=0.75` | Too much row sampling. |
| 16 | `4c4d0ae` | 0.7712 | discard | `colsample_bytree=1.0` | Full features overfit/reduced stochasticity. |
| 17 | `d1daf26` | 0.7776 | keep | `colsample_bytree=0.70` | Stronger feature sampling helps. |
| 18 | `8ed423e` | 0.7791 | keep | `colsample_bytree=0.55` | Best column sampling point found. |
| 19 | `1cf4f3f` | 0.7774 | discard | `colsample_bytree=0.40` | Too little feature availability. |
| 20 | `69d863f` | 0.7791 | discard | `colsample_bytree=0.50` | Tied, discarded by rule. |
| 21 | `91942f9` | 0.7771 | discard | Numeric time features | Redundant/noisy. |
| 22 | `8600faf` | 0.7776 | discard | Categorical hour | Still redundant/noisy. |
| 23 | `04ceb60` | 0.7756 | discard | Route category | Sparse high-cardinality feature hurt. |
| 24 | `92f6aee` | 0.7794 | keep | 400 trees, eta 0.02 | Schedule gain resumes. |
| 25 | `81b0471` | 0.7802 | keep | 600 trees, eta 0.015 | Smooth boosting helps. |
| 26 | `7338924` | 0.7813 | keep | 800 trees, eta 0.012 | Meaningful schedule gain. |
| 27 | `1b7f12e` | 0.7817 | keep | 1000 trees, eta 0.01 | Best schedule before flattening. |
| 28 | `191ca6d` | 0.7816 | discard | 1200 trees, eta 0.008 | Slower, no improvement. |
| 29 | `92276bc` | 0.7843 | keep | Depth 11 | Depth reopened under better schedule. |
| 30 | `32281af` | 0.7860 | keep | Depth 12 | More useful interaction capacity. |
| 31 | `a20ccac` | 0.7875 | keep | Depth 13 | Current best; runtime still legal. |
| 32 | `757d2a4` | 0.0000 | crash | Depth 14 | Timed out. |
| 33 | `deb23fa` | 0.7849 | discard | `min_child_weight=18` at depth 13 | Over-pruned. |
| 34 | `020d67e` | 0.0000 | crash | `min_child_weight=8` at depth 13 | Timed out. |
| 35 | `2d4bca0` | 0.7833 | discard | `max_cat_to_onehot=32` | Default categorical handling better. |

## Recommendations For Next Work

Most promising next HPO ideas:

1. Try `max_depth=13`, `n_estimators=900`, `learning_rate=0.011` or `0.012` as a simplification/runtime ablation. It may preserve most of the best AUC with lower runtime.
2. Try `max_depth=13`, `colsample_bytree=0.60` or `0.50` under the final 1000-tree schedule. Column sampling was tuned before the later depth increase, so the optimum might shift.
3. Try `reg_lambda` around the final model: `1.0`, `3.0`, `5.0`. L2 regularization has not been bracketed since the first experiment.
4. Try `reg_alpha` small values such as `0.05` or `0.1`. It could prune weak leaf weights differently from `min_child_weight`.
5. Research and test `grow_policy="lossguide"` with `max_leaves` only if runtime can be managed. The idea would be to allow selective deep interactions without forcing full depthwise expansion.

Less promising directions:

- More naive time features.
- Direct high-cardinality route features.
- Larger depth without stronger constraints.
- More trees beyond 1000 at lower eta.
- Forcing `max_cat_to_onehot=32`.

Main lesson:

The winning pattern is not more handcrafted features. It is a high-capacity XGBoost model carefully regularized enough to make deep categorical interactions generalize within the runtime budget.
