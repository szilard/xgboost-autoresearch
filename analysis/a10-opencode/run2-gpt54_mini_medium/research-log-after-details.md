# Detailed Research Log

## Sources Consulted

- XGBoost parameter tuning docs: confirmed the main knobs for bias/variance tradeoffs were `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `max_bin`, and categorical thresholds.
- XGBoost parameter reference: clarified `tree_method=hist`, `grow_policy=lossguide`, `max_leaves`, `max_cat_to_onehot`, `max_cat_threshold`, and the effect of `lambda`/`alpha`.
- XGBoost early stopping docs: showed that native `xgb.train()` supports `early_stopping_rounds` and `best_iteration`, which was important because the sklearn wrapper in this environment rejected `early_stopping_rounds` and `callbacks` in `fit()`.
- Feature-engine cyclical feature guide and scikit-learn time-feature engineering example: supported encoding periodic fields with sine/cosine rather than leaving them as raw ordinals.
- XGBoost feature interaction constraints docs: provided the idea of constraining interactions, which I tested and rejected because it over-constrained the model.
- Web searches for airline-delay / flight-delay feature engineering: surfaced route/origin/destination composite features and route-level interactions as common tabular tricks, which I tried briefly but dropped because the experiment timed out.

## Phase 1: Baseline And First Bias/Variance Tuning

### Iteration 1

Baseline only. I first verified the environment, discovered that system Python was missing `pandas`, and reran in `.venv`, which gave the starting point of `CV AUC 0.7445 ± 0.0043`. This established the reference point and confirmed the repo was runnable without changing any model code.

### Iteration 2

I expected the baseline to be underfit, so I increased the boosting budget and regularization balance: more trees, smaller learning rate, shallower trees, row/column subsampling, and stronger `lambda`/`alpha`/`gamma`. That moved the score to `0.7494`, which was a modest but real improvement, so I kept it.

### Iteration 3

The cyclical feature sources suggested that raw calendar-like variables should often be transformed into sine/cosine pairs. I added cyclic encodings for `Month`, `DayOfWeek`, and `DepTime` while keeping the categorical originals, and the AUC nudged up to `0.7495`; small gain, but it fit the periodicity intuition well enough to keep.

### Iteration 4

The XGBoost early-stopping docs made it clear that the best number of trees should be selected fold-by-fold instead of fixed by hand. I rewrote the training loop to native `xgb.train()` with `early_stopping_rounds=50`, `num_boost_round=2000`, and fold-specific `best_iteration`, which lifted the score to `0.7555` and gave the first substantial gain from optimization rather than feature engineering.

### Iteration 5

The parameter docs and the earlier tuning results suggested the model might benefit from a more adaptive tree growth policy. I switched from the initial depth-limited setup to `grow_policy="lossguide"` with `max_leaves=64`, while letting `max_depth=0` so the leaf budget controlled complexity; that jumped the score to `0.7685`, the first big improvement.

### Iteration 6

Because `DepTime` had already been transformed into a cyclical signal, I tested whether the raw HHMM number was just noise. Dropping the raw `DepTime` column while keeping the cyclic features improved slightly to `0.7692`, so I kept the simplification.

### Iteration 7

I checked whether Month and DayOfWeek categories were redundant once the cyclic features existed. Removing those raw categories hurt slightly (`0.7690`), so I treated that as a discard and kept the original categorical month/day features.

## Phase 2: Categorical Handling And Small Structural Changes

### Iteration 8

The XGBoost categorical docs suggested that low-cardinality categories can benefit from one-hot style splits instead of always using partition-based splits. I raised `max_cat_to_onehot` to `32` and saw a large jump to `0.7819`, which was the first clear sign that categorical split handling mattered a lot more than the earlier minor regularization tweaks.

### Iteration 9

I tested whether reducing the leaf budget to `48` would keep the gain while improving speed. The score dropped to `0.7781`, so the lower capacity was too aggressive and I reverted it.

### Iteration 10

I increased `max_cat_to_onehot` further to `64` to see whether even more categories should be handled with one-hot splits. The score stayed at `0.7819` but the run was slower, so I discarded it as a no-op.

### Iteration 11

I raised `max_cat_threshold` to `128` to let partition-based splits consider more categories. The result fell to `0.7812`, so the extra threshold was not helping.

### Iteration 12

Per-node column subsampling felt like a plausible overfitting control for the lossguide trees, so I tried `colsample_bynode=0.8`. The score barely moved (`0.7816`) and I treated it as a discard.

### Iteration 13

I lowered `min_child_weight` from `5` to `3` to see whether the lossguide trees were being over-pruned. That backfired to `0.7793`, which told me the model needed the stronger child-weight constraint, not less of it.

### Iteration 14

The route-feature searches on flight-delay modeling made route and carrier-route composites look worth testing. I added `Route` and `CarrierRoute` columns, but the experiment timed out before producing a result, so I recorded it as a crash and did not keep the idea.

### Iteration 15

I tried a faster boosting schedule with a higher learning rate and fewer rounds (`eta=0.05`, `num_boost_round=1200`, `early_stopping_rounds=30`). The score fell to `0.7801`, which suggested the original learning rate was closer to the sweet spot.

### Iteration 16

I tried a midpoint schedule (`eta=0.04`, `num_boost_round=1500`, `early_stopping_rounds=40`) to see whether the faster and original schedules bracketed a better point. It still underperformed at `0.7810`, so the learning-rate search was not the place to find the next big gain.

### Iteration 17

I tested feature interaction constraints based on the XGBoost docs, splitting time-related and location-related groups. The syntax accepted by this build was finicky, but once corrected the model collapsed to `0.7465`; the constraints were far too restrictive and harmed the model badly.

## Phase 3: Histogram Resolution And Leaf Budget Search

### Iteration 18

The docs said finer histogram bins can improve split quality at some computational cost, so I raised `max_bin` to `512`. That produced `0.7827`, a real gain, so I kept it.

### Iteration 19

I increased `max_bin` further to `1024` to see whether the bin resolution was still limiting the model. The score improved again to `0.7834`, which made the histogram resolution a clearly relevant knob.

### Iteration 20

I pushed `max_bin` to `2048` to probe the upper bound. The score stayed at `0.7834` while runtime worsened, so I discarded it.

### Iteration 21

I raised `max_leaves` to `96` to see whether the stronger histogram resolution needed more leaf capacity. The run timed out, so that version was too slow for the experiment budget.

### Iteration 22

I backed off to `max_leaves=80`, which completed and improved to `0.7873`. That was the first evidence that the leaf budget had a better optimum higher than the original `64` but not too high to break runtime.

### Iteration 23

I tried `max_leaves=88` to refine the leaf-budget optimum. It timed out, confirming that the runtime cliff was real.

### Iteration 24

I tried `max_leaves=84` as another nearby probe. It also timed out, so the workable region was clearly below that.

### Iteration 25

I widened `max_cat_to_onehot` to `48` to see if the categorical split balance moved once the leaf budget had increased. The score did not improve and runtime got worse, so I discarded it.

### Iteration 26

I raised `gamma` to `0.2` to force more conservative splits. The score stayed flat at `0.7873`, so the extra split penalty was not buying anything.

### Iteration 27

I increased `subsample` and `colsample_bytree` to `0.9`, expecting a milder regularization change. It reduced AUC to `0.7869`, so the model wanted the stronger stochasticity from `0.8`.

### Iteration 28

I tried `subsample=0.85` and `colsample_bytree=0.85` as a midpoint. The result fell further to `0.7861`, so the original `0.8` values were the best of the three.

### Iteration 29

I reduced `lambda` to `1.0` and `alpha` to `0.0`. That was a useful move: the score improved to `0.7874`, showing the model could tolerate less regularization once the leaf budget and histogram resolution had been raised.

### Iteration 30

I tried `lambda=0.5` to see if the improved tree capacity wanted even less penalty. The score slipped to `0.7873`, so `1.0` was better.

## Phase 4: Fine-Tuning The Same Region

### Iteration 31

I tested `min_child_weight=4`, expecting a modest relaxation from `5`. It underperformed at `0.7855`, which told me the tree still needed the stronger child-weight floor.

### Iteration 32

I nudged the learning rate upward to `0.035` with a shorter schedule. The score dropped to `0.7870`, so the original `0.03` was still the better balance.

### Iteration 33

I tried a midpoint `eta=0.032` to see whether the optimum was between the prior two values. It also lost ground (`0.7873`), so there was no benefit in moving away from `0.03`.

### Iteration 34

I added a tiny amount of L1 regularization with `alpha=0.02`. The score remained flat at `0.7873`, so the simpler `alpha=0.0` was preferable.

### Iteration 35

I lowered `gamma` to `0.0` to let the model split more freely. That did not improve the score, but it did show that the split penalty was not the key bottleneck.

### Iteration 36

I raised `max_leaves` to `82`, which gave `0.7878`. This was a meaningful gain and established a better leaf-budget sweet spot than `80`.

### Iteration 37

I tried `max_leaves=83` as a nearby refinement. It slipped slightly to `0.7877`, so `82` was better.

### Iteration 38

I increased `max_bin` to `1536` to test whether the histogram resolution still had room to help with the larger leaf budget. It did not; the score fell to `0.7874`, so `1024` remained the better choice.

### Iteration 39

I tried the DART booster with dropout parameters because the docs suggest it as a regularization option. The run timed out, and the experiment was too heavy for the budget, so I dropped the idea.

## Phase 5: Switching To Depthwise Trees

### Iteration 40

I switched from lossguide to a simpler depthwise tree (`max_depth=8`) to see whether a more standard tree-growth pattern could outperform the leaf-budgeted model. It was not better (`0.7877`), but it gave a much faster runtime path and set up the next depth sweep.

### Iteration 41

I increased `max_depth` to `9`. That produced a strong jump to `0.7926`, which was the first indication that the depthwise path had more upside than the previous leaf-budgeted path.

### Iteration 42

I moved to `max_depth=10`, which improved again to `0.7955`. The signal was clearly still there.

### Iteration 43

I tried `max_depth=11`, and the score rose to `0.7987`. That convinced me the deeper trees were still capturing useful interactions rather than just overfitting noise.

### Iteration 44

`max_depth=12` with the original schedule timed out, so I shortened the boosting budget to `1200` rounds with `30` rounds of patience. The shorter schedule completed, but the score dipped to `0.7978`, which meant the longer tree needed a bit more training than that.

### Iteration 45

I increased `max_depth` to `13` under the shorter schedule. The score climbed again to `0.7999`, showing the depthwise model still had headroom.

### Iteration 46

I moved to `max_depth=14`, which yielded `0.8002`. The gains were becoming smaller, but still positive.

### Iteration 47

I raised the depth to `15`, and the score reached `0.8025`. That was a meaningful jump and confirmed that the deeper tree was a better fit than the earlier lossguide setup.

### Iteration 48

`max_depth=16` improved slightly again to `0.8027`. The trend was still upward, though clearly with diminishing returns.

### Iteration 49

I increased `max_depth` to `17`, which pushed the score to `0.8034`. The gain was small but real, so I kept it moving.

### Iteration 50

`max_depth=18` gave `0.8041`, the best score up to that point. This became the key depth setting for the remainder of the search.

## Phase 6: Plateau Search Around Depth 18

### Iteration 51

I tested `max_depth=19` to see whether the curve was still rising. It matched the previous region but did not beat `18`, so I treated it as a discard.

### Iteration 52

I tried `max_depth=20`, expecting that more depth might still help. It regressed slightly to `0.8039`, which was enough to rule it out.

### Iteration 53

I lowered `eta` to `0.028` and increased the boosting budget to `2200` rounds with `45` rounds of patience. That version timed out, which showed that the lower learning rate made the schedule too expensive.

### Iteration 54

I raised `max_bin` to `2048` at `max_depth=18` to see whether the deeper tree could use the finer histogram resolution. It fell to `0.8037`, so the extra resolution did not help.

### Iteration 55

I lowered `gamma` to `0.0` at `max_depth=18` to let the deeper tree split more freely. It underperformed at `0.8038`, so `gamma=0.1` was a better generalization control.

### Iteration 56

I increased the boosting budget to `1500` rounds with `40` rounds of patience at `max_depth=18`. That nudged the score to `0.8042`, which became the final best configuration at this stage.

### Iteration 57

I tried an even longer schedule (`1800` rounds, `50` patience) because the deeper tree seemed close to saturation. The run timed out, so it was too much for the runtime budget.

### Iteration 58

I tested `1600` rounds with `45` patience. The score matched the best (`0.8042`) but runtime was worse, so I discarded it as a no-op improvement.

### Iteration 59

I tried a midpoint `gamma=0.05` at `max_depth=18`. The score fell back to `0.8038`, confirming that the original split penalty was still better.

### Iteration 60

I returned to `max_depth=19` with the better `1500/40` schedule to see whether the earlier timeout was purely a schedule issue. It still timed out, so the deeper tree was simply too expensive here.

### Iteration 61

I increased `min_child_weight` to `6` at `max_depth=18` and kept the longer schedule. That version timed out as well, indicating that the additional regularization interacted badly with the deep tree’s runtime.

### Iteration 62

I kept `min_child_weight=6` but shortened the schedule to `1200/30` to make it finish. It completed with `0.8035`, which was below the best score, so the stronger child-weight penalty was not worth it.

### Iteration 63

I raised `lambda` to `1.5` at `max_depth=18` with the longer schedule, expecting a modest regularization gain. The run timed out, so the extra penalty made the training loop too heavy.

### Iteration 64

I kept `lambda=1.5` but shortened the schedule to `1200/30` so it could complete. The score fell to `0.8031`, so the stronger L2 penalty was not beneficial even when the runtime issue was removed.

## What Worked

- The strongest single shift was moving from lossguide to depthwise trees and then increasing `max_depth` progressively.
- The winning depthwise region was `max_depth=18` with a slightly longer boosting schedule (`1500` rounds, `40` patience) and the earlier regularization settings.
- Histogram resolution mattered up to `max_bin=1024`, but not beyond that.
- One-hot handling for categorical splits helped a lot at `max_cat_to_onehot=32`, but higher thresholds were not better.
- Cyclical time features were a small but stable improvement and were worth keeping.

## What Did Not Work

- Explicit route/carrier-route features timed out and never got a usable result.
- Interaction constraints were too restrictive and crushed performance.
- DART timed out.
- Higher `max_bin` values beyond `1024` did not help.
- Pushing `max_leaves` beyond the workable region caused timeouts or regressions.
- Once the model moved to depthwise trees, most of the smaller regularization tweaks (`gamma`, `lambda`, `alpha`, `subsample`, `colsample_bytree`, `min_child_weight`) were either flat or worse than the current best setting.

## Final Best Configuration

- Booster: `gbtree`
- Tree method: `hist`
- Growth: depthwise
- `max_depth=18`
- `eta=0.03`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=5`
- `lambda=1.0`
- `alpha=0.0`
- `gamma=0.1`
- `max_cat_to_onehot=32`
- `max_bin=1024`
- Training: `num_boost_round=1500`, `early_stopping_rounds=40`
- Best observed CV AUC: `0.8042`
