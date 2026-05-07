# Research Log

## 7d0b368 baseline

Baseline unchanged train.py. CV AUC 0.7445 +/- 0.0043.

## Research before first non-baseline

Sources consulted: XGBoost parameter documentation and recent tuning guidance. Takeaways: tune tree capacity with max_depth/min_child_weight, use subsample and colsample_bytree for variance control, and try more estimators with a lower learning rate. Airline delay resources emphasize schedule, carrier, origin/destination, and temporal patterns; feature engineering will follow after a clean hyperparameter baseline.

## Experiment hypothesis - conservative regularized booster

Classification: exploration. Hypothesis: replacing the shallow 30-tree baseline with more lower-rate trees plus mild row/column sampling and higher child/leaf regularization should improve AUC by reducing underfit while controlling variance.

Result: commit 4c2aa50 CV AUC 0.7497 +/- 0.0048. Kept. More lower-rate trees with mild regularization improved over baseline by 0.0052.

## Experiment hypothesis - slower longer boosting

Classification: follow-up. Hypothesis: experiment 1 improved because lower-rate boosting with more trees reduced underfit; 220 estimators at eta 0.035 may continue that trend without making trees deeper.

Result: commit f37a992 CV AUC 0.7511 +/- 0.0047. Kept. Longer lower-rate boosting improved by 0.0014 over 4c2aa50 with runtime still low.

## Experiment hypothesis - restore depth 6

Classification: follow-up. Hypothesis: with lower learning rate and sampling in place, depth 6 may capture route/time/carrier interactions better than depth 5 without overfitting as much as the original baseline.

Result: commit dbc1fa8 CV AUC 0.7557 +/- 0.0046. Kept. Depth 6 improves materially, suggesting useful higher-order interactions remain.

## Experiment hypothesis - depth 7 probe

Classification: follow-up. Hypothesis: since depth 6 improved strongly, depth 7 may capture additional airport-route temporal interactions; current sampling and min_child_weight may keep variance acceptable.

Result: commit 9765533 CV AUC 0.7599 +/- 0.0047. Kept. Depth 7 continues the interaction-depth trend, with runtime still acceptable.

## Experiment hypothesis - depth 8 edge probe

Classification: follow-up. Hypothesis: depth 8 may continue improving if route-carrier-airport interactions are complex; if CV drops, depth 7 is likely the better variance point.

Result: commit 0e16eec CV AUC 0.7643 +/- 0.0048. Kept. Depth trend remains positive through 8; runtime still below limit.

## Experiment hypothesis - depth 9 probe

Classification: follow-up. Hypothesis: monotonic gains through depth 8 suggest the categorical route/time structure benefits from deeper splits; depth 9 tests whether this continues under current sampling.

Result: commit a0f8e7c CV AUC 0.7676 +/- 0.0052. Kept. Depth 9 improves but std rose slightly; continue one more depth probe then reassess.

## Experiment hypothesis - depth 10 probe

Classification: follow-up. Hypothesis: depth 10 may add a final increment if the problem is still interaction-limited, but the increased fold variance at depth 9 makes this a boundary test.

Result: commit d723a6b CV AUC 0.7718 +/- 0.0048. Kept. Depth 10 still improves; next tune regularization/sampling around deeper trees.

## Experiment hypothesis - lower child weight at depth 10

Classification: follow-up. Hypothesis: depth 10 is useful, so reducing min_child_weight from 3 to 1 may allow valuable smaller route/carrier/time leaves that were previously pruned.

Result: commit 414cbb1 CV AUC 0.7675 +/- 0.0044. Discard. Lower child weight overfit or added noisy fine leaves; reset to d723a6b.

## Experiment hypothesis - higher child weight at depth 10

Classification: follow-up. Hypothesis: since min_child_weight 1 lost, increasing to 5 may prune noisy deep leaves while retaining depth-10 interaction structure.

Result: commit 6097c86 CV AUC 0.7738 +/- 0.0047. Kept. More pruning at depth 10 improves over min_child_weight 3 and strongly over 1.

## Experiment hypothesis - stronger child weight pruning

Classification: follow-up. Hypothesis: min_child_weight 5 improved deep trees, so 8 may further reduce noisy small leaves while preserving useful depth-10 interactions.

Result: commit 16ca59b CV AUC 0.7751 +/- 0.0050. Kept. Stronger child-weight pruning continues to improve depth-10 model.

## Synthesis after 10 experiments

The main gains came from addressing underfit with more lower-rate trees and then increasing interaction depth. Depth improved monotonically from 5 through 10, but unconstrained small leaves hurt: min_child_weight 1 was a clear discard, while 5 and 8 improved. Current theory: this dataset has real high-order interactions among route, carrier, airport, and scheduled time, but those interactions need pruning because many categorical leaves are noisy. Next directions: research and tune additional regularization/sampling around the deep model, then add compact time/domain features inside prepare(df).

## Research after first 10 experiments

Sources consulted: XGBoost parameter documentation, XGBoost categorical-data documentation, and airline delay feature-engineering writeups. Takeaways: continue tuning child-weight/gamma/sampling for deep trees; categorical handling can be influenced by max_cat_to_onehot and max_cat_threshold; time-of-day blocks are commonly useful in flight delay models and should be added inside prepare(df) after regularization is bracketed.

## Experiment hypothesis - min_child_weight 12

Classification: follow-up. Hypothesis: child-weight pruning has helped through 8, so 12 may further reduce noisy deep categorical leaves; a drop would place the optimum near 8.

Result: commit abbf167 CV AUC 0.7757 +/- 0.0048. Kept. More child-weight pruning still improves but gain is small.

## Experiment hypothesis - min_child_weight 20

Classification: follow-up. Hypothesis: gains from child-weight pruning are flattening; 20 tests whether stronger pruning still helps or starts underfitting the useful deep splits.

Result: commit b14ad5b CV AUC 0.7749 +/- 0.0049. Discard. Over-pruning starts by 20; reset to abbf167 with min_child_weight 12.

## Experiment hypothesis - small gamma pruning

Classification: follow-up. Hypothesis: a small gamma can suppress marginal deep splits that pass child-weight but add little gain, improving generalization around the min_child_weight 12 best.

Result: commit 69a4b9a CV AUC 0.7755 +/- 0.0051. Discard. Small gamma did not beat no-gamma; reset to abbf167.

## Experiment hypothesis - full row sampling

Classification: follow-up. Hypothesis: after stronger child-weight pruning, using all rows per tree may improve split estimates and reduce sampling variance while column sampling still regularizes.

Result: commit 8f590da CV AUC 0.7752 +/- 0.0060. Discard. Removing row sampling did not help and increased fold spread; reset to abbf167.

## Experiment hypothesis - stronger row sampling

Classification: follow-up. Hypothesis: full row sampling was worse, so stronger stochastic row sampling at 0.75 may further regularize depth-10 trees and improve AUC.

Result: commit 4e236f2 CV AUC 0.7751 +/- 0.0050. Discard. More row sampling did not help; 0.85 remains local best.

## Experiment hypothesis - full column sampling

Classification: follow-up. Hypothesis: this dataset has few features, so using all columns per tree may improve split quality while row sampling and child-weight still regularize.

Result: commit 4c4d0ae CV AUC 0.7712 +/- 0.0047. Discard. Full columns overfit or reduced beneficial stochasticity; reset to abbf167.

## Experiment hypothesis - stronger column sampling

Classification: follow-up. Hypothesis: full columns were worse, so stronger column sampling at 0.70 may further regularize deep categorical splits and improve AUC.

Result: commit d1daf26 CV AUC 0.7776 +/- 0.0054. Kept. Stronger column sampling improved, though fold std rose slightly.

## Experiment hypothesis - colsample 0.55

Classification: follow-up. Hypothesis: colsample 0.70 improved over 0.85, so 0.55 may continue regularizing deep trees; if it drops, the useful range is near 0.70.

Result: commit 8ed423e CV AUC 0.7791 +/- 0.0050. Kept. Column sampling trend remains positive down to 0.55.

## Experiment hypothesis - aggressive colsample 0.40

Classification: follow-up. Hypothesis: if strong column randomness is still helping, 0.40 may improve further; otherwise it will starve trees of key schedule/airport splits and identify 0.55 as the local best.

Result: commit 1cf4f3f CV AUC 0.7774 +/- 0.0050. Discard. Too much column sampling loses signal; reset to 8ed423e.

## Experiment hypothesis - colsample 0.50 refinement

Classification: follow-up. Hypothesis: 0.55 beat 0.70 and 0.40 lost, so 0.50 may sit closer to the optimum than 0.55 while preserving enough feature availability.

Result: commit 69d863f CV AUC 0.7791 +/- 0.0050. Discard by rule because it tied rather than improved; reset to 8ed423e with colsample 0.55.

## Synthesis after 20 experiments

The second block refined regularization. Child-weight optimum is around 12: 20 over-prunes and 1 under-prunes badly. Gamma 0.05 did not add value beyond child-weight. Row sampling has a local best at 0.85; both 1.0 and 0.75 were worse. Column sampling is important: 1.0 was much worse, 0.70 improved, 0.55 improved again, 0.40 lost, and 0.50 tied but was discarded by rule. Current best is 0.7791 at depth 10, min_child_weight 12, subsample 0.85, colsample_bytree 0.55. Next direction: add compact time-of-day feature engineering inside prepare(df), because external airline-delay resources identify departure hour/time blocks as useful and the raw DepTime value may be awkwardly encoded.

## Experiment hypothesis - parsed departure time features

Classification: exploration. Hypothesis: airline-delay research highlights time-of-day effects; adding DepHour and DepMinutes inside prepare(df) should make temporal splits easier than relying only on raw HHMM DepTime.

Result: commit 91942f9 CV AUC 0.7771 +/- 0.0053. Discard. Parsed numeric time features hurt, likely redundant/noisy with raw DepTime. Reset to 8ed423e.

## Experiment hypothesis - categorical departure hour

Classification: exploration. Hypothesis: numeric parsed time hurt, but a categorical hour may capture rush-hour style groups without imposing HHMM ordinality or adding redundant minute detail.

Result: commit 8600faf CV AUC 0.7776 +/- 0.0056. Discard. Categorical hour did not beat raw DepTime-only best. Reset to 8ed423e.

## Experiment hypothesis - route interaction category

Classification: exploration. Hypothesis: direct Origin_Dest route categories may capture route-specific congestion and network effects more easily than separate origin/destination splits, without using count-derived leakage-prone features.

Result: commit 04ceb60 CV AUC 0.7756 +/- 0.0050. Discard. High-cardinality route interaction hurt and slowed training; reset to 8ed423e.

## Experiment hypothesis - smoother 400-tree schedule

Classification: follow-up. Hypothesis: after finding stronger pruning and column sampling, more trees with eta 0.02 may improve ranking by making the deep model learn more gradually without overfitting.

Result: commit 92f6aee CV AUC 0.7794 +/- 0.0049. Kept. Smoother longer boosting adds a small gain with acceptable runtime.

## Experiment hypothesis - 600-tree lower-rate schedule

Classification: follow-up. Hypothesis: 400 trees at eta 0.02 improved slightly; 600 at eta 0.015 may continue smoother learning while staying within runtime.

Result: commit 81b0471 CV AUC 0.7802 +/- 0.0051. Kept. Longer lower-rate boosting improved another 0.0008; runtime still acceptable.

## Experiment hypothesis - 800-tree schedule

Classification: follow-up. Hypothesis: if lower-rate boosting still has headroom, 800 trees at eta 0.012 may improve ranking; if runtime/gain tradeoff worsens or AUC drops, 600 is the better schedule.

Result: commit 7338924 CV AUC 0.7813 +/- 0.0050. Kept. Longer lower-rate schedule continues to improve, runtime 24.2s CV still acceptable.

## Experiment hypothesis - 1000-tree eta 0.01 schedule

Classification: follow-up. Hypothesis: 800 trees at eta 0.012 improved, so 1000 at eta 0.01 may continue smoother boosting gains while staying under the one-minute limit.

Result: commit 1b7f12e CV AUC 0.7817 +/- 0.0049. Kept. Best so far, but CV time 30.2s means schedule increases are nearing the practical limit.

## Experiment hypothesis - 1200-tree eta 0.008 schedule

Classification: follow-up. Hypothesis: smoother boosting has improved through 1000 trees; 1200 at eta 0.008 tests for remaining headroom, with runtime as the main risk.

Result: commit 191ca6d CV AUC 0.7816 +/- 0.0050. Discard. Slightly worse and slower than 1000-tree schedule; reset to 1b7f12e.

## Experiment hypothesis - depth 11 with final schedule

Classification: follow-up. Hypothesis: stronger pruning and column sampling may allow depth 11 to capture additional interactions beyond depth 10 without overfitting; runtime is the main risk.

Result: commit 92276bc CV AUC 0.7843 +/- 0.0050. Kept. Depth 11 gives a substantial gain under tuned pruning and sampling, with CV time 35.0s still under limit.

## Experiment hypothesis - depth 12 boundary with long schedule

Classification: follow-up. Hypothesis: depth 11 improved materially; depth 12 may capture still richer interactions, but overfit and runtime risk are now meaningful.

Result: commit 32281af CV AUC 0.7860 +/- 0.0049. Kept. Depth 12 improves again with 39.3s CV time, still under limit.

## Synthesis after 30 experiments

Feature engineering attempts were mostly negative: parsed numeric departure time, categorical departure hour, and high-cardinality Origin_Dest route all underperformed. The strongest gains are still pure XGBoost capacity/regularization. Longer lower-rate boosting improved through 1000 trees; 1200 was slower and slightly worse. Under the 1000-tree schedule, deeper trees became valuable again: depth 11 and 12 both improved materially. Current best is 0.7860 at depth 12, 1000 estimators, eta 0.01, min_child_weight 12, subsample 0.85, colsample_bytree 0.55, lambda 2.0. Current theory: high-order categorical interactions are real, and the model needs deep trees plus strong pruning/stochasticity, while naive extra derived features mostly add noise or redundant split choices. Next direction after research: bracket depth/regularization near depth 12, possibly increasing min_child_weight with deeper trees or trying categorical split parameters.

## Research after 30 experiments

Sources consulted: XGBoost parameter docs, categorical-data docs, grow_policy/max_leaves docs, and tuning notes. Takeaways: depth and min_child_weight jointly control complexity; deep trees can be useful but risk exponential cache/runtime growth; categorical features can be tuned with max_cat_to_onehot and max_cat_threshold; lossguide/max_leaves is another possible way to allow selective deep interactions.

## Experiment hypothesis - depth 13 boundary

Classification: follow-up. Hypothesis: depth 12 improved, so depth 13 may capture more useful categorical interactions, but runtime and overfit risk are now high.

Result: commit a20ccac CV AUC 0.7875 +/- 0.0049. Kept. Depth 13 improves and completed in 43.0s CV time, still within timeout though close enough to avoid large schedule increases.

## Experiment hypothesis - depth 14 timeout-boundary

Classification: follow-up. Hypothesis: depth 13 still improved, so depth 14 may add a little more interaction capacity; timeout risk is high, so this run is explicitly bounded.

Result: commit 757d2a4 timed out after 60s. Crash/runtime failure. Discard and reset to a20ccac.

## Experiment hypothesis - stronger pruning at depth 13

Classification: follow-up. Hypothesis: depth 13 is useful but near runtime/complexity limits; raising min_child_weight to 18 may prune noisy leaves, improve AUC, and reduce runtime.

Result: commit deb23fa CV AUC 0.7849 +/- 0.0050. Discard. Stronger child-weight pruning at depth 13 over-pruned; reset to a20ccac.

## Experiment hypothesis - lower pruning at depth 13

Classification: follow-up. Hypothesis: min_child_weight 18 over-pruned at depth 13; lowering to 8 may recover useful small-but-real leaves and improve beyond 12.

Result: commit 020d67e timed out after 60s. Crash/runtime failure. Looser pruning at depth 13 too expensive; reset to a20ccac.

## Experiment hypothesis - categorical one-hot threshold 32

Classification: exploration. Hypothesis: XGBoost categorical docs expose max_cat_to_onehot; raising it to 32 may improve splits for moderate-cardinality categorical fields while leaving high-cardinality airports partitioned.

Result: commit 2d4bca0 CV AUC 0.7833 +/- 0.0050. Discard. Raising one-hot threshold hurt; default categorical splitting is better. Reset to a20ccac.

