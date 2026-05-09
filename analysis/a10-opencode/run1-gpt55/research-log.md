# Research Log

## 7d0b368 - baseline

- Result: CV AUC 0.7445 +/- 0.0043.
- Notes: Baseline uses raw categorical features with XGBoost categorical support, `n_estimators=30`, `max_depth=6`, and `learning_rate=0.1`.

## d9c5c9e - conservative boosted trees

- Source: XGBoost parameter docs describe `learning_rate`/`eta` shrinkage as making boosting more conservative, `min_child_weight` as limiting splits with low Hessian support, and `subsample`/`colsample_bytree` as regularization against overfitting. The categorical tutorial notes `hist` is a supported tree method for categorical features.
- Hypothesis: The baseline is likely underfit with only 30 trees. More trees with lower learning rate plus row/column sampling should improve rank quality without overfitting badly.
- Classification: exploration.
- Result: CV AUC 0.7568 +/- 0.0048, keep.
- Observation: A large gain confirms the baseline was underfit and that regularized additional rounds are useful.

## 0971248 - extended conservative schedule

- Hypothesis: Since 160 rounds at 0.05 improved substantially, a lower learning rate with 300 rounds may continue reducing bias while preserving conservative updates.
- Classification: follow-up.
- Result: CV AUC 0.7586 +/- 0.0047, keep.
- Observation: Smaller gain than the first schedule change, but still positive. Additional capacity remains useful within runtime limits.

## 426dd6c - slower extended boosting

- Hypothesis: A yet slower 500-round schedule may capture additional signal if the 300-round model remains slightly underfit.
- Classification: follow-up.
- Result: CV AUC 0.7592 +/- 0.0045, keep.
- Observation: Improvement persists but is tapering. Runtime remains acceptable, so the model can afford more trees if they continue to help.

## 76d4a44 - longer low-rate boosting

- Hypothesis: More rounds at lower learning rate may still improve bias/variance because previous gains have not stopped and runtime remains far below 60 seconds.
- Classification: follow-up.
- Result: CV AUC 0.7614 +/- 0.0050, keep.
- Observation: The larger gain suggests the current model still benefits from substantially longer boosting schedules.

## 2844201 - classic low-rate schedule

- Hypothesis: 1200 rounds at 0.01 may improve over 800 at 0.015 by using smaller, more stable additive steps.
- Classification: follow-up.
- Result: CV AUC 0.7614 +/- 0.0049, discard.
- Observation: It ties the previous rounded AUC but costs noticeably more runtime, so the simpler 800-round schedule is better.

## 9dff1dd - shallower trees

- Hypothesis: Reducing `max_depth` from 6 to 5 might improve generalization by reducing tree complexity.
- Classification: exploration.
- Result: CV AUC 0.7559 +/- 0.0045, discard.
- Observation: Large drop indicates useful signal depends on interactions that depth 5 fails to capture under this schedule.

## 5e50bb0 - deeper regularized trees

- Hypothesis: Since depth 5 underfit badly, depth 7 may capture additional useful interactions; increasing `min_child_weight` to 10 should keep the extra depth from overfitting small leaves.
- Classification: exploration.
- Result: CV AUC 0.7696 +/- 0.0048, keep.
- Observation: Large improvement. The dataset benefits from richer interactions when leaf support is constrained.

## 3746f13 - deeper constrained trees

- Hypothesis: Depth 8 can capture still richer categorical and time/distance interactions if `min_child_weight=20` prevents unsupported leaves.
- Classification: follow-up.
- Result: CV AUC 0.7746 +/- 0.0050, keep.
- Observation: The depth/leaf-support tradeoff continues to be the strongest axis so far.

## c063231 - very deep constrained trees

- Hypothesis: A further depth increase may still capture useful high-order interactions if `min_child_weight=40` strongly restricts weak leaves.
- Classification: follow-up.
- Result: CV AUC 0.7754 +/- 0.0053, keep.
- Observation: Small improvement. The depth trend is tapering but not exhausted; very high leaf support is important for keeping depth useful.

## cda04be - one-hot categorical splits

- Source: XGBoost categorical docs describe `max_cat_to_onehot` as the threshold deciding one-hot splits versus partition-based splits. Category counts in this training slice are 7, 12, 20, 31 for calendar/carrier fields and 282 for origin/dest.
- Hypothesis: One-hot splits for lower-cardinality categoricals could isolate strong carrier/calendar effects while leaving airports partitioned.
- Classification: exploration.
- Result: CV AUC 0.7626 +/- 0.0055, discard.
- Observation: Large drop. Partition-based categorical splits are important, even for moderate-cardinality fields.

## 408ee25 - larger categorical partition threshold

- Hypothesis: Increasing `max_cat_threshold` to 128 might improve Origin/Dest partition splits by considering more airport categories per split.
- Classification: follow-up.
- Result: CV AUC 0.7751 +/- 0.0054, discard.
- Observation: Slightly worse than default. The existing categorical split threshold appears adequate or better regularized.

## fa6b2e3 - smaller categorical partition threshold

- Hypothesis: Lowering `max_cat_threshold` to 32 might regularize noisy airport partitions better than the default.
- Classification: follow-up.
- Result: CV AUC 0.7754 +/- 0.0055, discard.
- Observation: Ties best rounded AUC but adds an explicit parameter with no improvement, so default remains preferable.

## a57f0de - departure time features

- Source: BTS documentation defines delays relative to scheduled time and describes operational delay causes, motivating time-of-day features as a domain signal.
- Hypothesis: Converting HHMM-like `DepTime` into `DepHour` and minutes since midnight might make time-of-day split structure easier for trees.
- Classification: exploration.
- Result: CV AUC 0.7753 +/- 0.0054, discard.
- Observation: Slightly below best. Deep trees can already extract enough from raw `DepTime`, and extra redundant features do not help.

## bdc49a3 - stronger sampling regularization

- Hypothesis: Lowering row and column sampling from 0.85 to 0.75 might reduce variance in the deep model.
- Classification: exploration.
- Result: CV AUC 0.7746 +/- 0.0053, discard.
- Observation: More sampling hurts. The current model likely needs more feature/row exposure to learn sparse categorical interactions.

## 5d76b9b - lighter sampling regularization

- Hypothesis: Since heavier sampling hurt, exposing more rows/features with 0.95 sampling may improve sparse interaction learning.
- Classification: follow-up.
- Result: CV AUC 0.7752 +/- 0.0054, discard.
- Observation: Also below best. The existing 0.85/0.85 balance is better than either direction tested.

## 2ae42f3 - stronger L2 regularization

- Hypothesis: Adding `reg_lambda=5` might improve the deep model by damping noisy leaf weights.
- Classification: exploration.
- Result: CV AUC 0.7749 +/- 0.0054, discard.
- Observation: Slightly worse. `min_child_weight=40` seems to provide enough regularization; extra L2 hurts fit.

## 55f7beb - relaxed L2 regularization

- Hypothesis: Since `reg_lambda=5` hurt, slightly lower-than-default L2 may let supported deep leaves fit useful signal better.
- Classification: follow-up.
- Result: CV AUC 0.7755 +/- 0.0053, keep.
- Observation: Tiny but positive gain from a single simple parameter. Leaf weights benefit from being a little less constrained.

## eba944b - no L2 regularization

- Hypothesis: If `reg_lambda=0.5` helps, removing L2 entirely may further improve leaf weights because `min_child_weight=40` already controls weak leaves.
- Classification: follow-up.
- Result: CV AUC 0.7757 +/- 0.0054, keep.
- Observation: Another small gain. Explicit L2 appears unnecessary with the current high child-weight constraint.

## d610e9c - small split-gain threshold

- Hypothesis: A small `gamma=0.1` threshold might prune weak deep splits while leaving useful leaf weights unconstrained.
- Classification: follow-up.
- Result: CV AUC 0.7756 +/- 0.0054, discard.
- Observation: Slightly worse. With `min_child_weight=40`, additional split-gain regularization is not helpful.

## 15a0f69 - lossguide tree growth

- Source: XGBoost tree-method docs note `hist` supports `grow_policy=lossguide` and `max_leaves`, choosing splits by highest loss change rather than depthwise expansion.
- Hypothesis: Asymmetric lossguide trees with 512 leaves could focus splits where categorical interactions are strongest.
- Classification: exploration.
- Result: CV AUC 0.7757 +/- 0.0054, discard.
- Observation: Tied best rounded AUC but almost doubled runtime and added complexity, so depthwise remains preferable.

## 2bc3828 - higher histogram resolution

- Source: XGBoost tree-method docs note `hist` uses approximate histogram splits and that higher `max_bin` can improve split optimality while maintaining good performance.
- Hypothesis: More bins may refine splits for `DepTime` and `Distance` without changing categorical handling.
- Classification: exploration.
- Result: CV AUC 0.7760 +/- 0.0054, keep.
- Observation: Small but clean gain with effectively unchanged runtime. Higher numeric split resolution is beneficial.

## ae1b2c3 - very high histogram resolution

- Hypothesis: If `max_bin=512` helps, `max_bin=1024` may further improve numeric split placement while staying inside the runtime budget.
- Classification: follow-up.
- Result: CV AUC 0.7762 +/- 0.0054, keep.
- Observation: Small additional gain. Runtime is still acceptable, so higher bin resolution remains useful.

## 97e92b6 - max histogram resolution probe

- Hypothesis: `max_bin=2048` may further refine numeric split thresholds beyond 1024 while staying inside the 1-minute experiment budget.
- Classification: follow-up.
- Result: CV AUC 0.7764 +/- 0.0056, keep.
- Observation: Another small gain. Runtime increased but remains acceptable; this may be near the practical resolution limit.

## 95bce1f - extreme histogram resolution

- Hypothesis: `max_bin=4096` might squeeze out another split-resolution gain if numeric thresholds remain limiting.
- Classification: follow-up.
- Result: CV AUC 0.7764 +/- 0.0056, discard.
- Observation: Ties 2048 rounded AUC without improvement, so 2048 is the better simplicity point.

## 9b6922a - deeper higher-support trees

- Hypothesis: With better split resolution and no L2, depth 10 might capture additional interactions if leaf support is raised to 80.
- Classification: follow-up.
- Result: CV AUC 0.7724 +/- 0.0058, discard.
- Observation: Raising child weight this far over-constrains the model; depth 10 with such support cannot recover the depth 9 signal.

## 9001593 - depth 10 same child support

- Hypothesis: The depth 10 failure may have been caused by `min_child_weight=80`; keeping child support at 40 can isolate whether extra depth helps.
- Classification: follow-up.
- Result: CV AUC 0.7777 +/- 0.0056, keep.
- Observation: Strong improvement. Extra depth is useful, but doubling child-weight was too conservative.

## e4fdacb - depth 11 trees

- Hypothesis: If depth 10 helps with child support unchanged, depth 11 may capture further useful high-order interactions.
- Classification: follow-up.
- Result: CV AUC 0.7787 +/- 0.0055, keep.
- Observation: Depth continues to help. Runtime remains acceptable but is approaching a point to monitor.

## bef667f - depth 12 trees

- Hypothesis: Depth 12 may continue the high-order interaction trend at the current leaf-support level.
- Classification: follow-up.
- Result: CV AUC 0.7795 +/- 0.0057, keep.
- Observation: Depth still helps, though runtime is rising. The dataset appears to reward very deep supported trees.

## ca42c5c - depth 13 trees

- Hypothesis: Depth 13 may still improve if high-order categorical interactions remain underfit.
- Classification: follow-up.
- Result: CV AUC 0.7798 +/- 0.0056, keep.
- Observation: Positive but smaller gain. Depth trend is tapering; future depth increases need runtime and simplicity scrutiny.

## 9d3e816 - depth 14 trees

- Source: XGBoost tuning docs frame `max_depth` as a bias/variance control; DART research indicates dropout can help some cases but is slower, so continuing the productive depth axis first is cheaper.
- Hypothesis: Depth 14 may still capture sparse high-order categorical interactions before overfitting dominates.
- Classification: follow-up.
- Result: CV AUC 0.7800 +/- 0.0057, keep.
- Observation: Small but positive gain. Depth is near a plateau but has not fully stopped.

## ae01d40 - depth 15 trees

- Hypothesis: Depth 15 might continue the depth trend if sparse high-order interactions remain underfit.
- Classification: follow-up.
- Result: CV AUC 0.7800 +/- 0.0055, discard.
- Observation: Ties depth 14 while adding complexity and runtime, so depth 14 is the better stopping point for this axis.

## fa28534 - relaxed child support at depth 14

- Hypothesis: Depth 14 may be underusing its capacity with `min_child_weight=40`; lowering to 30 could allow more useful sparse interactions while retaining support.
- Classification: follow-up.
- Result: CV AUC 0.7822 +/- 0.0054, keep.
- Observation: Strong gain. The best child-weight level shifted lower at higher depth. Runtime is close to the cap, so future changes must be careful.

## 206181a - lower child support

- Hypothesis: Lowering `min_child_weight` from 30 to 25 may unlock additional useful leaves while staying barely within the runtime cap.
- Classification: follow-up.
- Result: CV AUC 0.7834 +/- 0.0049, keep.
- Observation: Strong gain but runtime is now at the practical limit. Avoid heavier depth/lower-child configurations unless compensating elsewhere.

## bba5906 - shorter deep schedule

- Hypothesis: 700 rounds at `learning_rate=0.017` may preserve total boosting strength while reducing runtime.
- Classification: ablation/simplification.
- Result: CV AUC 0.7831 +/- 0.0051, discard.
- Observation: Faster but lower AUC. The 800-round schedule still matters at the current depth/child-weight setting.

## 6c8284d - lower bin resolution at depth 14

- Hypothesis: With deeper trees, `max_bin=1024` might retain AUC while reducing runtime compared with 2048.
- Classification: ablation/simplification.
- Result: CV AUC 0.7829 +/- 0.0053, discard.
- Observation: Lower bin resolution loses AUC. `max_bin=2048` remains justified despite runtime cost.

## b916663 - route categorical interaction

- Source: BTS documentation identifies airport and carrier operations as delay-related, motivating route-level Origin-Dest structure; XGBoost categorical docs support native categorical partitioning.
- Hypothesis: A `Route` categorical interaction could capture route-specific delay patterns beyond independent Origin and Dest effects.
- Classification: exploration.
- Result: timed out before CV output, crash.
- Observation: The feature likely increases categorical split cost too much for the current near-cap model. Revisit only with a faster base configuration.

## 8ea18eb - numeric calendar features

- Hypothesis: Ordered numeric Month/Day/Week features might expose seasonal structure beyond categorical partitioning with modest overhead.
- Classification: exploration.
- Result: timed out before CV output, crash.
- Observation: The current model is too close to the runtime cap for even lightweight extra features. Further feature tests need a faster configuration first.

## 0d89941 - gentler schedule compression

- Hypothesis: 750 rounds at `learning_rate=0.016` may preserve most of the 800-round model while reducing runtime.
- Classification: ablation/simplification.
- Result: CV AUC 0.7833 +/- 0.0050, discard.
- Observation: Very close but below best. The 800-round schedule still edges it out.

## 65d556c - depth 13 with lower child support

- Hypothesis: Depth 13 with `min_child_weight=25` may retain the lower-child gain while reducing depth/runtime versus depth 14.
- Classification: ablation/simplification.
- Result: CV AUC 0.7830 +/- 0.0052, discard.
- Observation: Below best. Depth 14 contributes real signal when child support is 25.

## 1736b5d - per-node column sampling

- Source: XGBoost random forest docs describe `colsample_bynode` as per-split column sampling, useful for randomization.
- Hypothesis: Per-node sampling may regularize deep trees and reduce split cost without changing depth or child support.
- Classification: exploration.
- Result: CV AUC 0.7815 +/- 0.0055, discard.
- Observation: Lower AUC. The model needs reliable access to all core features at deep split points.

## 7e90b07 - full column exposure

- Hypothesis: Keeping row sampling but using all columns per tree may improve deep interaction discovery.
- Classification: follow-up.
- Result: timed out before CV output, crash.
- Observation: Full column exposure is too expensive at the current depth/bin/child setting. `colsample_bytree=0.85` helps stay under the cap.

## 6cf99c2 - slightly lower row sampling

- Hypothesis: Lowering only `subsample` to 0.8 may improve runtime or regularization without changing feature availability.
- Classification: exploration.
- Result: CV AUC 0.7825 +/- 0.0052, discard.
- Observation: Worse. The current 0.85 row sampling remains better.

## 14ef49e - conservative logistic step

- Source: XGBoost parameter docs mention `max_delta_step` can make logistic updates more conservative, especially for imbalanced cases.
- Hypothesis: A finite step cap might stabilize deep logistic updates even though the slice is balanced.
- Classification: exploration.
- Result: CV AUC 0.7834 +/- 0.0055, discard.
- Observation: Ties best rounded AUC but adds complexity and runtime. Not worth keeping.

## 818d803 - slightly lower child support

- Hypothesis: Lowering `min_child_weight` from 25 to 23 may continue the child-weight gain if runtime allows.
- Classification: follow-up.
- Result: timed out before CV output, crash.
- Observation: The current runtime boundary makes 25 the practical lower child-weight limit at depth 14 and 800 rounds.

## 439dad3 - slightly higher learning rate

- Hypothesis: Keeping 800 rounds but raising `learning_rate` from 0.015 to 0.016 may reduce residual underfitting without changing runtime much.
- Classification: follow-up.
- Result: CV AUC 0.7837 +/- 0.0050, keep.
- Observation: Small gain. The model benefits from slightly stronger updates, though runtime is now extremely close to the cap.

## b78c4ac - higher deep learning rate

- Hypothesis: Raising `learning_rate` to 0.017 may continue reducing residual underfit at the same tree count.
- Classification: follow-up.
- Result: CV AUC 0.7840 +/- 0.0051, keep.
- Observation: Improvement continues. Runtime is on the boundary, so only same-cost parameter changes are reasonable now.

## 355862a - learning rate 0.018

- Hypothesis: A further increase to 0.018 may continue the same-cost underfit reduction.
- Classification: follow-up.
- Result: CV AUC 0.7836 +/- 0.0053, discard.
- Observation: Worse. The local learning-rate optimum is around 0.017 for this 800-round setup.

## c52a0f5 - learning rate 0.0175

- Hypothesis: A midpoint between 0.017 and 0.018 may improve over both.
- Classification: follow-up.
- Result: CV AUC 0.7840 +/- 0.0051, discard.
- Observation: Ties 0.017 rounded AUC without improving simplicity; keep 0.017.

## 3434060 - slightly higher child support

- Hypothesis: Raising `min_child_weight` from 25 to 27 under the stronger learning rate may preserve AUC while reducing variance/runtime.
- Classification: follow-up.
- Result: CV AUC 0.7835 +/- 0.0053, discard.
- Observation: Worse. The current local optimum remains `min_child_weight=25`.

## fa7622d - learning rate 0.0172

- Source: XGBoost docs note learning-rate shrinkage controls update conservativeness; this is same-cost micro-tuning near the current optimum.
- Hypothesis: A small increase over 0.017 may improve more than 0.0175/0.018.
- Classification: follow-up.
- Result: CV AUC 0.7838 +/- 0.0053, discard.
- Observation: Lower than best. The rounded optimum remains exactly 0.017 among tested values.

## 0c1251b - slightly shorter stronger schedule

- Hypothesis: 780 rounds at `learning_rate=0.0174` may preserve total boosting strength while saving runtime.
- Classification: ablation/simplification.
- Result: CV AUC 0.7837 +/- 0.0052, discard.
- Observation: Lower AUC. The full 800 rounds remain useful.

## 1fe0af2 - child support boundary

- Hypothesis: `min_child_weight=24` may improve over 25 while avoiding the timeout seen at 23.
- Classification: follow-up.
- Result: CV AUC 0.7840 +/- 0.0053, discard.
- Observation: Ties best rounded AUC but does not improve it. Keep the existing 25 setting.

## b04ea1a - row sampling 0.875

- Hypothesis: A small increase from `subsample=0.85` to 0.875 may expose enough more rows to improve deep interactions without the timeout risk of full exposure.
- Classification: follow-up.
- Result: CV AUC 0.7844 +/- 0.0053, keep.
- Observation: Good gain. Row sampling optimum shifted upward; column sampling should stay at 0.85 based on prior failures.

## 4d79cc8 - row sampling 0.9

- Hypothesis: If 0.875 improves, 0.9 may further improve row exposure while remaining under the timeout.
- Classification: follow-up.
- Result: CV AUC 0.7847 +/- 0.0054, keep.
- Observation: Another gain. Higher row exposure helps, but runtime remains near the boundary.

## 73a2c7a - row sampling 0.925

- Hypothesis: The row-exposure trend may continue above 0.9.
- Classification: follow-up.
- Result: CV AUC 0.7845 +/- 0.0051, discard.
- Observation: Lower than 0.9. The row-sampling peak is around 0.9.

## 34ae542 - column sampling 0.875

- Hypothesis: Slightly more column exposure may improve interactions now that row sampling is better tuned.
- Classification: follow-up.
- Result: timed out before CV output, crash.
- Observation: Even a small column-sampling increase is too expensive at the current model size. Keep `colsample_bytree=0.85`.

## 93c044d - column sampling 0.825

- Hypothesis: Slightly lower column sampling may regularize and reduce cost while preserving row-sampling gains.
- Classification: follow-up.
- Result: CV AUC 0.7847 +/- 0.0054, discard.
- Observation: Ties best rounded AUC but does not improve it. Keep 0.85.

## 244616f - learning rate with higher row sampling

- Hypothesis: The row-sampling improvement may shift the best learning rate upward to 0.0175.
- Classification: follow-up.
- Result: CV AUC 0.7847 +/- 0.0053, discard.
- Observation: Ties best rounded AUC but does not beat 0.017.

## 9778fb3 - row sampling 0.9125

- Hypothesis: A midpoint between 0.9 and 0.925 might improve over both.
- Classification: follow-up.
- Result: timed out before CV output, crash.
- Observation: 0.9125 exceeds the practical runtime boundary; 0.9 is the best feasible row-sampling setting.
