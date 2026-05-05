# Research Log — `may4` run

## Setup notes

- Branch: `may4` from commit `0841565`
- Eval: 5-fold stratified CV on `2005-slice1-100k.csv`, metric = ROC AUC
- Run command: `.venv/bin/python3 train.py > run.log 2>&1` (system `python3` lacks deps)

## Initial research (before first non-baseline experiment)

Sources reviewed: XGBoost docs, Analytics Vidhya tuning guide, szilard/benchm-ml, szilard/GBM-perf, datascience.la GBM benchmark talks, UC Berkeley Air Travel Delay project, Nature Sci Reports hybrid ML model.

Key takeaways:
- Published ceiling on this 100k airline slice: ~0.749 AUC (Pafka's "best config": n_trees=1000, lr=0.01, depth=16, subsample=0.5, min_child_weight=1).
- Deep trees (depth 12-16) outperform shallow on this dataset — counter to common advice.
- High-impact feature engineering: `hour = DepTime // 100`, `route = Origin_Dest`, time-of-day buckets, hub/busy-hour flags, Month×DayOfWeek interactions.
- Note re: `enable_categorical=True` — modern native handling may already lift us above 1-hot-encoded benchm-ml numbers (our baseline 0.7445 at n=30 already exceeds Pafka's "fast config" 0.735 at n=300 with 1-hot).
- Big headroom on `n_estimators` — baseline at 30 is severely undertrained. CV time 0.6s leaves plenty of compute.

## Experiments

### Exp 1 — `0841565` — baseline — **0.7445 ± 0.0043** — keep
Out-of-the-box `train.py`. n_estimators=30, max_depth=6, learning_rate=0.1, enable_categorical=True. CV time 0.6s, very fast. Establishes the floor.

### Exp 2 — `2fc7171` — n_estimators 30→300 — **0.7548 ± 0.0053** — keep
*Hypothesis*: 30 trees @ lr=0.1 is severely undertrained; published "fast config" is 300/0.1/d6. Single isolated lever — only n_estimators changed.
*Outcome*: +0.0103 AUC. Already beats Pafka's published 1-hot fast config (~0.735), confirming native categorical handling adds value. Std rose slightly (0.0043 → 0.0053) — slight uptick in variance but well within noise. CV time 4.5s.
*Read*: Lots of headroom remains. Next move: keep pushing n_estimators or start exploring depth.

### Exp 3 — `cb1a41e` — n_estimators 300→1000 — **0.7583 ± 0.0049** — keep
*Hypothesis*: still on the rising part of the n_estimators curve; same lr=0.1, isolated lever. If gains continue, we're not yet overfitting at this lr.
*Outcome*: +0.0035 AUC. Diminishing returns kicking in (was +0.0103 from 30→300). Std stable. CV time 14.8s.
*Read*: lr=0.1 isn't overfitting yet at 1000 trees on this data. Two productive directions remain: (a) deeper trees (Pafka's published finding: depth 12-16 wins), (b) lower lr with more trees. Try depth next as an independent lever.

### Exp 4 — `5ff3bf0` — max_depth 6→10 — **0.7693 ± 0.0044** — keep
*Hypothesis*: Pafka's published finding — deep trees (12-16) outperform shallow on airline. Test the trend with a moderate jump to depth 10.
*Outcome*: +0.0110 AUC, biggest single move yet. Std actually fell (0.0049 → 0.0044) — deep trees more stable here. CV time 41.7s.
*Read*: Confirms deep-tree win. We're now ~0.005 below the published ceiling (~0.749). Time budget tightening — pushing depth further (12-16) may exceed 60s with n=1000. Cleanest next isolated move: lower lr (0.1→0.05) with same n=1000 to extract more from the now-larger trees without blowing time budget.

### Exp 5 — `37ba671` — learning_rate 0.1→0.05 — **0.7702 ± 0.0043** — keep
*Hypothesis*: with depth=10 trees being more powerful per-step, lr=0.1 may be overshooting; smaller steps should let n=1000 trees converge more cleanly.
*Outcome*: +0.0009 — marginal but positive, std slightly tighter. CV 46.2s.
*Read*: We've now passed Pafka's published 100k 1-hot ceiling (~0.749), consistent with `enable_categorical=True` adding value. From here, gains will get smaller per single-lever change. Time to try feature engineering.

### Exp 6 — `8551218` — add Hour = DepTime // 100 — **0.7699 ± 0.0049** — discard
*Hypothesis*: research suggested hour-of-day is "single most useful derived feature" for airline delay; explicit Hour feature may give trees a cleaner split candidate than raw HHMM DepTime.
*Outcome*: -0.0003 AUC, std slightly up. Reset to `37ba671`.
*Read*: Hour = DepTime // 100 is a strict coarsening of DepTime — XGBoost can already replicate it via splits on DepTime. Adding it as a separate feature gave no new information and added a dimension. Lesson: redundant numeric derivations of existing numerics likely won't help here. Try a *genuinely new* feature next, e.g. `route = Origin_Dest` (high-cardinality categorical that captures route-specific patterns), or move to other regularization levers (subsample, mcw, depth=12).

### Exp 7 — `2f417c5` — add Route = Origin_Dest categorical — **0.7451 ± 0.0033** — discard (also 98s timeout)
*Hypothesis*: a high-cardinality "genuinely new" categorical (Origin × Dest) should help where Hour didn't, by giving trees a direct route-level split candidate.
*Outcome*: AUC collapsed -0.025 to 0.7451; CV time 98s, exceeds 60s budget. Reset to `37ba671`.
*Read*: Two failure modes compounded — (1) ~5000 unique Route levels with native categorical handling caused overfitting on rare routes (many routes have <5 samples in 100k), badly hurting generalization; (2) the high-cardinality categorical also dramatically slowed XGBoost's tree construction. Lesson: at depth=10, XGBoost is already capturing Origin×Dest interactions implicitly. Adding the explicit feature added overfit risk without info. Avoid raw high-cardinality categorical FE here. **Pivot from FE to regularization/hyperparams**: try subsample (Pafka's "best" used 0.5) — adds regularization AND speeds up training (half the data per tree).

### Exp 8 — `f453d36` — subsample=0.5 — **0.7484 ± 0.0040** — discard
*Hypothesis*: regularization via row sub-sampling, paired with our depth=10/lr=0.05/n=1000 setup. Pafka's "best" config used 0.5.
*Outcome*: -0.022 AUC, CV 52.8s. Reset to `37ba671`.
*Read*: subsample=0.5 doesn't transplant cleanly from Pafka's recipe (which had depth=16 and lr=0.01 — much different learning dynamics). In our regime the under-data-per-tree hurts more than the regularization helps. **Lesson on isolated lever**: when params are coupled (subsample × n_estimators × lr × depth all interact for "effective signal per tree"), a single-lever change breaks the optimum. Future regularizers to test: colsample_bytree (different mechanism — feature subset per tree, not row subset), gamma, mcw — each smaller in scope.

### Exp 9 — `1d77ee2` — max_depth 10→12 — **timeout** — crash/discard
*Hypothesis*: push capacity toward Pafka's depth=16 sweet spot. Single isolated lever.
*Outcome*: didn't finish in 70s; would exceed 60s budget. Reset.
*Read*: To go deeper we must trade off n_estimators or use other speedup. Defer; revisit after extracting cheaper wins.

### Exp 10 — `75a453d` — colsample_bytree=0.8 — **0.7799 ± 0.0048** — keep
*Hypothesis*: column subsampling (different from row subsampling) regularizes via feature-level diversity; fast, classic XGBoost lever.
*Outcome*: +0.0097 AUC — biggest win since depth 6→10. CV 41.7s, actually faster than baseline (fewer features per split eval). Std barely moved.
*Read*: A different regularization mechanism than subsample=0.5 produces the OPPOSITE result. Trees in this regime were over-relying on a few dominant features (DepTime, Origin, Dest); forcing feature diversity gives a stronger ensemble. **Direction confirmed**: more colsample regularization may stack; combining with capacity (depth) may also stack.

## Synthesis at exp 10

**What works (kept changes, ordered by impact)**:
- max_depth 6→10: **+0.0110**
- n_estimators 30→300: **+0.0103**
- colsample_bytree=0.8: **+0.0097**
- n_estimators 300→1000: **+0.0035**
- learning_rate 0.1→0.05: **+0.0009**

Total: 0.7445 → 0.7799 (+0.0354).

**What doesn't work and why**:
- Hour = DepTime//100: redundant with DepTime — trees already learn the splits
- Route = Origin_Dest: high-cardinality categorical → overfitting on rare routes + 2x slowdown
- subsample=0.5: under-trains in our high-effective-signal regime (lr=0.05, n=1000)
- max_depth=12 with n=1000: too slow (>70s)

**Current theory of what matters**:
1. *Capacity wins*: more trees, deeper trees (up to a point), enough effective rounds (n×lr).
2. *Feature-level regularization helps; row-level doesn't*: at 100k balanced, each tree already has plenty of data — the issue is not over-fitting to specific samples but to specific dominant features.
3. *Generic FE on existing columns is unproductive at this depth*: depth=10 trees can already learn most coarsenings/interactions implicitly. Only fundamentally new info would help — but our features (Month, DOW, Carrier, Origin, Dest, DepTime, Distance) cover most of what's available without external joins.
4. *We've already exceeded Pafka's published 100k 1-hot ceiling (~0.749) by ~0.03.* Native categorical handling is the main multiplier here.

**Direction for next 10**:
- Stack colsample_bytree further (0.7, 0.6) — maybe more is more
- Push depth=12 again, paired with reduced n_estimators to fit time budget
- Try gamma (different reg mechanism — split-gain threshold)
- Try max_bin (numeric feature quantization granularity)
- Larger n_estimators if depth stays at 10 and time allows — maybe 1500 or 2000

## Exps 11-46 (highlights only — see results.tsv for full ledger)

### Wins
- Exp 11 colsample_bytree 0.8→0.7 → 0.7836 (+0.0037)
- Exp 12 colsample_bytree 0.7→0.6 → 0.7841 (+0.0005, plateau at 0.6)
- Exp 15 max_depth 10→11 → 0.7879 (+0.0038)
- Exp 16 add min_child_weight=5 → 0.7885 (+0.0006, also frees time)
- Exp 17 max_depth 11→12 → 0.7905 (+0.0020)
- Exp 20 add gamma=0.3 → 0.7907 (+0.0002, halves CV time)
- Exp 22 add max_bin=512 → 0.7923 (+0.0016)
- Exp 23 max_bin 512→1024 → 0.7927 (+0.0004)
- Exp 25 max_depth 12→13 → 0.7946 (+0.0019)
- Exp 26 max_depth 13→14 → 0.7966 (+0.0020)
- Exp 27 max_depth 14→15 → 0.7979 (+0.0013)
- Exp 28 max_depth 15→16 → 0.7988 (+0.0009)
- Exp 29 max_depth 16→20 → 0.8007 (+0.0019, **crossed 0.80**)
- Exp 30 max_depth=0 (unlimited) → 0.8016 (+0.0009)
- Exp 33 mcw 5→3 → 0.8045 (+0.0029)
- Exp 34 drop mcw (default 1) → **0.8068** (+0.0023, code simpler)
- Exp 44 gamma 0.3→0.35 → **0.8072** (+0.0004)

### Losses (illustrative; full list in results.tsv)
- Lower lr (0.04, 0.045): timeouts — converging at lower rate needs more rounds
- Higher colsample (0.7, full): timeouts — more features per tree blows compute
- subsample anywhere (0.5, 0.9): under-trains in this regime
- colsample_bynode added with bytree=0.6: over-regularizes (-0.012)
- gamma 0.2 / 0.4: 0.35 is the sweet spot
- max_bin > 1024: regresses
- n_estimators > 1000 (1500 etc): no help, trees converge before then
- DepTime sin/cos cyclical: timeouts (continuous numerics push over budget)
- Target encoding (Origin_te, Dest_te) per-fold OOF: -0.014 AUC AND >80s total — native cat handling already does target-aware splitting; explicit TE hurts and is slower
- Early stopping (90/10 inner val): -0.005 AUC — 10% lost training data hurts more than per-fold optimal-n helps; trees in our regime already converge to a good point at fixed n=1000

## Synthesis at exp 46

**Current best (`d910675`)**: 0.8072 ± 0.0039, CV time ~44s.
```python
xgb.XGBClassifier(
    n_estimators=1000, max_depth=0, learning_rate=0.05,
    colsample_bytree=0.6, gamma=0.35, max_bin=1024,
    enable_categorical=True, random_state=42, n_jobs=-1,
)
```
Total: 0.7445 → 0.8072 = **+0.0627** over baseline. Far above Pafka's published 100k 1-hot ceiling (~0.749).

**Time budget is the binding constraint now.** At 44s we have ~13s of headroom; most "add capacity" levers blow >60s, while most time-saving levers cost AUC. We're locally optimum in the cheap-tweak landscape.

**What I'd need to keep gaining**:
1. Major structural: per-fold OOF target encoding *paired with* dropping native cat handling on Origin/Dest (so total compute doesn't grow). Risky — hurt AUC in raw form.
2. More compute / relaxed time budget would unlock: depth pushes, larger n, more features, multi-seed ensembling.
3. External features (TailNum lineage / weather / etc.) — out of scope per program rules (only the given columns are usable).

**For more loop iterations**: small local-search moves around the current optimum. Expected gains ≤ +0.001 per attempt.

## Exp 49 — IMPORTANT: seed sensitivity finding

`a7984ab` — random_state 42 → 7 — **0.7985 ± 0.0040** vs the seed=42 result of 0.8072. Same code, only the model's `random_state` changed.

**Implication**: with `colsample_bytree=0.6`, the model's stochasticity gives ~0.009 AUC variance across seeds. Many of our recent "wins" of +0.001 to +0.003 are within that noise band. The reported 0.8072 is partially seed-luck, and the true expected CV AUC across seeds is probably closer to ~0.80.

**Why we still report 0.8072**: program.md fixes the seed at 42 and CV is the metric. The number is reproducible with that seed. The real check is the held-out 2006 ground-truth eval, which will show the actual generalization.

**Why we didn't ensemble seeds**: 3-seed averaging at full config = 132s — over the 60s budget. Smaller per-seed models (e.g. n=300, lr=0.13) might fit but each is significantly weaker; the ensemble may not beat the single tuned 0.8072 model. Untested but plausible direction if budget relaxes.

---

# Final summary (53 experiments, branch `may4`)

## Result

- **CV AUC = 0.8072 ± 0.0039** (commit `d910675`, seed=42), CV time ~44s, total run ~58-62s.
- **Net improvement: +0.0627** over the baseline (0.7445 → 0.8072).
- **+0.058 above Pafka's published 100k 1-hot ceiling** (~0.749 from szilard/benchm-ml). Native `enable_categorical=True` is the structural difference; the rest is hyperparameter tuning.

## Final config

```python
xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=0,           # unlimited; bounded by gamma + mcw=1
    learning_rate=0.05,
    colsample_bytree=0.6,
    gamma=0.35,
    max_bin=1024,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
```
Dataset: 5-fold stratified CV on `2005-slice1-100k.csv`. Features kept as the prepare.py defaults — no feature engineering retained (every FE attempt failed; see below).

## Experiment ledger (53 total)

23 keeps, 19 discards, 11 timeouts. Full ledger in `results.tsv`. AUC trajectory:
- 0.7445 (baseline) → 0.7548 → 0.7583 → 0.7693 → 0.7702 (n_estimators + depth=10 + lr=0.05)
- → 0.7799 → 0.7836 → 0.7841 (colsample_bytree sweep, optimum at 0.6)
- → 0.7879 → 0.7885 → 0.7905 (depth 11→12 + add mcw=5)
- → 0.7907 → 0.7923 → 0.7927 (gamma=0.3 + max_bin sweep, optimum 1024)
- → 0.7946 → 0.7966 → 0.7979 → 0.7988 → 0.8007 → 0.8016 (depth 13→14→15→16→20→∞; **crossed 0.80 at depth=20**)
- → 0.8045 → 0.8068 (mcw 5→3→default 1)
- → **0.8072** (gamma 0.3 → 0.35)

## What the data says about this dataset

1. **Capacity wins**: more trees, very deep trees (depth ∈ [16, ∞]), lots of histogram bins (1024). Pafka's published "deep trees beat shallow on airline" finding holds and *strengthens* with native categorical handling.
2. **Feature-level reg helps; row-level reg hurts.** `colsample_bytree=0.6` was a +0.0097 win. *Every* `subsample` value tried (0.5, 0.8, 0.9, 0.95) under-trained. With 100k balanced samples, trees aren't over-fitting to specific rows — they're over-relying on a few dominant features (DepTime, Origin, Dest), and forcing per-tree feature diversity diversifies the ensemble.
3. **Feature engineering on these columns is unproductive.** Trees with depth=∞ already learn everything coarsenings/interactions of the 8 raw features can give. Hour was redundant with DepTime; Route was redundant with Origin×Dest splits AND high-cardinality enough to overfit; cyclical sin/cos of DepTime didn't fit in time and theoretically adds little to a tree splitter; even per-fold OOF target encoding of Origin/Dest *hurt* AUC by 0.014 — native categorical handling already does target-aware grouping at split time, more efficiently.
4. **`gamma=0.35` is the structural unlock for time budget.** Without it, depth=12 alone times out at n=1000. With it, trees self-prune, fitting depth=∞ in 44s. gamma also lifts AUC slightly (sweet spot is narrow — 0.2 and 0.4 both regress).
5. **Early stopping with inner-val didn't help.** -0.005 AUC. The 10% inner val carved from training data costs more than per-fold optimal-`n` saves; trees in our regime already converge by ~round 800 even at fixed n=1000.

## Important caveat: seed sensitivity

Exp 49 changed *only* the model's `random_state` (42 → 7) and AUC dropped to 0.7985 — a **0.0087 swing across seeds**. With `colsample_bytree=0.6`'s stochastic feature subsampling, that's the noise floor. **Implications**:
- Many of the small wins in this log (+0.001 to +0.003) are within seed noise.
- True expected CV AUC across seeds is probably ~0.80, not 0.807.
- The held-out 2006 ground-truth eval is the real test; the seed=42 CV number is reproducible but partially seed-luck.
- Multi-seed ensembling would tighten this — but at the budget, each seeded model has to be small enough that the ensemble may not beat the single tuned 0.8072 model. Untested.

## Time budget is the binding constraint

At 44s base, ~13s of headroom. Most "add capacity" levers (lower lr, more features, depth+13, lossguide, looser colsample, almost any subsample) push past 60s. The few capacity additions that *did* fit (depth pushes past 12, max_bin=1024) only fit because `gamma=0.35` self-prunes weak splits. We're locally optimum in the cheap-tweak landscape.

## Untried directions if budget relaxes

1. Multi-seed ensembling with trimmed per-seed models (3 seeds × n=400 lr=0.075 ≈ budget). Would replace the seed=42 number with a more stable average.
2. Replace native categorical handling on Origin/Dest with target encoding (untried in *replace* form — only added alongside).
3. External features (TailNum lineage, weather, prior-flight delay) — out of scope per program rules.
4. Lower lr (0.04, 0.03) with much higher n — needs faster training to fit time, probably needs a relaxed budget.
