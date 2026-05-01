# autoresearch XGBoost

This is an experiment to have an AI/LLM agent conduct autonomous research in optimizing (tuning) XGBoost on a given dataset.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b <tag>`. Do this directly — do NOT run git checkout main or switch branches first. Branch from whatever HEAD is currently at.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - downloading the data. Do not modify.
   - `train.py` - the file you modify. Data preparation, feature engineering, choosing hyperparameters and model training (with possible early stopping etc.).
   - `check_groundtruth.py` - script to check the "ground truth" AUC by the human. Do not access this file.
4. **Verify data exists**: Check that `data-cache/` contains data. If not, tell the human to run `python3 prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

You launch it simply as: `python3 train.py`.

**What you CAN do:**
- Modify `train.py` - this is the only file you edit. Everything is fair game that will lead to a model that generalizes on unseen data: data preparation, feature engineering, choosing hyperparameters, and model training. You can also implement new features such as early stopping etc.
- Search the web and read external resources. This is not optional — you MUST do research before relying solely on your own intuition. See the **Research** section below.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains downloading the data. 
- Install new packages or add dependencies. You can only use what's already installed.
- Change the evaluation. Keep using either (1) the 5-fold cross validation or (2) the evaluation on the given separate dataset (slice) in `train.py` (whichever given) to get the evaluation metric (AUC).
- Modify the evaluation harness. The code in `check_groundtruth.py` is the ground truth metric.
- Don't use any of the data files other than `2005-slice1-100k.csv` for training, and (1) the same dataset for cross-validation or (2) `2006-slice1-100k.csv` for evaluation (whichever is the case).
- Read, run, or reference `run_groundtruth_all.sh`. This is a human-only tool for post-hoc evaluation of experiments against the held-out test set. It is never part of the experiment loop. If you find yourself wanting to use it, stop and tell the human immediately — it means something has gone wrong with your understanding of the task.

## Research

You are expected to actively search the web and read external resources throughout the experiment. Do not treat this as a fallback for when you're stuck — treat it as a primary input to your experiment design.

**When to research:**
- Before your very first non-baseline experiment: search for best practices on tuning XGBoost for binary classification, common feature engineering techniques for tabular data, and known good hyperparameter ranges.
- Every 10 experiments (aligned with your synthesis pause): search for new ideas you haven't tried yet. Look for Kaggle competition write-ups, blog posts, papers, or documentation pages relevant to what you're working on.
- Whenever you hit a plateau (3+ consecutive discards with <0.001 movement): stop and research before trying another random tweak. Look for techniques specifically designed to break through plateaus — different feature transformations, interaction terms, encoding strategies, or XGBoost parameters you haven't explored.
- When trying a new category of change (e.g., first time doing feature engineering, first time tuning categorical handling): read about that specific topic first.

**What to search for:**
- XGBoost documentation for parameters you haven't tried
- "XGBoost tuning guide" / "XGBoost best practices"
- "feature engineering for airline delay prediction" or similar domain-specific resources
- "categorical feature handling XGBoost"
- Kaggle notebooks and competition solutions for similar tabular classification problems
- Academic papers or blog posts on gradient boosting optimization

**How to use what you find:**
- Cite your source when proposing an experiment (e.g., "Based on [XGBoost docs on monotonic constraints], trying...")
- Don't blindly copy — adapt what you read to this specific dataset and problem
- If a source suggests a technique, understand *why* it works before implementing it

**Important:** Research time does not count against the experiment timeout. Take as long as you need to read and understand a resource before designing your next experiment. A well-researched experiment is worth more than three random ones.

**The goal is simple: get the highest AUC.** Everything is fair game that will lead to a model that generalizes on unseen data: data preparation, feature engineering, choosing hyperparameters, and model training. Read XGBoost documentation online, search the web for how to tune XGBoost. Try out adding new elements such early stopping. Be creative! The only constraint is that the code runs without crashing and finishes in reasonable time.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome - that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 AUC improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 AUC improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
CV time: 3.0s
CV AUC: 0.7445 ± 0.0043
Final model training time: 0.3s
4/5 model training time: 0.2s
```

or 

```
Training time: 0.3s
Eval time: 0.1s
Eval AUC: 0.7152
```

depending on the case (cross-validation or separate evaluation dataset).

You can extract the key metric from the log file:

```
grep "^CV AUC:" run.log
```

or

```
grep "^Eval AUC:" run.log
```


## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated - commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	CV_AUC	status	description
```

or

```
commit	Eval_AUC	status	description
```

1. git commit hash (short, 7 chars)
2. CV/Eval AUC achieved (e.g. 0.7300) - use 0.0000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	test_AUC	status	description
a1b2c3d	0.7326	keep	baseline
b2c3d4e	0.7411	keep	increase number of trees
c3d4e5f	0.0000	crash	XGBoost OOM
```

## Research log

Also maintain a research log `research-log.md` with details of your thinking, hypotheses, and observations for each experiment. This helps track your reasoning and decisions over time. Make is so that it can be related to `results.tsv` and the corresponding git commits.


## The experiment loop

The experiment runs on a dedicated branch (e.g. `mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Choose your next experiment deliberately.** Before touching any code:
   - Review `results.tsv` and recent commits.
   - State a short **hypothesis**: what you are changing, why you think it will help, and (if applicable) which prior result motivates this step.
   - Classify the experiment as one of: *follow-up* to a promising result, *ablation/simplification* of a promising result, or *exploration* of a meaningfully different direction.
   - **Do not** run near-duplicate experiments unless you can state exactly what is different and why it matters. Avoid random-walk behavior and cosmetic variations of the same idea.
   - If you haven't done web research in the last 10 experiments, or if your last 3+ experiments were discards, do research now before proposing your next change. See the **Research** section.
3. Tune `train.py` with that experimental idea by directly hacking the code.
4. git commit
5. Run the experiment: `python3 train.py > run.log 2>&1` (redirect everything - do NOT use tee or let output flood your context)
6. Read out the results: `grep "^CV AUC:" run.log` or `grep "^Eval AUC:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If CV/Eval AUC improved (higher), you "advance" the branch, keeping the git commit
10. If CV/Eval AUC is equal or worse, you git reset back to where you started
11. **Every 10 experiments**, pause and briefly synthesize what you have learned so far: what kinds of changes help, what kinds do not, what your current best theory is about what matters on this dataset, and what direction to try next. Write this synthesis as a short note in your context (not a file) to inform subsequent experiments.
The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take <1 minute total (+ a few seconds for startup and eval overhead). If a run exceeds 1 minute, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder - read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~1 minute then you can run approx 60/hour, for a total of about 480 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
