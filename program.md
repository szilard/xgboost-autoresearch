# autoresearch XGBoost

This is an experiment to have an AI/LLM agent conduct autonomous research in optimizing (tuning) XGBoost on a given dataset.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b <tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - downloading the data. Do not modify.
   - `train.py` - the file you modify. Data preparation, feature engineering, choosing hyperparameters and model training (with possible cross validation and/or early stopping etc.).
   - `evaluate.py` - reads the test dataset and runs the evaluation
4. **Verify data exists**: Check that `data-cache/` contains data. If not, tell the human to run `python3 prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

You launch it simply as: `python3 train.py`.

**What you CAN do:**
- Modify `train.py` - this is the only file you edit. Everything is fair game: data preparation, feature engineering, choosing hyperparameters, and model training (with possible cross validation and/or early stopping etc.).
- Read XGBoost documentation online, search the web for how to tune XGBoost hyperparameters etc. and use that information for your experiments.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains downloading the data. 
- Install new packages or add dependencies. You can only use what's already installed.
- Modify the evaluation harness. The code in `evaluate.py` is the ground truth metric.
- Don't use any of the data files other than `airline-1m-slice100k-1.csv` for training and `airline-1m-slice100k-2.csv` for evaluating (only via the `evaluate.py` script).

**The goal is simple: get the highest test AUC.** Everything is fair game: data preparation, feature engineering, choosing hyperparameters, and model training (with possibly adding cross validation and/or early stopping etc.). The only constraint is that the code runs without crashing and finishes in reasonable time.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome - that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 AUC improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 AUC improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
Training time: 0.3s
Test AUC: 0.7300
```

You can extract the key metric from the log file:

```
grep "Test AUC:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated - commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	test_AUC	status	description
```

1. git commit hash (short, 7 chars)
2. test AUC achieved (e.g. 0.7300) - use 0.0000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	test_AUC	status	description
a1b2c3d	0.7326	keep	baseline
b2c3d4e	0.7411	keep	increase number of trees
c3d4e5f	0.0000	crash	XGBoost OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python3 train.py > run.log 2>&1` (redirect everything - do NOT use tee or let output flood your context)
5. Read out the results: `grep "^Test AUC:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If test AUC improved (higher), you "advance" the branch, keeping the git commit
9. If test AUC is equal or worse, you git reset back to where you started
The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take <1 minute total (+ a few seconds for startup and eval overhead). If a run exceeds 2 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder - read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~1 minute then you can run approx 60/hour, for a total of about 480 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
