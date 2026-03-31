#!/bin/bash
#
# Evaluate every experiment in results.tsv against the ground truth test set.
#
# Uses a temporary worktree so the main repo is never modified. Safe to kill
# at any point — cleanup removes the worktree automatically.
#
# Usage:
#   ./run_groundtruth_all.sh [results.tsv] [output.tsv] [timeout_seconds]
#
# Defaults:
#   results.tsv      -> results.tsv in the repo root
#   output.tsv       -> groundtruth_all.tsv in the repo root
#   timeout_seconds  -> 180
#
# Prerequisites:
#   - results.tsv must exist with columns: commit, CV_AUC, status, description
#   - check_groundtruth.py must exist in the repo root
#   - The virtual environment (if any) should be activated before running
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

RESULTS_INPUT="${1:-results.tsv}"
OUTPUT_ARG="${2:-groundtruth_all.tsv}"
case "$OUTPUT_ARG" in
  /*) OUTPUT_FILE="$OUTPUT_ARG" ;;
  *)  OUTPUT_FILE="$REPO_ROOT/$OUTPUT_ARG" ;;
esac
TIMEOUT="${3:-180}"

if [ ! -f "$RESULTS_INPUT" ]; then
  echo "ERROR: $RESULTS_INPUT not found. Run experiments first."
  exit 1
fi

if [ ! -f "check_groundtruth.py" ]; then
  echo "ERROR: check_groundtruth.py not found in $REPO_ROOT"
  exit 1
fi

# Create a temporary worktree so we never touch the main repo
WORKTREE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gt-worktree-XXXXXX")
WORKTREE_BRANCH="gt-eval-$$"

cleanup() {
  echo ""
  echo "Cleaning up worktree..."
  git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
  git -C "$REPO_ROOT" branch -D "$WORKTREE_BRANCH" 2>/dev/null || true
  rm -rf "$WORKTREE_DIR" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

echo "Creating temporary worktree at $WORKTREE_DIR ..."
git worktree add -b "$WORKTREE_BRANCH" "$WORKTREE_DIR" HEAD --quiet

# Symlink the data-cache into the worktree so we don't copy gigabytes
if [ -d "$REPO_ROOT/data-cache" ]; then
  rm -rf "$WORKTREE_DIR/data-cache"
  ln -s "$REPO_ROOT/data-cache" "$WORKTREE_DIR/data-cache"
fi

# Write header
echo -e "commit\tstatus\tdescription\tcv_auc\ttest_auc\tbootstrap_auc\tbootstrap_ci" > "$OUTPUT_FILE"

# Read results.tsv, skip header
LINE_NUM=0
tail -n +2 "$RESULTS_INPUT" | while IFS=$'\t' read -r commit cv_auc status description; do
  LINE_NUM=$((LINE_NUM + 1))

  # Skip empty lines
  [ -z "$commit" ] && continue

  echo "=== [$LINE_NUM] $commit ($description) ==="

  # Check if commit exists in git
  if ! git cat-file -e "$commit" 2>/dev/null; then
    echo "  SKIP: commit $commit not found in git"
    echo -e "$commit\t$status\t$description\t$cv_auc\tN/A\tN/A\tcommit not found" >> "$OUTPUT_FILE"
    continue
  fi

  # Extract that commit's train.py into the worktree
  if ! git show "$commit:train.py" > "$WORKTREE_DIR/train.py" 2>/dev/null; then
    echo "  SKIP: could not extract train.py from $commit"
    echo -e "$commit\t$status\t$description\t$cv_auc\tN/A\tN/A\textract failed" >> "$OUTPUT_FILE"
    continue
  fi

  # Run check_groundtruth.py in the worktree
  GT_LOG="$WORKTREE_DIR/gt_run.log"
  if ! timeout "$TIMEOUT" python3 "$WORKTREE_DIR/check_groundtruth.py" > "$GT_LOG" 2>&1; then
    echo "  CRASH/TIMEOUT"
    echo -e "$commit\t$status\t$description\t$cv_auc\t0.0000\t0.0000\tcrash/timeout" >> "$OUTPUT_FILE"
    continue
  fi

  # Extract results from log
  CV_AUC=$(grep "^CV AUC:" "$GT_LOG" | sed 's/CV AUC: //' | cut -d' ' -f1)
  TEST_AUC=$(grep "^Test AUC:" "$GT_LOG" | sed 's/Test AUC: //')
  BOOT_AUC=$(grep "^Bootstrap AUC:" "$GT_LOG" | sed 's/Bootstrap AUC: //' | cut -d' ' -f1)
  BOOT_CI=$(grep "^Bootstrap AUC:" "$GT_LOG" | grep -o '\[.*\]')

  echo "  CV=$CV_AUC  Test=$TEST_AUC  Bootstrap=$BOOT_AUC $BOOT_CI"
  echo -e "$commit\t$status\t$description\t$cv_auc\t$TEST_AUC\t$BOOT_AUC\t$BOOT_CI" >> "$OUTPUT_FILE"
done

echo ""
echo "=== DONE: results written to $OUTPUT_FILE ==="
