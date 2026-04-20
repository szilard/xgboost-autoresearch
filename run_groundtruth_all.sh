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
#   timeout_seconds  -> 3000
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
TIMEOUT="${3:-3000}"

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
echo -e "commit\tstatus\tdescription\tcv_auc\ttest_auc_2005s2_full\ttest_auc_2005s2_4_5\ttest_auc_2006_full" > "$OUTPUT_FILE"

# Read results.tsv into an array first (avoid pipe + while issues)
LINE_NUM=0
while IFS=$'\t' read -r commit cv_auc status description; do
  LINE_NUM=$((LINE_NUM + 1))

  # Skip header
  [ "$LINE_NUM" -eq 1 ] && continue

  # Skip empty lines
  [ -z "$commit" ] && continue

  echo "=== [$LINE_NUM] $commit | $status | $description ==="

  # Skip crashed experiments
  if [ "$cv_auc" = "0.0000" ]; then
    echo "  SKIP: crashed experiment (CV AUC=0.0000)"
    echo -e "$commit\t$status\t$description\t$cv_auc\tN/A\tN/A\tN/A" >> "$OUTPUT_FILE"
    continue
  fi

  # Skip discarded experiments (commits were reverted, won't exist in git)
  if [ "$status" = "discard" ]; then
    echo "  SKIP: discarded experiment"
    echo -e "$commit\t$status\t$description\t$cv_auc\tN/A\tN/A\tN/A" >> "$OUTPUT_FILE"
    continue
  fi

  # Check if commit exists in git
  if ! git cat-file -e "$commit" 2>/dev/null; then
    echo "  SKIP: commit $commit not found in git (discarded experiment)"
    echo -e "$commit\t$status\t$description\t$cv_auc\tN/A\tN/A\tN/A" >> "$OUTPUT_FILE"
    continue
  fi

  # Extract that commit's train.py into the worktree
  if ! git show "$commit:train.py" > "$WORKTREE_DIR/train.py" 2>/dev/null; then
    echo "  SKIP: could not extract train.py from $commit"
    echo -e "$commit\t$status\t$description\t$cv_auc\tN/A\tN/A\tN/A" >> "$OUTPUT_FILE"
    continue
  fi

  # Copy check_groundtruth.py into worktree (it uses __file__ to find train.py)
  cp "$REPO_ROOT/check_groundtruth.py" "$WORKTREE_DIR/check_groundtruth.py"

  # Run check_groundtruth.py in the worktree
  GT_LOG="$WORKTREE_DIR/gt_run.log"
  if ! timeout "$TIMEOUT" python3 "$WORKTREE_DIR/check_groundtruth.py" > "$GT_LOG" 2>&1; then
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 137 ] || [ "$EXIT_CODE" -eq 139 ]; then
      echo "  CRASH: OOM or segfault (exit $EXIT_CODE)"
    elif [ "$EXIT_CODE" -eq 124 ]; then
      echo "  TIMEOUT: exceeded ${TIMEOUT}s"
    else
      echo "  CRASH: exit code $EXIT_CODE"
      tail -5 "$GT_LOG" 2>/dev/null || true
    fi
    echo -e "$commit\t$status\t$description\t$cv_auc\tCRASH\tCRASH\tCRASH" >> "$OUTPUT_FILE"
    continue
  fi

  # Extract results from log — format is "Test AUC (label): 0.XXXX"
  AUC_2005S2_FULL=$(grep "^Test AUC (full model - eval 2005 slice 2):" "$GT_LOG" | grep -oP '[\d.]+$' || echo "N/A")
  AUC_2005S2_4_5=$(grep "^Test AUC (4/5 model - eval 2005 slice 2):" "$GT_LOG" | grep -oP '[\d.]+$' || echo "N/A")
  AUC_2006_FULL=$(grep "^Test AUC (full model - eval 2006):" "$GT_LOG" | grep -oP '[\d.]+$' || echo "N/A")

  echo "  2005s2_full=$AUC_2005S2_FULL  2005s2_4/5=$AUC_2005S2_4_5  2006=$AUC_2006_FULL"
  echo -e "$commit\t$status\t$description\t$cv_auc\t$AUC_2005S2_FULL\t$AUC_2005S2_4_5\t$AUC_2006_FULL" >> "$OUTPUT_FILE"
done < "$RESULTS_INPUT"

echo ""
echo "=== DONE: results written to $OUTPUT_FILE ==="
