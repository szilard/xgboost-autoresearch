import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CATEGORICAL_COLS = [
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "UniqueCarrier",
    "Origin",
    "Dest",
]
NUMERIC_COLS = ["DepTime", "Distance"]
TARGET = "dep_delayed_15min"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Target: Y/N -> 1/0.
    df[TARGET] = (df[TARGET] == "Y").astype(np.int8)
    # Cast categoricals so XGBoost's native categorical path can pick them up.
    # `c-1`..`c-12` etc. stay as strings inside the category dtype — XGBoost
    # only needs the column dtype to be `category`, the underlying codes are
    # integers it can partition on.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    # Light feature engineering that's cheap and domain-motivated.
    # DepTime is HHMM (e.g. 1357 = 13:57); a tree can learn this from the raw
    # int, but giving it `hour` as a direct numeric feature is a small freebie
    # and a natural hook for later cyclical encoding experiments.
    df["hour"] = (df["DepTime"] // 100).clip(0, 23).astype(np.int16)
    return df


def split_xy(df: pd.DataFrame):
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + ["hour"]
    X = df[feature_cols]
    y = df[TARGET].values
    return X, y, feature_cols


def build_model(scale_pos_weight: float, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        # --- Objective ---
        objective="binary:logistic",
        eval_metric="auc",

        # --- Tree method + native categorical ---
        tree_method="hist",
        enable_categorical=True,
        # Features with <= this many levels get one-hot style splits; larger
        # ones (Origin/Dest) get partition splits. Default 4 is fine; raise
        # this if you suspect low-cardinality categoricals are being over-split.
        max_cat_to_onehot=4,

        # --- Core GBDT hyperparameters (baseline) ---
        learning_rate=0.1,
        max_depth=8,
        n_estimators=2000,          # upper bound; early stopping decides actual
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,

        # --- Class imbalance ---
        scale_pos_weight=scale_pos_weight,

        # --- Early stopping ---
        early_stopping_rounds=50,

        # --- Misc ---
        n_jobs=-1,
        random_state=random_state,
        verbosity=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", type=Path)
    parser.add_argument("--valid-frac", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    print(f"[load] reading {args.data}")
    df = load(args.data)
    print(f"[load] rows={len(df):,}  positive_rate={df[TARGET].mean():.4f}")

    X, y, feature_cols = split_xy(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_frac,
        random_state=args.seed,
        stratify=y,
    )
    # Rebuild category dtypes on the split frames so both share the same
    # category set (XGBoost requires consistent categories between train and
    # eval/test).
    for col in CATEGORICAL_COLS:
        unified = pd.api.types.union_categoricals(
            [X_train[col], X_valid[col]]
        ).categories
        X_train[col] = pd.Categorical(X_train[col], categories=unified)
        X_valid[col] = pd.Categorical(X_valid[col], categories=unified)

    # scale_pos_weight: only apply if the training set is meaningfully
    # imbalanced (neg/pos > ~3). Under that, it rarely helps AUC and can hurt
    # probability calibration. See research.md §5.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = neg / max(pos, 1)
    spw = ratio if ratio > 3.0 else 1.0
    print(f"[class] train neg={neg:,} pos={pos:,}  neg/pos={ratio:.3f}  "
          f"scale_pos_weight={spw:.3f}")

    model = build_model(scale_pos_weight=spw, random_state=args.seed)

    print("[fit] training with early stopping on validation AUC")
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )
    fit_secs = time.time() - t0

    best_iter = getattr(model, "best_iteration", None)
    valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    print(f"[fit] done in {fit_secs:.1f}s  best_iter={best_iter}  valid_auc={auc:.4f}")

    # Feature importance (gain). Good first cut for feature-engineering ideas.
    importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    print("\n[importance] gain-based feature importances:")
    for name, val in importances.items():
        print(f"  {name:<16} {val:.4f}")
