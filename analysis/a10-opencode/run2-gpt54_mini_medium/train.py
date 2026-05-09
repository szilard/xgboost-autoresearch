import numpy as np
import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )

    # Cyclical encodings for periodic time fields.
    month = pd.to_numeric(df["Month"].str.replace("c-", "", regex=False), errors="coerce")
    day_of_week = pd.to_numeric(df["DayOfWeek"].str.replace("c-", "", regex=False), errors="coerce")
    dep_time = pd.to_numeric(df["DepTime"], errors="coerce")
    dep_minutes = (dep_time // 100) * 60 + (dep_time % 100)

    X["Month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["Month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["DayOfWeek_sin"] = np.sin(2 * np.pi * (day_of_week - 1) / 7)
    X["DayOfWeek_cos"] = np.cos(2 * np.pi * (day_of_week - 1) / 7)
    X["DepTime_sin"] = np.sin(2 * np.pi * dep_minutes / (24 * 60))
    X["DepTime_cos"] = np.cos(2 * np.pi * dep_minutes / (24 * 60))

    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 18,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "lambda": 1.0,
    "alpha": 0.0,
    "gamma": 0.1,
    "max_cat_to_onehot": 32,
    "max_bin": 1024,
    "tree_method": "hist",
    "seed": 42,
    "nthread": -1,
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

t0 = time.time()
scores = []
best_iters = []
for train_idx, valid_idx in cv.split(X_train, y_train):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_valid = X_train.iloc[valid_idx]
    y_fold_valid = y_train[valid_idx]

    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_fold_valid, label=y_fold_valid, enable_categorical=True)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    scores.append(roc_auc_score(y_fold_valid, booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))))
    best_iters.append(booster.best_iteration + 1)

print(f"CV time: {time.time() - t0:.1f}s")
print(f"CV AUC: {pd.Series(scores).mean():.4f} ± {pd.Series(scores).std():.4f}")

t0 = time.time()
best_rounds = int(pd.Series(best_iters).median())
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
xgb.train(params, dtrain, num_boost_round=best_rounds, verbose_eval=False)
print(f"Final model training time: {time.time() - t0:.1f}s")
t0 = time.time()
train_idx_4_5 = list(cv.split(X_train, y_train))[0][0]
X_4_5 = X_train.iloc[train_idx_4_5]
y_4_5 = y_train[train_idx_4_5]
dtrain_4_5 = xgb.DMatrix(X_4_5, label=y_4_5, enable_categorical=True)
xgb.train(params, dtrain_4_5, num_boost_round=best_rounds, verbose_eval=False)
print(f"4/5 model training time: {time.time() - t0:.1f}s")
