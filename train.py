import pandas as pd
import numpy as np
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    # Convert HHMM to minutes since midnight
    X["DepMinutes"] = (df["DepTime"] // 100) * 60 + (df["DepTime"] % 100)
    X["DepMinuteOfHour"] = df["DepTime"] % 100
    X["DepTime_sin"] = np.sin(2 * np.pi * X["DepMinutes"] / 1440)
    X["DepTime_cos"] = np.cos(2 * np.pi * X["DepMinutes"] / 1440)
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=24,
    learning_rate=0.025,
    subsample=0.7,
    colsample_bytree=0.5,
    min_child_weight=3,
    enable_categorical=True,
    tree_method="hist",
    max_bin=2048,
    eval_metric="auc",
    random_state=42,
    n_jobs=2,
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

t0 = time.time()
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"CV time: {time.time() - t0:.1f}s")
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

t0 = time.time()
model.fit(X_train, y_train)
print(f"Final model training time: {time.time() - t0:.1f}s")


# 4/5 model: train on the first fold's training split (same as one CV fold)
train_idx_4_5 = list(cv.split(X_train, y_train))[0][0]
X_4_5 = X_train.iloc[train_idx_4_5]
y_4_5 = y_train[train_idx_4_5]

from sklearn.base import clone

model_4_5 = clone(model)
t0 = time.time()
model_4_5.fit(X_4_5, y_4_5)
print(f"4/5 model training time: {time.time() - t0:.1f}s")