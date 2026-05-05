import numpy as np
import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


_train_dh = (train["DepTime"] // 100).astype("int32")
train["CarrierHour"] = train["UniqueCarrier"].astype(str) + "_" + _train_dh.astype(str)
cat_cols_all = cat_cols + ["CarrierHour"]
cat_levels = {col: sorted(train[col].unique()) for col in cat_cols_all}

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    dep_hour = (df["DepTime"] // 100).astype("int32")
    X["DepHour"] = dep_hour
    X["DepMinute"] = (df["DepTime"] % 100).astype("int32")
    X["CarrierHour"] = df["UniqueCarrier"].astype(str) + "_" + dep_hour.astype(str)
    for col in cat_cols_all:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=360,
    max_depth=23,
    learning_rate=0.074,
    min_child_weight=2,
    subsample=0.95,
    colsample_bytree=0.85,
    max_bin=512,
    max_cat_to_onehot=32,
    max_cat_threshold=128,
    reg_lambda=0.5,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
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