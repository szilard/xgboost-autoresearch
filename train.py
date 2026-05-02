import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["UniqueCarrier", "Origin", "Dest", "Dep20Min"]
num_cols = ["DepTime", "Distance", "DepMinute", "DayofMonthNum", "MonthNum", "DayOfWeekNum", "SinHour", "CosHour"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols if col != "Dep20Min"}
cat_levels["Dep20Min"] = list(range(73))

def prepare(df):
    df = df.copy()
    dep_minutes = (df["DepTime"] // 100) * 60 + (df["DepTime"] % 100)
    df["Dep20Min"] = dep_minutes // 20
    df["DepMinute"] = df["DepTime"] % 100
    df["DayofMonthNum"] = df["DayofMonth"].str[2:].astype(int)
    df["MonthNum"] = df["Month"].str[2:].astype(int)
    df["DayOfWeekNum"] = df["DayOfWeek"].str[2:].astype(int)
    import numpy as np
    hr = df["DepTime"] // 100 + (df["DepTime"] % 100) / 60
    df["SinHour"] = np.sin(2 * np.pi * hr / 12)
    df["CosHour"] = np.cos(2 * np.pi * hr / 12)
    X = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=340,
    max_depth=14,
    learning_rate=0.07,
    min_child_weight=2,
    subsample=0.96,
    colsample_bytree=0.7,
    max_cat_threshold=128,
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