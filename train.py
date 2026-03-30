import pandas as pd
import numpy as np
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/airline-10m-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    X["DepTime2"] = X["DepTime"] ** 2
    X["DepTime3"] = X["DepTime"] ** 3
    X["DepTime4"] = X["DepTime"] ** 4
    X["DepMinute"] = X["DepTime"] % 100
    X["DepHour"] = pd.Categorical((X["DepTime"] // 100).clip(0, 23).astype(int))
    X["DepTime_x_Dist"] = X["DepTime"] * X["Distance"]
    # Cyclical encoding
    hour = (X["DepTime"] // 100).clip(0, 23)
    X["DepHour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["DepHour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = pd.to_numeric(df["DayOfWeek"], errors="coerce")
    X["DayOfWeek_sin"] = np.sin(2 * np.pi * dow / 7)
    X["DayOfWeek_cos"] = np.cos(2 * np.pi * dow / 7)
    month = pd.to_numeric(df["Month"], errors="coerce")
    X["Month_sin"] = np.sin(2 * np.pi * month / 12)
    X["Month_cos"] = np.cos(2 * np.pi * month / 12)
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=2500,
    max_depth=11,
    learning_rate=0.01,
    min_child_weight=20,
    gamma=0.4,
    reg_alpha=0.5,
    reg_lambda=3,
    subsample=0.95,
    colsample_bytree=0.55,
    tree_method="hist",
    max_bin=2048,
    max_cat_threshold=40,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

t0 = time.time()
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"CV time: {time.time() - t0:.1f}s")
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

t0 = time.time()
model.fit(X_train, y_train)
print(f"Final model training time: {time.time() - t0:.1f}s")


