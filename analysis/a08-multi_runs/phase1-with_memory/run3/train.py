import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


_carrier_hour_train = train["UniqueCarrier"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)
def _dep20_of(dt):
    mod = (dt // 100) * 60 + (dt % 100)
    return (mod // 20).astype(int)
_dep20_train = _dep20_of(train["DepTime"])
cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}
cat_levels["CarrierHour"] = sorted(_carrier_hour_train.unique())
cat_levels["Dep20Min"] = sorted(_dep20_train.unique())

def prepare(df):
    import numpy as np
    X = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    ch = df["UniqueCarrier"].astype(str) + "_" + (df["DepTime"] // 100).astype(str)
    X["CarrierHour"] = pd.Categorical(
        ch.where(ch.isin(cat_levels["CarrierHour"])),
        categories=cat_levels["CarrierHour"],
    )
    d20 = _dep20_of(df["DepTime"])
    X["Dep20Min"] = pd.Categorical(
        d20.where(d20.isin(cat_levels["Dep20Min"])),
        categories=cat_levels["Dep20Min"],
    )
    minute_of_day = (X["DepTime"] // 100) * 60 + (X["DepTime"] % 100)
    angle6 = 2 * np.pi * minute_of_day / 360.0
    X["DepTimeSin"] = np.sin(angle6)
    X["DepTimeCos"] = np.cos(angle6)
    X["DepMinute"] = (X["DepTime"] % 100).astype(int)
    month_int = df["Month"].str.removeprefix("c-").astype(int)
    m_angle = 2 * np.pi * (month_int - 1) / 12.0
    X["MonthSin"] = np.sin(m_angle).to_numpy()
    X["MonthCos"] = np.cos(m_angle).to_numpy()
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=440,
    max_depth=14,
    learning_rate=0.04,
    colsample_bytree=0.4,
    min_child_weight=3,
    max_bin=512,
    enable_categorical=True,
    max_cat_threshold=128,
    random_state=0,
    n_jobs=4,
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

t0 = time.time()
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=2)
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