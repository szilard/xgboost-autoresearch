import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


data_dir = Path(__file__).parent / "../../../data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target = "dep_delayed_15min"

cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}


def prepare(df):
    X = df[num_cols + cat_cols].copy()
    X["hour"] = (df["DepTime"] // 100).clip(0, 23).astype(np.int16)
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y
  
X_train, y_train = prepare(train)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=4,
    learning_rate=0.1,
    max_depth=8,
    n_estimators=2000,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
)

t0 = time.time()
model.fit(X_train, y_train)
print(f"Final model training time: {time.time() - t0:.1f}s")
