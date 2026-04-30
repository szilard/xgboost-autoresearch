import pandas as pd
import numpy as np
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
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
    n_estimators=8000,
    max_depth=14,
    learning_rate=0.02,
    min_child_weight=3,
    colsample_bytree=0.55,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="auc",
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

t0 = time.time()
fold_scores = []
best_iters = []
for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    m = clone(model)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred = m.predict_proba(X_va)[:, 1]
    fold_scores.append(roc_auc_score(y_va, pred))
    best_iters.append(m.best_iteration)
print(f"CV time: {time.time() - t0:.1f}s")
scores = np.array(fold_scores)
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"best_iterations: {best_iters} (mean={int(np.mean(best_iters))})")

mean_iters = int(np.mean(best_iters))
final_model = clone(model)
final_model.set_params(n_estimators=mean_iters, early_stopping_rounds=None)

t0 = time.time()
final_model.fit(X_train, y_train)
print(f"Final model training time: {time.time() - t0:.1f}s")


# 4/5 model: train on the first fold's training split (same as one CV fold)
train_idx_4_5 = list(cv.split(X_train, y_train))[0][0]
X_4_5 = X_train.iloc[train_idx_4_5]
y_4_5 = y_train[train_idx_4_5]

model_4_5 = clone(final_model)
t0 = time.time()
model_4_5.fit(X_4_5, y_4_5)
print(f"4/5 model training time: {time.time() - t0:.1f}s")
