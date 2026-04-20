import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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

base_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_cat_to_onehot": 4,
    "learning_rate": 0.1,
    "max_depth": 8,   # orig claude
    #"max_depth": 12,  # likely better based on our results
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_best_rounds = []
cv_fold_auc = []

cv_t0 = time.time()
for fold, (fit_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train[fit_idx]
    X_valid = X_train.iloc[valid_idx]
    y_valid = y_train[valid_idx]

    cv_model = xgb.XGBClassifier(
        **base_params,
        n_estimators=2000,
        early_stopping_rounds=50,
    )
    cv_model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    best_round = int(cv_model.best_iteration) + 1
    cv_best_rounds.append(best_round)

    valid_prob = cv_model.predict_proba(X_valid)[:, 1]
    cv_fold_auc.append(float(roc_auc_score(y_valid, valid_prob)))

cv_time_sec = time.time() - cv_t0
optimal_n_estimators = max(1, int(np.round(np.mean(cv_best_rounds))))

model = xgb.XGBClassifier(
    **base_params,
    n_estimators=optimal_n_estimators,
)

t0 = time.time()
model.fit(X_train, y_train)
final_fit_time_sec = time.time() - t0

results = {
    "rows": int(len(train)),
    "positive_rate": float((y_train == 1).mean()),
    "cv_folds": 5,
    "cv_best_rounds": cv_best_rounds,
    "cv_auc_mean": float(np.mean(cv_fold_auc)),
    "cv_auc_std": float(np.std(cv_fold_auc)),
    "cv_time_sec": float(cv_time_sec),
    "optimal_n_estimators": int(optimal_n_estimators),
    "final_fit_time_sec": float(final_fit_time_sec),
}

print(results)
