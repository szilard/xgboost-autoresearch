# pip install optuna optuna-integration[xgboost]

import time
import warnings
import optuna
import pandas as pd
import xgboost as xgb
from pathlib import Path
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


data_dir = Path(__file__).parent / "../../data-cache"
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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1500),
        "max_depth":         trial.suggest_int("max_depth", 5, 25),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 0.15),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "max_bin":           trial.suggest_categorical("max_bin", [128, 256, 512, 1024, 2048, 4096]),
        "tree_method":       "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1,
    }
    scores = []
    for fold, (idx_tr, idx_val) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
        y_tr, y_val = y_train[idx_tr], y_train[idx_val]
        m = xgb.XGBClassifier(**params, eval_metric="auc",
                               callbacks=[XGBoostPruningCallback(trial, "validation_0-auc")])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The reported value is ignored", category=UserWarning)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        trial.report(sum(scores) / len(scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return sum(scores) / len(scores)

t0 = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=123),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps = 100),  ## pruning only after 100 steps
)
study.optimize(objective, n_trials = 100, show_progress_bar=True)   ## n_trials = number of HPO iterations
print(f"Optuna time: {time.time() - t0:.1f}s")
print(f"Best CV AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_params = study.best_params | {"enable_categorical": True, "random_state": 42, "n_jobs": -1}
t0 = time.time()
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)
print(f"Final model (using all the data) training time: {time.time() - t0:.1f}s")



test = pd.read_csv(f"{data_dir}/2005-slice2-1m.csv")
X_test, y_test = prepare(test)

t0 = time.time()
y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
print(f"Test time: {time.time() - t0:.1f}s")
print(f"Test AUC: {test_auc:.4f}")


