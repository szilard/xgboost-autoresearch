import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path(__file__).parent / "data-cache"
train = pd.read_csv(f"{data_dir}/airline-1m-slice100k-1.csv")

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
    # Feature engineering
    dep = pd.to_numeric(X["DepTime"], errors="coerce").fillna(1200)
    X["DepHour"] = (dep // 100).clip(0, 23).astype(int)
    X["DepMinute"] = (dep % 100).clip(0, 59).astype(int)
    X["DepTimeMin"] = (X["DepHour"] * 60 + X["DepMinute"]).astype(int)
    X["DepTimeFrac"] = X["DepTimeMin"] / 1440.0
    X["DepHourSq"] = (X["DepHour"] ** 2).astype(int)
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)

# Split for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=5000,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.80,
    colsample_bytree=0.5,
    colsample_bylevel=0.7,
    min_child_weight=25,
    reg_alpha=0.4,
    reg_lambda=1.2,
    gamma=0.0,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=150,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

t0 = time.time()
# Retrain on full data with best n_estimators
best_n = model.best_iteration + 1
model2 = xgb.XGBClassifier(
    n_estimators=best_n,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.80,
    colsample_bytree=0.5,
    colsample_bylevel=0.7,
    min_child_weight=25,
    reg_alpha=0.4,
    reg_lambda=1.2,
    gamma=0.0,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
model2.fit(X_train, y_train)
model = model2
print(f"Best n_estimators: {best_n}")
print(f"Training time: {time.time() - t0:.1f}s")

## run evaluation
exec(open(__file__.replace("train.py", "evaluate.py")).read())
