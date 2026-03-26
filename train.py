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
    X["IsWeekend"] = X["DayOfWeek"].cat.codes.isin([5, 6]).astype(int)
    X["IsLateNight"] = ((X["DepHour"] >= 21) | (X["DepHour"] <= 5)).astype(int)
    X["IsMorningRush"] = ((X["DepHour"] >= 6) & (X["DepHour"] <= 9)).astype(int)
    X["IsEveningRush"] = ((X["DepHour"] >= 16) & (X["DepHour"] <= 19)).astype(int)
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)

# Split for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.2,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=80,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

t0 = time.time()
# Retrain on full data with best n_estimators
best_n = model.best_iteration + 1
model2 = xgb.XGBClassifier(
    n_estimators=best_n,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.2,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
model2.fit(X_train, y_train)
model = model2
print(f"Best n_estimators: {best_n}")
print(f"Training time: {time.time() - t0:.1f}s")

exec(open(__file__.replace("train.py", "evaluate.py")).read())
