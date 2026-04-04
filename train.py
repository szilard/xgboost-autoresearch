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


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

# Target encoding for Origin and Dest (computed on training data only)
y_train_raw = (train[target] == "Y").astype(int)
global_mean = y_train_raw.mean()
origin_te = y_train_raw.groupby(train["Origin"]).mean()
dest_te = y_train_raw.groupby(train["Dest"]).mean()
carrier_te = y_train_raw.groupby(train["UniqueCarrier"]).mean()

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    X["DepHour"] = (X["DepTime"] // 100).clip(0, 23)
    X["DepMinute"] = X["DepTime"] % 100
    X["Origin_te"] = df["Origin"].map(origin_te).fillna(global_mean)
    X["Dest_te"] = df["Dest"].map(dest_te).fillna(global_mean)
    X["Carrier_te"] = df["UniqueCarrier"].map(carrier_te).fillna(global_mean)
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    min_child_weight=10,
    subsample=0.8,
    max_bin=1024,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)


t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


from sklearn.metrics import roc_auc_score

eval = pd.read_csv(f"{data_dir}/2006-slice1-100k.csv")

X_eval, y_eval = prepare(eval)

t0 = time.time()
eval_auc  = roc_auc_score(y_eval,  model.predict_proba(X_eval)[:, 1])
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")