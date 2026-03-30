import pandas as pd
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
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.02,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=1,
    reg_lambda=5,
    subsample=0.8,
    colsample_bytree=0.8,
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


