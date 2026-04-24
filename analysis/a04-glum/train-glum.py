# pip install glum tabmat

import pandas as pd
import time
from pathlib import Path
from glum import GeneralizedLinearRegressor
from sklearn.metrics import roc_auc_score
import tabmat as tm


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
    X = tm.from_pandas(X, cat_threshold=1, cat_missing_method='zero')
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_train, y_train = prepare(train)


model = GeneralizedLinearRegressor(family="binomial")

t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


test = pd.read_csv(f"{data_dir}/2005-slice2-1m.csv")
X_test, y_test = prepare(test)

t0 = time.time()
y_prob = model.predict(X_test)
test_auc = roc_auc_score(y_test, y_prob)
print(f"Test time: {time.time() - t0:.1f}s")
print(f"Test AUC: {test_auc:.4f}")



