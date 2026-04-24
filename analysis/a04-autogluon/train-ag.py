# pip install autogluon.tabular[all]

import pandas as pd
import time
from pathlib import Path
from autogluon.tabular import TabularPredictor


data_dir = Path(__file__).parent / "../../data-cache"
train = pd.read_csv(f"{data_dir}/2005-slice1-100k.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
    df_out = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        df_out[col] = pd.Categorical(
            df_out[col].where(df_out[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    df_out[target] = (df[target] == "Y").astype(int)
    return df_out

train = prepare(train)


t0 = time.time()
predictor = TabularPredictor(
    label=target,
    eval_metric="roc_auc",
    path=".autogluon-model",
).fit(
    train,
    presets="best_quality",
)
print(f"Training time: {time.time() - t0:.1f}s")

leaderboard = predictor.leaderboard(silent=True)
print(leaderboard[["model", "score_val", "fit_time"]].to_string())



from sklearn.metrics import roc_auc_score

test = pd.read_csv(f"{data_dir}/2005-slice2-1m.csv")
test = prepare(test)

t0 = time.time()
y_prob = predictor.predict_proba(test).iloc[:, 1]
test_auc = roc_auc_score(test[target], y_prob)
print(f"Test time: {time.time() - t0:.1f}s")
print(f"Test AUC: {test_auc:.4f}")


