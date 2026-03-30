# Temporarily set __file__ to train.py so that train.py's internal exec of evaluate.py
# (which uses __file__.replace("train.py", "evaluate.py")) resolves correctly and doesn't
# loop back into this script.
__file__ = __file__.replace("check_overfitting.py", "train.py")
exec(open(__file__).read())
__file__ = __file__.replace("train.py", "check_overfitting.py")

## run evaluation on slice 3 (overfitting check)
test  = pd.read_csv(f"{data_dir}/airline-1m-slice100k-3.csv")

X_test, y_test = prepare(test)

from sklearn.metrics import roc_auc_score
test3_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Test AUC (slice 3): {test3_auc:.4f}")
