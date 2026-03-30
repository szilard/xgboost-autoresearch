from sklearn.metrics import roc_auc_score

test  = pd.read_csv(f"{data_dir}/airline-1m-slice100k-2.csv")

X_test, y_test = prepare(test)

test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
print(f"Test AUC: {test_auc:.4f}")
