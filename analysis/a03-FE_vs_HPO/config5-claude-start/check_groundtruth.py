#exec(open(__file__.replace("check_groundtruth.py", "train.py")).read())
#exec(open(__file__.replace("check_groundtruth.py", "train-claude.py")).read())
exec(open(__file__.replace("check_groundtruth.py", "train-claude-earlystopping.py")).read())
print()

from sklearn.metrics import roc_auc_score

combos = [
    ("full model - eval 2005 slice 2", model,     "2005-slice2-1m.csv"),
]

for label, m, csv in combos:
    test = pd.read_csv(f"{data_dir}/{csv}")
    X_test, y_test = prepare(test)

    t0 = time.time()
    y_prob = m.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Test time ({label}): {time.time() - t0:.1f}s")
    print(f"Test AUC ({label}): {test_auc:.4f}")
