exec(open(__file__.replace("check_groundtruth.py", "train.py")).read())
import math
print()

from sklearn.metrics import roc_auc_score

combos = [
    ("eval 2006 slice 2", model,     "2006-slice2-1m.csv"),
    ("eval 2007 slice 2", model,     "2007-slice2-1m.csv"),
]

for label, m, csv in combos:
    test = pd.read_csv(f"{data_dir}/{csv}")
    X_test, y_test = prepare(test)

    t0 = time.time()
    y_prob = m.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Test time ({label}): {time.time() - t0:.1f}s")
    print(f"Test AUC ({label}): {test_auc:.4f}")



booster = model.get_booster()
trees = booster.get_dump()
n_trees = len(trees)
n_leaves = sum(tree.count('leaf=') for tree in trees)
n_datapoints = len(X_train)
eff_depth = math.log2(n_leaves / n_trees)
print(f"\nModel numb_trees: {n_trees}")
print(f"Model eff_depth: {eff_depth:.4f}")
print(f"Model leaves_per_data: {n_leaves / n_datapoints:.4f}")