exec(open(__file__.replace("check_overfitting.py", "train.py")).read())

from sklearn.metrics import roc_auc_score

test  = pd.read_csv(f"{data_dir}/airline-10m-slice2-1m.csv")

X_test, y_test = prepare(test)

t0 = time.time()
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Test time (slice 2): {time.time() - t0:.1f}s")
print(f"Test AUC: {test_auc:.4f}")


## To get confidence intervals, we can use bootstrapping:
import numpy as np
y_prob = model.predict_proba(X_test)[:, 1]

rng = np.random.default_rng(42)
n = len(y_test)
n_boot = 100
t0 = time.time()
boot_aucs = [
    roc_auc_score(y_test[idx], y_prob[idx])
    for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))
]
boot_aucs = np.array(boot_aucs)
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f"Boostrap time (slice 2): {time.time() - t0:.1f}s")
print(f"Bootstrap AUC: {boot_aucs.mean():.4f} [95% CI: {ci_lo:.4f}–{ci_hi:.4f}]")
