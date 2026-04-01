exec(open(__file__.replace("check_groundtruth.py", "train.py")).read())
print()

from sklearn.metrics import roc_auc_score
import numpy as np

rng = np.random.default_rng(42)
n_boot = 100

combos = [
    ("full model - eval 2005 slice 2", model,     "2005-slice2-1m.csv"),
    ("4/5 model - eval 2005 slice 2",  model_4_5, "2005-slice2-1m.csv"),
    ("full model - eval 2006",         model,     "2006-slice2-1m.csv"),
]

for label, m, csv in combos:
    test = pd.read_csv(f"{data_dir}/{csv}")
    X_test, y_test = prepare(test)

    t0 = time.time()
    y_prob = m.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Test time ({label}): {time.time() - t0:.1f}s")
    print(f"Test AUC ({label}): {test_auc:.4f}")

    n = len(y_test)
    t0 = time.time()
    boot_aucs = np.array([
        roc_auc_score(y_test[idx], y_prob[idx])
        for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))
    ])
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"Bootstrap time ({label}): {time.time() - t0:.1f}s")
    print(f"Bootstrap AUC ({label}): {boot_aucs.mean():.4f} [95% CI: {ci_lo:.4f}–{ci_hi:.4f}]")
    print()

