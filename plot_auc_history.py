import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("groundtruth_all.tsv", sep="\t")
df.insert(0, " n ", range(1, len(df) + 1))
df.columns = df.columns.str.strip()

for col in ["eval_auc", "test_auc_2006s2", "test_auc_2007s2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

plt.figure(figsize=(10, 6))
keep = df["status"] == "keep"
plt.plot(df.loc[~keep, "n"], df.loc[~keep, "eval_auc"], marker="o", color="lightgrey", linestyle="none", label="discard")
plt.plot(df.loc[keep, "n"], df.loc[keep, "eval_auc"], color="cornflowerblue", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "eval_auc"], marker="o", color="steelblue", linestyle="none", label="keep", zorder=2)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2006s2"], color="#3a8f56", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2006s2"], marker="o", color="#3a8f56", linestyle="none", label="test 2006s2", zorder=2)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2007s2"], color="#d47a20", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2007s2"], marker="o", color="#d47a20", linestyle="none", label="test 2007s2", zorder=2)
plt.xlabel("n")
plt.ylabel("AUC")
plt.ylim(ymin=0.70, ymax=0.76)
plt.title("AUC vs n")
plt.grid(True, color="lightgrey", linewidth=0.5)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("auc_history.png", dpi=150)
plt.show()

for col in ["cv_auc", "test_auc_2005s2_full", "test_auc_2005s2_4_5", "test_auc_2006_full"]:
    idx = df[col].idxmax()
    print(f"{col}: max={df.loc[idx, col]:.4f} at n={df.loc[idx, 'n']}")