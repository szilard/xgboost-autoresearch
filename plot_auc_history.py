import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("groundtruth_all.tsv", sep="\t")
df.insert(0, " n ", range(1, len(df) + 1))
df.columns = df.columns.str.strip()

for col in ["cv_auc", "test_auc_2005s2_full", "test_auc_2005s2_4_5", "test_auc_2006_full"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

plt.figure(figsize=(10, 6))
keep = df["status"] == "keep"
plt.plot(df.loc[~keep, "n"], df.loc[~keep, "cv_auc"], marker="o", color="lightgrey", linestyle="none", label="CV discard")
plt.plot(df.loc[keep, "n"], df.loc[keep, "cv_auc"], color="cornflowerblue", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "cv_auc"], marker="o", color="steelblue", linestyle="none", label="CV keep", zorder=2)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2005s2_full"], color="#d94040", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2005s2_full"], marker="o", color="#d94040", linestyle="none", label="test 2005s2 full", zorder=2)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2005s2_4_5"], color="#d040d0", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2005s2_4_5"], marker="o", color="#d040d0", linestyle="none", label="test 2005s2 4_5", zorder=2)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2006_full"], color="#3a8f56", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(df.loc[keep, "n"], df.loc[keep, "test_auc_2006_full"], marker="o", color="#3a8f56", linestyle="none", label="test 2006 full", zorder=2)
plt.xlabel("n")
plt.ylabel("AUC")
plt.ylim(ymin=0.70, ymax=0.85)
plt.title("AUC vs n")
plt.grid(True, color="lightgrey", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("auc_history.png", dpi=150)
plt.show()
