import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("compare-scenarios/groundtruth_all-scen1.tsv", sep="\t")
df2 = pd.read_csv("compare-scenarios/groundtruth_all-scen2.tsv", sep="\t")
df1.insert(0, " n ", range(1, len(df1) + 1))
df1.columns = df1.columns.str.strip()
df2.insert(0, " n ", range(1, len(df2) + 1))
df2.columns = df2.columns.str.strip()

df = df1[["n", "test_auc_2006_full"]].merge(df2[["n", "test_auc_2006s2"]], on="n", how="outer")
df = df.rename(columns={"test_auc_2006_full": "auc_2006s2_scen1", "test_auc_2006s2": "auc_2006s2_scen2"})


for col in ["auc_2006s2_scen1", "auc_2006s2_scen2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

plt.figure(figsize=(10, 6))
s1 = df.dropna(subset=["auc_2006s2_scen1"])
s2 = df.dropna(subset=["auc_2006s2_scen2"])
plt.plot(s1["n"], s1["auc_2006s2_scen1"], color="#3a8f56", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(s1["n"], s1["auc_2006s2_scen1"], marker="o", color="#3a8f56", linestyle="none", label="scenario 1: train and keep/discard CV on 2005, test 2006s2", zorder=2)
plt.plot(s2["n"], s2["auc_2006s2_scen2"], color="#a0d080", linewidth=1.2, linestyle="-", zorder=1)
plt.plot(s2["n"], s2["auc_2006s2_scen2"], marker="o", color="#a0d080", linestyle="none", label="scenario 2: train on 2005, keep/discard on 2006s1, test 2006s2", zorder=2)
plt.xlabel("n")
plt.ylabel("AUC")
plt.title("AUC vs n — scenario comparison")
plt.grid(True, color="lightgrey", linewidth=0.5)
plt.ylim(ymin=0.70, ymax=0.76)
plt.legend(loc="lower right", fontsize=15)
plt.tight_layout()
plt.savefig("compare-scenarios/auc_history_compare.png", dpi=150)
plt.show()
