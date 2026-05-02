import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "results-methods.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()
df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce")
df = df.dropna(subset=["AUC"]).sort_values("AUC")

def method_color(name):
    n = name.lower()
    if "optuna" in n:   return "brown"
    if "autogl" in n:   return "steelblue"
    if "claude" in n:   return "mediumpurple"
    return "darkgrey"

colors = [method_color(m) for m in df["method"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(df["method"], df["AUC"], color=colors)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
ax.set_xlabel("AUC")
ax.set_xlim(0.68, df["AUC"].max() + 0.02)
ax.set_title("AUC by method")
ax.grid(axis="x", color="lightgrey", linewidth=0.5)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "results_methods.png", dpi=150)
plt.show()
