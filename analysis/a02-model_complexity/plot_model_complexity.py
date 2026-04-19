import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("groundtruth_all.tsv", sep="\t")
df.insert(0, "n", range(1, len(df) + 1))
df.columns = df.columns.str.strip()

for col in ["numb_trees", "eff_depth", "leaves_per_data"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

panels = [
    ("numb_trees",      "# trees",                      "steelblue"),
    ("eff_depth",       "eff depth log2(leaves/trees)",  "mediumpurple"),
    ("leaves_per_data", "leaves / datapoints",            "coral"),
]

for col, ylabel, color in panels:
    d = df[["n", col]].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d["n"], d[col], color=color, linewidth=1.2, marker="o", zorder=1)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs n")
    ax.set_ylim(ymin=0)
    ax.grid(True, color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"model_complexity_{col}.png", dpi=150)
    plt.show()
