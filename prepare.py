import polars as pl
from pathlib import Path

data_dir = Path(__file__).parent / "data-cache"

keep_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier",
             "Origin", "Dest", "Distance", "dep_delayed_15min"]

for year in [2005, 2006, 2007]:
    print(f"\nProcessing {year}...")
    df = pl.read_csv(data_dir / f"{year}.csv", null_values="NA")

    df = df.with_columns(
        pl.when(pl.col("DepDelay").cast(pl.Int32, strict=False) >= 15).then(pl.lit("Y")).otherwise(pl.lit("N"))
          .alias("dep_delayed_15min"),
        *[("c-" + pl.col(col).cast(pl.Utf8)).alias(col)
          for col in ["Month", "DayofMonth", "DayOfWeek"]],
    )

    df = df.select(keep_cols).drop_nulls()

    # Separate by class and shuffle each independently
    yes = df.filter(pl.col("dep_delayed_15min") == "Y").sample(fraction=1.0, shuffle=True, seed=42)
    no  = df.filter(pl.col("dep_delayed_15min") == "N").sample(fraction=1.0, shuffle=True, seed=42)

    # Non-overlapping balanced slices: take from front of each class pool
    def balanced_slice(yes, no, n):
        half = n // 2
        return pl.concat([yes[:half], no[:half]]).sample(fraction=1.0, shuffle=True, seed=42)

    slice_100k = balanced_slice(yes, no, 100_000)
    # slice_1m is non-overlapping: starts after the 50K used for slice_100k
    slice_1m   = balanced_slice(yes[50_000:], no[50_000:], 1_000_000)

    slice_100k.write_csv(data_dir / f"{year}-slice1-100k.csv")
    slice_1m.write_csv(  data_dir / f"{year}-slice2-1m.csv")
    print(f"  slice1 (100k): {len(slice_100k):,} rows")
    print(f"  slice2 (1m):   {len(slice_1m):,} rows")
