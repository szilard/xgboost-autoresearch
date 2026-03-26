import polars as pl
import os
import urllib.request
from pathlib import Path

data_dir = Path(__file__).parent / "data-cache"

os.makedirs(data_dir, exist_ok=True)
def download_with_progress(url, dest):
    def reporthook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        print(f"\rDownloading: {percent}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()

url = "https://s3.amazonaws.com/benchm-ml--main/train-1m.csv"
download_with_progress(url, f"{data_dir}/train-1m.csv")

df = pl.read_csv(f"{data_dir}/train-1m.csv")

df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)

df_shuffled[:100_000].write_csv(f"{data_dir}/airline-1m-slice100k-1.csv")
df_shuffled[100_000:200_000].write_csv(f"{data_dir}/airline-1m-slice100k-2.csv")
df_shuffled[200_000:300_000].write_csv(f"{data_dir}/airline-1m-slice100k-3.csv")