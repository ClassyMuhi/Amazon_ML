import time
import gc
import pandas as pd

from src.normalization import add_normalized_features
from src.blocking import generate_candidate_pairs

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"

S1_SIZE = 5000
S2_SIZE = 50000


print("=" * 70)
print("PREDICTION BLOCKING BENCHMARK")
print("=" * 70)

start = time.time()

print("\nLoading Source1...")
s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    nrows=S1_SIZE,
    dtype=str,
    keep_default_na=False,
    na_filter=False,
)

print(f"Source1 rows: {len(s1):,}")

print("\nNormalizing Source1...")
s1 = add_normalized_features(s1)

print(
    f"Source1 normalization time: "
    f"{time.time() - start:.2f} seconds"
)

print("\nLoading Source2...")
s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    nrows=S2_SIZE,
    dtype=str,
    keep_default_na=False,
    na_filter=False,
)

print(f"Source2 rows: {len(s2):,}")

print("\nNormalizing Source2...")
target_start = time.time()

s2 = add_normalized_features(s2)

print(
    f"Source2 normalization time: "
    f"{time.time() - target_start:.2f} seconds"
)

print("\nRunning blocking...")

blocking_start = time.time()

candidates = generate_candidate_pairs(
    s1,
    s2,
    target_label="S2",
)

blocking_time = time.time() - blocking_start

print("\n" + "=" * 70)
print("BENCHMARK RESULT")
print("=" * 70)

print(
    f"Source1 rows      : {len(s1):,}"
)

print(
    f"Source2 rows      : {len(s2):,}"
)

print(
    f"Candidates        : {len(candidates):,}"
)

print(
    f"Blocking time     : {blocking_time:.2f} seconds"
)

print(
    f"Total time        : {time.time() - start:.2f} seconds"
)

print("=" * 70)

del candidates
del s1
del s2

gc.collect()