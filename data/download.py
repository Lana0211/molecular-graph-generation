"""
Download ZINC250k dataset used as the MOSES benchmark training/test split.

ZINC250k is sourced from the MOSES benchmark paper:
  Polykovskiy et al., "Molecular Sets (MOSES): A Benchmarking Platform
  for Molecular Generation Models", Frontiers in Pharmacology, 2020.

The raw CSV contains a 'SMILES' column and a 'SPLIT' column ('train'/'test').
"""

import os
import urllib.request
import pandas as pd

# Resolve paths relative to this file so the script works from any working directory
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
MOSES_URL = (
    "https://media.githubusercontent.com/media/molecularsets/moses/"
    "master/data/dataset_v1.csv"
)
LOCAL_CSV = os.path.join(DATA_DIR, "dataset_v1.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")


def download():
    """Download the raw MOSES CSV if it is not already present."""
    if os.path.exists(LOCAL_CSV):
        print(f"[data] Found existing {LOCAL_CSV}, skipping download.")
        return LOCAL_CSV
    print(f"[data] Downloading MOSES dataset to {LOCAL_CSV} ...")
    urllib.request.urlretrieve(MOSES_URL, LOCAL_CSV)
    print("[data] Download complete.")
    return LOCAL_CSV


def split_and_save():
    """Split the raw CSV into train/test subsets and persist them."""
    csv_path = download()
    df = pd.read_csv(csv_path)

    # Strip whitespace / BOM characters that can corrupt column names on Windows
    df.columns = df.columns.str.strip()
    print(f"[data] Total rows: {len(df)}")

    # Locate the split column regardless of case (CSV uses 'SPLIT', 'split', etc.)
    split_col = next(
        (c for c in df.columns if c.upper() == "SPLIT"),
        None,
    )
    if split_col is None:
        raise ValueError(f"[data] No SPLIT column found. Got: {df.columns.tolist()}")

    # Use .str.lower() so values like 'Train' or 'TRAIN' are handled correctly
    train = df[df[split_col].str.lower() == "train"][["SMILES"]].reset_index(drop=True)
    test  = df[df[split_col].str.lower() == "test"][["SMILES"]].reset_index(drop=True)
    print(f"[data]  train: {len(train)}, test: {len(test)}")

    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"[data] Saved {TRAIN_CSV} and {TEST_CSV}")
    return train, test


def load(split="train"):
    """Return a list of SMILES strings for the requested split."""
    path = TRAIN_CSV if split == "train" else TEST_CSV
    if not os.path.exists(path):   # generate split files on first call
        split_and_save()
    df = pd.read_csv(path)
    return df["SMILES"].tolist()


if __name__ == "__main__":
    split_and_save()
