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
    """Download the raw MOSES CSV if it is not already present.

    Idempotent: if the file already exists on disk we skip the network call,
    so repeated notebook runs don't re-download ~20 MB every time.
    Returns the local path to the raw CSV.
    """
    # Short-circuit when the dataset has already been fetched in a previous run.
    if os.path.exists(LOCAL_CSV):
        print(f"[data] Found existing {LOCAL_CSV}, skipping download.")
        return LOCAL_CSV
    # First-time setup: stream the file from the MOSES GitHub mirror to disk.
    print(f"[data] Downloading MOSES dataset to {LOCAL_CSV} ...")
    urllib.request.urlretrieve(MOSES_URL, LOCAL_CSV)
    print("[data] Download complete.")
    return LOCAL_CSV


def split_and_save():
    """Split the raw CSV into train/test subsets and persist them as CSVs.

    The raw MOSES file mixes both splits in one table (tagged by a SPLIT
    column). We separate them into train.csv / test.csv so the rest of the
    project can load each split with a single read.
    Returns the (train, test) DataFrames.
    """
    # Make sure the raw file is on disk, then load it into a DataFrame.
    csv_path = download()
    df = pd.read_csv(csv_path)

    # Strip whitespace / BOM characters that can corrupt column names on Windows
    # (a leading BOM would otherwise turn 'SMILES' into '﻿SMILES').
    df.columns = df.columns.str.strip()
    print(f"[data] Total rows: {len(df)}")

    # Locate the split column regardless of case (CSV uses 'SPLIT', 'split', etc.).
    # next(...) returns the first matching column name, or None if there is none.
    split_col = next(
        (c for c in df.columns if c.upper() == "SPLIT"),
        None,
    )
    if split_col is None:
        raise ValueError(f"[data] No SPLIT column found. Got: {df.columns.tolist()}")

    # Filter rows by split value and keep only the SMILES column.
    # .str.lower() makes the match robust to 'Train' / 'TRAIN' / 'train'.
    # reset_index(drop=True) renumbers the rows 0..N after filtering.
    train = df[df[split_col].str.lower() == "train"][["SMILES"]].reset_index(drop=True)
    test  = df[df[split_col].str.lower() == "test"][["SMILES"]].reset_index(drop=True)
    print(f"[data]  train: {len(train)}, test: {len(test)}")

    # Persist both splits so subsequent runs can skip the split step entirely.
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"[data] Saved {TRAIN_CSV} and {TEST_CSV}")
    return train, test


def load(split="train"):
    """Return a plain list of SMILES strings for the requested split.

    This is the main entry point used by the notebook / training scripts.
    Lazily generates the split files on first call so callers never have to
    invoke split_and_save() themselves.
    """
    # Pick the target file based on the requested split.
    path = TRAIN_CSV if split == "train" else TEST_CSV
    # Generate the split files on first call if they don't exist yet.
    if not os.path.exists(path):
        split_and_save()
    # Read back the single-column CSV and hand back a Python list of strings.
    df = pd.read_csv(path)
    return df["SMILES"].tolist()


if __name__ == "__main__":
    split_and_save()
