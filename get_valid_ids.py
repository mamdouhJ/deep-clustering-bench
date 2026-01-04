"""
This script goes over a list of IDs and randomly sample the following: 
80 datasets with:
    - <=10k samples
    - <=1000 features
    - >= 5 classes 
10 datasets with:
    - >= 50k samples
10 datasets with:
    - <= 1000 samples
    - 2-4 classes
"""
import numpy as np
from bench.datasets import load_dataset

NORMAL_DATASETS = []
BIG_DATASETS = []
SMALL_DATASETS = []

def load_ids(filepath: str) -> list[int]:
    with open(filepath, "r") as f:
        ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    return ids

def get_attrs(X,y):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    return n_samples, n_features, n_classes

def save_ids(filepath: str, ids: list[int]) -> None:
    with open(filepath, "w") as f:
        for did in ids:
            f.write(f"{did}\n")

def run(filepath):
    ids = load_ids(filepath)
    for idx, did in enumerate(ids):
        print("Processing dataset", idx+1, "of", len(ids), "with ID:", did)
        try:
            X, y = load_dataset(did)
            n_samples, n_features, n_classes = get_attrs(X,y)
            if n_samples <= 10000 and n_features <= 1000 and n_classes >= 5:
                NORMAL_DATASETS.append(did)
            if n_samples >= 50000 and n_features <= 1000 and n_classes >= 5:
                BIG_DATASETS.append(did)
            if n_samples <= 1000 and n_features <= 1000 and 2 <= n_classes <= 4:
                SMALL_DATASETS.append(did)
        except Exception as e:
            print(f"Failed to load dataset {did}: {e}")
    print("Normal datasets:", NORMAL_DATASETS)
    print("Big datasets:", BIG_DATASETS)
    print("Small datasets:", SMALL_DATASETS)
    NORMAL_DATASETS_SAMPLE = np.random.choice(NORMAL_DATASETS, size=min(80, len(NORMAL_DATASETS)), replace=False).tolist()
    save_ids("normal_ids.txt", NORMAL_DATASETS_SAMPLE)
    BIG_DATASETS_SAMPLE = np.random.choice(BIG_DATASETS, size=min(10, len(BIG_DATASETS)), replace=False).tolist()
    save_ids("big_ids.txt", BIG_DATASETS_SAMPLE)
    SMALL_DATASETS_SAMPLE = np.random.choice(SMALL_DATASETS, size=min(10, len(SMALL_DATASETS)), replace=False).tolist()
    save_ids("small_ids.txt", SMALL_DATASETS_SAMPLE)

if __name__ == "__main__":
    run("valid_ids.txt")