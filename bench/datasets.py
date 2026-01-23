"""
loading and saving datasets from OpenML in compressed .npz format.
"""
import os
import numpy as np
import openml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ.setdefault("OPENML_SKIP_PARQUET", "true")

def _map_non_numeric_labels_to_numeric(labels):
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_mapping[label] for label in labels])
    return numeric_labels

def _check_numeric_features(X):
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("All features must be numeric.")

def save_npz(did, X, y):
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data','raw'))
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"dataset_{did}.npz")
    np.savez_compressed(filepath, X=X, y=y)

def preprocess_dataset(X,preprocess='minmax'):
    if preprocess == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = MinMaxScaler(feature_range=(0, 0.9999999))
        X_scaled = scaler.fit_transform(X)
    return X_scaled

def dataset_already_saved(did):
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data','raw'))
    filepath = os.path.join(cache_dir, f"dataset_{did}.npz")
    return os.path.exists(filepath)

def load_dataset(did,preprocess="minmax"):
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data','raw'))
    if dataset_already_saved(did):
        filepath = os.path.join(cache_dir, f"dataset_{did}.npz")
        with np.load(filepath) as data:
            X = data['X']
            y = data['y']
            X = preprocess_dataset(X,preprocess=preprocess) if preprocess is not None else X
        return X, y
    dataset = openml.datasets.get_dataset(did)
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    X = X.to_numpy()
    X = X.astype(np.float32)
    _check_numeric_features(X)
    y = y.to_numpy()
    y = _map_non_numeric_labels_to_numeric(y)
    y = y.astype(np.int64)
    save_npz(did, X, y)
    X = preprocess_dataset(X)

    return X, y

def get_dataset_metadata(did):
    X, y = load_dataset(did)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
    }

def get_ids_from_txt(filepath):
    with open(filepath, "r") as f:
        ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    return ids


