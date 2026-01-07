import os 
import time
import argparse
from pathlib import Path
from datasets import load_dataset
from utils import get_config_file_from_number, split_indices, save_results_jsonl
from algorithms import DeepClusteringAlgorithm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
"""
parser = argparse.ArgumentParser(description="Given dataset ID, algorithm name, and config number, run the deep clustering algorithm on the dataset.")
parser.add_argument("--did", type=int, required=True, help="Dataset ID (OpenML ID)")
parser.add_argument("--algo", type=str, required=True, choices=['DDC', 'IDEC', 'DipEncoder', 'VaDE', 'DCN'], help="Algorithm name (choose from: 'DDC', 'IDEC', 'DipEncoder', 'VaDE', 'DCN')")
parser.add_argument("--config_number", type=int, required=True, help="Configuration number (line number in the config file)")
args = parser.parse_args()
"""
args = argparse.Namespace()
args.did = 36
args.algo = "IDEC"
args.config_number = 0
args.num_runs = 5
config_file = get_config_file_from_number(args.did, args.algo, args.config_number)
print(config_file)
run_id = config_file["run_id"]
config = config_file["config"]
seed = config["seed"]
t0 = time.time()
X, y = load_dataset(args.did)
results_list = []
for i in range(args.num_runs):
    config['seed'] = config['seed'] + 1
    print(f"The config for run {i+1} is: {config}")
    model = DeepClusteringAlgorithm(config)
    if config['early_stopping']:
        train_idx, val_idx = split_indices(X.shape[0], seed, val_ratio=0.2)
        X_train = X[train_idx]
        X_val = X[val_idx]
        model.fit(X_train, val_set=X_val)
    else:
        model.fit(X)
    metrics = model.get_metrics(y, X)

    results_config = {
        "run_id": run_id,
        "did": args.did,
        "algo": args.algo,
        "config_number": args.config_number,
        "seed": seed,
        "metrics": metrics,
        "time_taken": time.time() - t0,
    }
    results_list.append(results_config)
    os.makedirs(ROOT / "results" / str(args.did) / args.algo, exist_ok=True)
    save_results_jsonl(results_list, ROOT / "results" / str(args.did) / args.algo / f"result_{run_id}.jsonl")

