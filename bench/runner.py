import os
import time
import numpy as np
import argparse
import atexit
from pathlib import Path

from datasets import load_dataset
from utils import get_config_file_from_number, split_indices, save_results_jsonl, save_results_npz
from algorithms import DeepClusteringAlgorithm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(
        description="Given dataset ID, algorithm name, and config number, run the deep clustering algorithm on the dataset."
    )
    parser.add_argument("--did", type=int, required=True, help="Dataset ID (OpenML ID)")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["DDC", "IDEC", "DipEncoder", "VaDE", "DeepECT"],
        help="Algorithm name",
    )
    parser.add_argument("--config_number", type=int, required=True, help="Configuration number (line number in the config file)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to execute with different seeds")
    args = parser.parse_args()
    #create a namespace for debugging
    #args = argparse.Namespace(did=44666, algo="VaDE", config_number=20, num_runs=5)
    config_file = get_config_file_from_number(args.did, args.algo, args.config_number)
    run_id = config_file["run_id"]
    config = config_file["config"]

    # Result paths
    results_dir = ROOT / "results" / str(args.did) / args.algo
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"result_{run_id}.jsonl"
    lock_path = results_dir / f"result_{run_id}.lock"

    # 1) Skip if already done
    if results_path.exists():
        print(f"[SKIP] Results already exist: {results_path}")
        return
    #skip if there exists 200 results files for the same dataset and algorithm
    existing_results = list(results_dir.glob(f"result_*.jsonl"))
    if len(existing_results) >= 200:
        print(f"[SKIP] Too many existing results for this config: {len(existing_results)} results found in {results_dir}")
        return
    # 2) Atomic lock to prevent duplicate concurrent runs
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"did={args.did}\n")
            f.write(f"algo={args.algo}\n")
            f.write(f"config_number={args.config_number}\n")
            f.write(f"run_id={run_id}\n")
    except FileExistsError:
        print(f"[SKIP] Lock exists (another worker is running this): {lock_path}")
        return

    def _cleanup_lock():
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(_cleanup_lock)

    # 3) Run

    t0 = time.time()
    results_list = []
    artifacts_list = [] 
    try:
        X, y = load_dataset(args.did)
        print(f"[LOAD] Loaded dataset did={args.did} with X.shape={X.shape} y.shape={y.shape}")
        print(f"The uniqure labels are: {np.unique(y)}")

        base_seed = int(config["seed"])  # original seed from config

        for i in range(args.num_runs):
            run_seed = base_seed + i + 1
            config["seed"] = run_seed

            print(f"[RUN] did={args.did} algo={args.algo} cfg={args.config_number} run={i+1}/{args.num_runs} seed={run_seed}")

            model = DeepClusteringAlgorithm(config)

            if config.get("early_stopping", False):
                # IMPORTANT: split should depend on current run_seed
                train_idx, val_idx = split_indices(X.shape[0], run_seed, val_ratio=0.2)
                model.fit(X[train_idx], val_set=X[val_idx])
            else:
                model.fit(X)
            print(f"the unique pseudo-labels are: {np.unique(model.model.labels_)}")
            print(f"The shape of the pseudo labels is: {model.model.labels_.shape}")
            print(model.model.labels_)
            metrics = model.get_metrics(y, X)
            
            results_list.append({
                "run_id": run_id,
                "did": args.did,
                "algo": args.algo,
                "config_number": args.config_number,
                "seed": run_seed,
                "metrics": metrics,
                "time_taken": time.time() - t0,
            })
            #artifacts_list.append(model.get_artifacts())
            save_results_npz(model.get_artifacts(), results_dir / f"artifacts_{run_id}_run{i+1}.npz")

        # Write once at the end
        save_results_jsonl(results_list, results_path)
        print(f"[OK] Wrote results: {results_path}")
        save_results_npz(model.get_artifacts(), results_dir / f"artifacts_{run_id}.npz")
    finally:
        _cleanup_lock()


if __name__ == "__main__":
    main()
