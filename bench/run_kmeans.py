import os
import time
import numpy as np
import argparse
import atexit
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from sklearn.metrics import davies_bouldin_score, silhouette_score
from clustpy.metrics import unsupervised_clustering_accuracy as uca

from datasets import load_dataset
from utils import save_results_jsonl, save_results_npz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--did", type=int, required=True, help="Dataset ID (OpenML ID)")
    args = parser.parse_args()

    seeds = [42, 75, 128, 256, 1000]
    algo_name = "KMeans"
    
    results_dir = ROOT / "results" / str(args.did) / algo_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "result_baseline.jsonl"
    lock_path = results_dir / "result_baseline.lock"

    if results_path.exists():
        print(f"[SKIP] Results already exist: {results_path}")
        return

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()}\ndid={args.did}\nalgo={algo_name}\n")
    except FileExistsError:
        print(f"[SKIP] Lock exists: {lock_path}")
        return

    def _cleanup_lock():
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(_cleanup_lock)

    results_list = []
    try:
        X, y = load_dataset(args.did)
        c = np.unique(y).shape[0]
        print(f"[LOAD] did={args.did} | X={X.shape} | y={y.shape} | c={c}")
        
        if c <= 1:
            print(f"[WARNING] Dataset {args.did} has only 1 class. Skipping.")
            return

        for i, run_seed in enumerate(seeds):
            print(f"[RUN] did={args.did} run={i+1}/{len(seeds)} seed={run_seed}")

            t0 = time.time()
            kmeans = KMeans(n_clusters=c, random_state=run_seed, n_init=10).fit(X)
            time_taken = time.time() - t0

            labels = kmeans.labels_
            
            metrics = {
                "ARI": ari(y, labels),
                "NMI": nmi(y, labels),
                "UCA": uca(y, labels),
                "Davies_Bouldin": davies_bouldin_score(X, labels),
                "Silhouette": silhouette_score(X, labels)
            }
            
            results_list.append({
                "did": args.did,
                "algo": algo_name,
                "seed": run_seed,
                "metrics": metrics,
                "time_taken": time_taken,
            })
            
            artifacts = {
                "labels": labels,
                "cluster_centers": kmeans.cluster_centers_
            }
            save_results_npz(artifacts, results_dir / f"artifacts_baseline_run{i+1}.npz")

        save_results_jsonl(results_list, results_path)
        print(f"[OK] Wrote metrics: {results_path}")

    finally:
        _cleanup_lock()

if __name__ == "__main__":
    main()