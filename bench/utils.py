import json
from pathlib import Path
import numpy as np
from config_schema import DCConfig

HERE = Path(__file__).resolve().parent        
ROOT = HERE.parent       

def save_configs_jsonl(configs: list[DCConfig], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for cfg in configs:
            row = {
                "run_id": cfg.run_id(),   # stable id for resume/dedup
                "did": cfg.data.did,
                "algo": cfg.algo,
                "seed": cfg.seed,
                "config": cfg.to_dict(),  # exactly what DeepClusteringAlgo expects
            }
            f.write(json.dumps(row) + "\n")

def get_config_file_from_number(did: int, algo: str, config_number: int,config_parent_dir: str = ROOT / "configs/") -> str:
    config_path = Path(config_parent_dir) / str(did) / algo / "configs.jsonl"
    with config_path.open("r") as f:
        for i, line in enumerate(f):
            if i == config_number:
                data = json.loads(line)
                return data
    raise ValueError(f"Config number {config_number} not found for did={did}, algo={algo}.")

def split_indices(n, seed, val_ratio=0.2):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(n * val_ratio)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

def save_results_jsonl(results: list[dict], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    # Example usage
    did = 36
    algo = "DDC"
    config_number = 0
    config = get_config_file_from_number(did, algo, config_number)
    print(json.dumps(config, indent=2))