from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if TYPE_CHECKING:
    from config_schema import DCConfig

HERE = Path(__file__).resolve().parent        
ROOT = HERE.parent       


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, flat_key))
        elif isinstance(value, list):
            out[flat_key] = json.dumps(value, separators=(",", ":"))
        else:
            out[flat_key] = value
    return out

def save_configs_jsonl(configs: list["DCConfig"], out_path: str) -> None:
    rows = []
    for cfg in configs:
        row = {
            "run_id": cfg.run_id(),   # stable id for resume/dedup
            "did": cfg.data.did,
            "algo": cfg.algo,
            "seed": cfg.seed,
            "config": cfg.to_dict(),  # exactly what DeepClusteringAlgo expects
        }
        rows.append(row)
    write_jsonl(rows, out_path)

def get_config_file_from_number(did: int, algo: str, config_number: int,config_parent_dir: str = ROOT / "configs/") -> str:
    config_path = Path(config_parent_dir) / str(did) / algo / "configs.jsonl"
    with config_path.open("r") as f:
        for i, line in enumerate(f):
            if i == config_number:
                data = json.loads(line)
                return data
    raise ValueError(f"Config number {config_number} not found for did={did}, algo={algo}.")

def split_indices(n, seed, val_ratio=0.2):
    if np is None:
        raise ModuleNotFoundError("numpy is required for split_indices.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(n * val_ratio)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

def save_results_jsonl(results: list[dict], out_path: str) -> None:
    write_jsonl(results, out_path)

def save_results_npz(results: dict, out_path: str) -> None:
    if np is None:
        raise ModuleNotFoundError("numpy is required for save_results_npz.")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **results)

if __name__ == "__main__":
    # Example usage
    did = 36
    algo = "DDC"
    config_number = 0
    config = get_config_file_from_number(did, algo, config_number)
    print(json.dumps(config, indent=2))
