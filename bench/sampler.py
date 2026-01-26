from __future__ import annotations
import os
from pathlib import Path
import hashlib
import numpy as np
from utils import save_configs_jsonl
from globals import ALGOS
from config_schema import DCConfig, DataConfig, PhaseConfig, TrainConfig, ModelConfig, validate
from search_space import LR_GRID, BATCH_SIZE_GRID, BOTTLE_NECK_GRID, NUMBER_OF_LAYERS_GRID, LAYERS_GRID, EARLY_STOPPING_OPTIONS, ALGO_SPECS
from datasets import get_dataset_metadata, load_dataset, get_ids_from_txt

HERE = Path(__file__).resolve().parent        
ROOT = HERE.parent                            

def get_seed(did: int, algo: str,config_number: int) -> int:
    """
    Generate a deterministic seed based on dataset ID and algorithm name.
    Parameters:
    did (int): Dataset ID.
    algo (str): Algorithm name.
    Returns:
    int: A deterministic seed value.
    """
    seed_str = f"{did}_{algo}_{config_number}"
    seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
    seed = int(seed_hash, 16) % (2**16)  # Limit to 16-bit integer
    return seed

def get_epochs(    
    lr: float,
    *,
    default_epochs: int = 100,
    lr_ref: float = 1e-3,
    C: float = 0.6,
    min_epochs: int = 100,
    max_epochs: int = 2000,
) -> int:
    """
    Map learning rate -> number of epochs.

    Smaller lr -> more epochs
    Larger lr -> fewer epochs
    """
    raw = default_epochs * (lr_ref / lr) ** C
    return int(max(min_epochs, min(max_epochs, round(raw))))

def _valid_batches(n_train: int) -> list[int]:
    """Return list of valid batch sizes for given n_train."""
    return [b for b in BATCH_SIZE_GRID if b <= n_train]

def sample_one_config(
        seed: int,
        did: int,
        algo: str,
        input_dim: int,
        n_clusters: int,
        n_train: int,
) -> DCConfig:
    """Sample one random configuration for given dataset and algorithm."""
    rng = np.random.default_rng(seed)
    data = DataConfig(
        input_dim=input_dim,
        n_clusters=n_clusters,
        did=did,
    )
    
    pretrain_lr = float(rng.choice(LR_GRID))
    clustering_lr = float(rng.choice(LR_GRID))
    pretrain = PhaseConfig(
        lr=pretrain_lr,
        epochs=float(get_epochs(pretrain_lr,default_epochs= 100,lr_ref = 1e-3)),
    )
    clustering = PhaseConfig(
        lr=clustering_lr,
        epochs=float(get_epochs(clustering_lr,default_epochs= 150,lr_ref = 1e-4)),
    )
    train = TrainConfig(
        batch_size=int(rng.choice(_valid_batches(n_train))),
        early_stopping=rng.choice(EARLY_STOPPING_OPTIONS),
    )
    arch_layers = int(rng.choice(NUMBER_OF_LAYERS_GRID))
    arch = [int(rng.choice(LAYERS_GRID)) for _ in range(arch_layers)]
    model = ModelConfig(
        arch=list(arch),
        bottleneck_dim=int(rng.choice(BOTTLE_NECK_GRID)),
        nn_type=ALGO_SPECS[algo],
    )
    config = DCConfig(
        algo=algo,
        data=data,
        pretrain=pretrain,
        clustering=clustering,
        train=train,
        model=model,
        seed=seed,
    )
    validate(config)  
    return config

def sample_for_pair(
        did: int,
        algo: str,
        input_dim: int,
        n_clusters: int,
        n_train: int,
        n_configs: int,
) -> list[DCConfig]:
    """Sample multiple configurations for given dataset and algorithm."""
    configs = []
    count = 0
    seed_count = 0
    while len(configs) < n_configs:
        try:
            seed = get_seed(did, algo,seed_count)
            seed_count += 1
            config = sample_one_config(
                seed=seed,
                did=did,
                algo=algo,
                input_dim=input_dim,
                n_clusters=n_clusters,
                n_train=n_train,
            )
            configs.append(config)
            count += 1
            
        except Exception as e:
            print(f"Warning: Failed to sample config for did={did}, algo={algo}: {e}")
            continue
    return configs

if __name__ == "__main__":
    small_ids_path = ROOT / "small_ids.txt"
    normal_ids_path = ROOT / "normal_ids.txt"
    big_ids_path = ROOT / "big_ids.txt"

    small_ids = get_ids_from_txt(small_ids_path)
    normal_ids = get_ids_from_txt(normal_ids_path)
    big_ids = get_ids_from_txt(big_ids_path)

    for did in small_ids:
        metadata = get_dataset_metadata(did)
        for algo in ALGOS:
            configs = sample_for_pair(
                did=did,
                algo=algo,
                input_dim=metadata["n_features"],
                n_clusters=metadata["n_classes"],
                n_train=metadata["n_samples"],
                n_configs=200,
            )
            os.makedirs(f"./configs/{did}", exist_ok=True)
            os.makedirs(f"./configs/{did}/{algo}", exist_ok=True)
            save_configs_jsonl(configs, f"./configs/{did}/{algo}/configs.jsonl")

    for did in normal_ids:
        metadata = get_dataset_metadata(did)
        for algo in ALGOS:
            configs = sample_for_pair(
                did=did,
                algo=algo,
                input_dim=metadata["n_features"],
                n_clusters=metadata["n_classes"],
                n_train=metadata["n_samples"],
                n_configs=200,
            )
            os.makedirs(f"./configs/{did}", exist_ok=True)
            os.makedirs(f"./configs/{did}/{algo}", exist_ok=True)
            save_configs_jsonl(configs, f"./configs/{did}/{algo}/configs.jsonl")

    for did in big_ids:
        metadata = get_dataset_metadata(did)
        for algo in ALGOS:
            configs = sample_for_pair(
                did=did,
                algo=algo,
                input_dim=metadata["n_features"],
                n_clusters=metadata["n_classes"],
                n_train=metadata["n_samples"],
                n_configs=200,
            )
            os.makedirs(f"./configs/{did}", exist_ok=True)
            os.makedirs(f"./configs/{did}/{algo}", exist_ok=True)
            save_configs_jsonl(configs, f"./configs/{did}/{algo}/configs.jsonl")


    
