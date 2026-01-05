from __future__ import annotations
import hashlib
import numpy as np

from config_schema import DCConfig, DataConfig, PhaseConfig, TrainConfig, ModelConfig, validate
from search_space import LR_GRID, BATCH_SIZE_GRID, BOTTLE_NECK_GRID, NUMBER_OF_LAYERS_GRID, LAYERS_GRID, EARLY_STOPPING_OPTIONS, ALGO_SPECS


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
    print(seed)
    return seed

def get_epochs(    
    lr: float,
    *,
    default_epochs: int = 100,
    lr_ref: float = 1e-3,
    C: float = 0.5,
    min_epochs: int = 100,
    max_epochs: int = 3000,
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
    
    pretrain_lr = rng.choice(LR_GRID)
    clustering_lr = rng.choice(LR_GRID)
    pretrain = PhaseConfig(
        lr=pretrain_lr,
        epochs=get_epochs(pretrain_lr,default_epochs= 100,lr_ref = 1e-3),
    )
    clustering = PhaseConfig(
        lr=clustering_lr,
        epochs=get_epochs(clustering_lr,default_epochs= 150,lr_ref = 1e-4),
    )
    train = TrainConfig(
        batch_size=rng.choice(_valid_batches(n_train)),
        early_stopping=rng.choice(EARLY_STOPPING_OPTIONS),
    )
    arch_layers = rng.choice(NUMBER_OF_LAYERS_GRID)
    arch = [rng.choice(LAYERS_GRID) for _ in range(arch_layers)]
    model = ModelConfig(
        arch=tuple(arch),
        bottleneck_dim=rng.choice(BOTTLE_NECK_GRID),
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
    while len(configs) < n_configs:
        try:
            seed = get_seed(did, algo,count)
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
    # Example usage
    did = 12345
    algo = "IDEC"
    input_dim = 100
    n_clusters = 10
    n_train = 5000
    n_configs = 10

    sampled_configs = sample_for_pair(
        did=did,
        algo=algo,
        input_dim=input_dim,
        n_clusters=n_clusters,
        n_train=n_train,
        n_configs=n_configs,
    )

    for i, config in enumerate(sampled_configs):
        print(f"Config {i+1}:\n{config.to_dict()}\n")
    
