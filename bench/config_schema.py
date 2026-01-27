from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional
import json
import hashlib


# --- Types ---
AlgoName = Literal["IDEC", "DDC", "DeepECT", "DipEncoder", "VaDE"]
NNType = Literal["ae"]  # extend if needed


# --- Sections matching your dict ---
@dataclass(frozen=True)
class DataConfig:
    input_dim: int
    n_clusters: int
    did: int

@dataclass(frozen=True)
class PhaseConfig:
    lr: float
    epochs: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    early_stopping: bool



@dataclass(frozen=True)
class ModelConfig:
    arch: tuple[int, ...]     
    bottleneck_dim: int
    nn_type: NNType


@dataclass(frozen=True)
class DCConfig:
    """
    Structured config that can export to the exact dict expected by DeepClusteringAlgo.
    """
    algo: AlgoName
    data: DataConfig
    pretrain: PhaseConfig
    clustering: PhaseConfig
    train: TrainConfig
    model: ModelConfig
    seed: int 

    def to_dict(self) -> dict[str, Any]:
        """
        Export to the flat dict format expected by your DeepClusteringAlgo wrapper.
        """
        return {
            "algo": self.algo,
            "seed": int(self.seed),
            "data": asdict(self.data),
            "pretrain": asdict(self.pretrain),
            "clustering": asdict(self.clustering),
            "batch_size": int(self.train.batch_size),
            "early_stopping": bool(self.train.early_stopping),
            "arch": list(self.model.arch),
            "bottleneck_dim": int(self.model.bottleneck_dim),
            "nn_type": self.model.nn_type,
        }

    def run_id(self) -> str:
        """
        Stable identifier for resume/dedup. Based on the exported dict.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# --- Validation rules (this is the whole point) ---
def validate(cfg: DCConfig, algo_specs: dict[str, str] | None = None) -> None:
    """
    Raise ValueError if the config is invalid.
    algo_specs: optional mapping like {"IDEC":"ae", "VaDE":"vae", ...}
    """
    # Data sanity
    if cfg.data.input_dim <= 0:
        raise ValueError("data.input_dim must be > 0")
    if cfg.data.n_clusters <= 1:
        raise ValueError("data.n_clusters must be > 1")

    # Phase sanity
    for phase_name, phase in [("pretrain", cfg.pretrain), ("clustering", cfg.clustering)]:
        if phase.lr <= 0:
            raise ValueError(f"{phase_name}.lr must be > 0")
        if phase.epochs <= 0:
            raise ValueError(f"{phase_name}.epochs must be > 0")

    # Training sanity
    if cfg.train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0")

    # Model sanity
    if len(cfg.model.arch) < 1:
        raise ValueError("model.arch must have at least one layer")
    if any(h <= 0 for h in cfg.model.arch):
        raise ValueError("model.arch layers must all be > 0")
    if cfg.model.bottleneck_dim <= 0:
        raise ValueError("model.bottleneck_dim must be > 0")
    #if cfg.algo == "DDC" and cfg.train.early_stopping == True:
    #    raise ValueError("DDC does not support early_stopping=True")

    # Common-sense AE/VAE shape rule (optional but very helpful)
    # Bottleneck should usually be smaller than smallest hidden layer.
    if cfg.model.bottleneck_dim >= min(cfg.model.arch):
        raise ValueError(
            f"model.bottleneck_dim ({cfg.model.bottleneck_dim}) should be < min(arch) ({min(cfg.model.arch)})"
        )

    # Algo ↔ nn_type compatibility
    if algo_specs is not None:
        expected = algo_specs.get(cfg.algo)
        if expected is None:
            raise ValueError(f"algo_specs has no entry for algo={cfg.algo}")
        if cfg.model.nn_type != expected:
            raise ValueError(f"{cfg.algo} requires nn_type='{expected}', got '{cfg.model.nn_type}'")
    else:
        # Fallback rule if you don't pass algo_specs
        if cfg.algo == "VaDE" and cfg.model.nn_type != "vae":
            raise ValueError("VaDE requires nn_type='vae'")
        if cfg.algo != "VaDE" and cfg.model.nn_type != "ae":
            raise ValueError(f"{cfg.algo} requires nn_type='ae'")

if __name__ == "__main__":
    # Simple self-test
    cfg = DCConfig(
        algo="IDEC",
        data=DataConfig(input_dim=20, n_clusters=5, did=42),
        pretrain=PhaseConfig(lr=1e-3, epochs=10),
        clustering=PhaseConfig(lr=1e-3, epochs=10),
        train=TrainConfig(batch_size=64, early_stopping=True),
        model=ModelConfig(arch=(100, 50), bottleneck_dim=10, nn_type="ae"),
        seed=1234,
    )
    print("Config dict:", cfg.to_dict())
    print("Run ID:", cfg.run_id())
    validate(cfg, algo_specs={"IDEC": "ae", "VaDE": "vae"})
    print("Validation passed.")