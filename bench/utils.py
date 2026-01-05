import json
from pathlib import Path
from config_schema import DCConfig
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