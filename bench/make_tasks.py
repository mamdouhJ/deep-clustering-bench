import json
import argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ALGOS = ["DDC", "IDEC", "DipEncoder", "VaDE", "DeepECT"]

def count_lines(p: Path) -> int:
    with p.open("r") as f:
        return sum(1 for _ in f)

def main(seed: int = 0):
    out_path = ROOT / "tasks" / "tasks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    configs_root = ROOT / "configs"

    for did_dir in configs_root.iterdir():
        if not did_dir.is_dir():
            continue
        did = int(did_dir.name)

        for algo in ALGOS:
            cfg_path = did_dir / algo / "configs.jsonl"
            if not cfg_path.exists():
                continue

            n_cfgs = count_lines(cfg_path)
            for config_number in range(n_cfgs):
                tasks.append({"did": did, "algo": algo, "config_number": config_number})

    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)

    with out_path.open("w") as out:
        for t in tasks:
            out.write(json.dumps(t) + "\n")

    print(f"Wrote {len(tasks)} shuffled tasks to {out_path} (seed={seed})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(seed=args.seed)
