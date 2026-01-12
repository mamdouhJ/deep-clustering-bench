
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ALGOS = ["DDC", "IDEC", "DipEncoder", "VaDE", "DeepECT"]

def count_lines(p: Path) -> int:
    with p.open("r") as f:
        return sum(1 for _ in f)

def main():
    out_path = ROOT / "tasks" / "tasks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_tasks = 0
    with out_path.open("w") as out:
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
                if n_cfgs != 200:
                    print(f"Warning: Expected 200 configs for did={did}, algo={algo}, but found {n_cfgs}")
                for config_number in range(n_cfgs):
                    out.write(json.dumps({
                        "did": did,
                        "algo": algo,
                        "config_number": config_number,
                    }) + "\n")
                    n_tasks += 1

    print(f"Wrote {n_tasks} tasks to {out_path}")

if __name__ == "__main__":
    main()
