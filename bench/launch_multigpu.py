# bench/launch_multigpu.py
from __future__ import annotations

import json
import os
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

import fcntl  # Linux file locking (works on Slurm nodes)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TASKS_FILE = ROOT / "tasks" / "tasks.jsonl"
STATE_FILE = ROOT / "tasks" / "state.json"
QUEUE_LOCK = ROOT / "tasks" / "queue.lock"

LOG_DIR = ROOT / "logs" / "workers"

RUNNER_CMD = [os.environ.get("PYTHON", "python"), str(HERE / "runner.py")]


def init_state(reset: bool = False) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if reset or not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({"next_idx": 0}))


def claim_next_task() -> dict | None:
    """
    Atomically claim the next task by incrementing state["next_idx"] under a lock.
    Returns None when tasks are exhausted.
    """
    QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)

    with open(QUEUE_LOCK, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            state = json.loads(STATE_FILE.read_text())
            idx = int(state.get("next_idx", 0))

            task = None
            with TASKS_FILE.open("r") as f:
                for i, line in enumerate(f):
                    if i == idx:
                        task = json.loads(line)
                        break

            if task is None:
                return None  # no more tasks

            state["next_idx"] = idx + 1
            STATE_FILE.write_text(json.dumps(state))
            return task
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def worker(gpu_id: int, num_runs: int) -> None:
    """
    One worker bound to one GPU. Claims tasks until exhausted.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = LOG_DIR / f"gpu{gpu_id}.log"
    with log_path.open("a") as log:
        log.write(f"[START] pid={os.getpid()} gpu={gpu_id}\n")
        log.flush()

        while True:
            task = claim_next_task()
            if task is None:
                log.write("[DONE] no more tasks\n")
                log.flush()
                return

            did = task["did"]
            algo = task["algo"]
            cfg = task["config_number"]

            cmd = RUNNER_CMD + [
                "--did", str(did),
                "--algo", str(algo),
                "--config_number", str(cfg),
                "--num_runs", str(num_runs),
            ]

            t0 = time.time()
            p = subprocess.run(
                cmd,
                env=env,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            dt = time.time() - t0

            if p.returncode != 0:
                log.write(f"[FAIL] did={did} algo={algo} cfg={cfg} dt={dt:.1f}s rc={p.returncode}\n")
                log.write(p.stdout + "\n")
                log.write(p.stderr + "\n")
            else:
                log.write(f"[OK] did={did} algo={algo} cfg={cfg} dt={dt:.1f}s\n")
            log.flush()


def main(num_gpus: int = 10, num_runs: int = 5, reset_state: bool = False) -> None:
    if not TASKS_FILE.exists():
        raise FileNotFoundError(f"Missing tasks file: {TASKS_FILE}. Run: python bench/make_tasks.py")

    init_state(reset=reset_state)

    procs: list[Process] = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker, args=(gpu_id, num_runs))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--num_gpus", type=int, default=10)
    ap.add_argument("--num_runs", type=int, default=5)
    ap.add_argument("--reset_state", action="store_true", help="Reset queue to start from the first task")
    args = ap.parse_args()

    main(num_gpus=args.num_gpus, num_runs=args.num_runs, reset_state=args.reset_state)
