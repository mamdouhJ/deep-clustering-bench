#!/usr/bin/env python3
"""Build a meta-dataset from all available benchmarking results.

The script scans:
- results/<did>/<algo>/result_*.jsonl
- configs/<did>/<algo>/configs.jsonl

and outputs one row per run line with flattened configuration + metric fields.
"""

from __future__ import annotations
import time
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"
DEFAULT_OUT = ROOT / "data" / "meta_dataset.jsonl"

# Reuse shared helpers from bench/utils.py.
sys.path.insert(0, str(ROOT / "bench"))
from utils import flatten_dict, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a meta-dataset from benchmark results.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Path to results directory.")
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR, help="Path to configs directory.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output file path.")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Also write aggregated rows with mean/std metrics per (did, algo, config_number).",
    )
    parser.add_argument(
        "--aggregate-out",
        type=Path,
        default=None,
        help="Aggregated output path (default: <out_stem>_agg.<ext>).",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format.",
    )
    return parser.parse_args()


def load_config_indices(configs_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    by_line: list[dict[str, Any]] = []

    for idx, cfg_row in enumerate(read_jsonl(configs_path)):
        cfg_row["_config_number"] = idx
        run_id = cfg_row.get("run_id")
        if isinstance(run_id, str):
            by_run_id[run_id] = cfg_row
        by_line.append(cfg_row)

    return by_run_id, by_line


def find_matching_config(
    result_row: dict[str, Any],
    configs_by_run_id: dict[str, dict[str, Any]],
    configs_by_line: list[dict[str, Any]],
) -> dict[str, Any] | None:
    run_id = result_row.get("run_id")
    if isinstance(run_id, str) and run_id in configs_by_run_id:
        return configs_by_run_id[run_id]

    config_number = result_row.get("config_number")
    if isinstance(config_number, int) and 0 <= config_number < len(configs_by_line):
        return configs_by_line[config_number]

    return None


def build_rows(results_dir: Path, configs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []

    stats = {
        "result_files": 0,
        "result_lines": 0,
        "rows_written": 0,
        "missing_config": 0,
        "bad_json_lines": 0,
    }

    for result_path in sorted(results_dir.glob("*/*/result_*.jsonl")):
        stats["result_files"] += 1
        t0 = time.time()
        try:
            did = int(result_path.parts[-3])
        except (ValueError, IndexError):
            # Unexpected layout; skip safely.
            continue

        algo = result_path.parts[-2]
        configs_path = configs_dir / str(did) / algo / "configs.jsonl"
        if not configs_path.exists():
            continue

        configs_by_run_id, configs_by_line = load_config_indices(configs_path)

        result_rows: list[dict[str, Any]] = []
        try:
            result_rows = read_jsonl(result_path)
        except json.JSONDecodeError:
            # Keep behavior explicit for corrupted files: skip entire file.
            stats["bad_json_lines"] += 1
            continue

        for result_row in result_rows:
            stats["result_lines"] += 1

            matched_config = find_matching_config(result_row, configs_by_run_id, configs_by_line)
            if matched_config is None:
                stats["missing_config"] += 1
                continue

            row: dict[str, Any] = {
                "did": result_row.get("did", did),
                "algo": result_row.get("algo", algo),
                "run_id": result_row.get("run_id"),
                "config_number": result_row.get("config_number", matched_config.get("_config_number")),
                "seed": result_row.get("seed"),
                "time_taken": result_row.get("time_taken"),
                "result_file": str(result_path.relative_to(ROOT)),
            }

            metrics = result_row.get("metrics") or {}
            if isinstance(metrics, dict):
                for k, v in flatten_dict(metrics, prefix="metrics").items():
                    row[k] = v

            config = matched_config.get("config") or {}
            if isinstance(config, dict):
                for k, v in flatten_dict(config, prefix="config").items():
                    row[k] = v

            rows.append(row)
            stats["rows_written"] += 1
            print(f"[TIMING] {result_path} — {len(result_rows)} rows in {time.time()-t0:.2f}s", flush=True)

    return rows, stats


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("did"), row.get("algo"), row.get("config_number"))
        grouped[key].append(row)

    out_rows: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        first = group_rows[0]
        agg_row: dict[str, Any] = {
            "did": first.get("did"),
            "algo": first.get("algo"),
            "config_number": first.get("config_number"),
            "run_id": first.get("run_id"),
            "num_runs": len(group_rows),
        }

        # Keep configuration context in the aggregated output.
        for key, value in first.items():
            if key.startswith("config."):
                agg_row[key] = value

        metric_keys = sorted(
            key for key in first.keys() if key.startswith("metrics.") and is_numeric(first.get(key))
        )
        for key in metric_keys:
            vals = [row[key] for row in group_rows if is_numeric(row.get(key))]
            if not vals:
                continue
            agg_row[f"{key}.mean"] = statistics.mean(vals)
            agg_row[f"{key}.std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        time_vals = [row["time_taken"] for row in group_rows if is_numeric(row.get("time_taken"))]
        if time_vals:
            agg_row["time_taken.mean"] = statistics.mean(time_vals)
            agg_row["time_taken.std"] = statistics.stdev(time_vals) if len(time_vals) > 1 else 0.0

        out_rows.append(agg_row)

    out_rows.sort(key=lambda r: (r.get("did"), r.get("algo"), r.get("config_number")))
    return out_rows


def resolve_aggregate_out_path(out_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}_agg{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}_agg")


def resolve_out_path(out_path: Path, output_format: str) -> Path:
    expected_suffix = ".csv" if output_format == "csv" else ".jsonl"
    if out_path.suffix.lower() == expected_suffix:
        return out_path
    if out_path.suffix:
        return out_path.with_suffix(expected_suffix)
    return out_path.with_name(f"{out_path.name}{expected_suffix}")


def write_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, stats = build_rows(args.results_dir, args.configs_dir)
    out_path = resolve_out_path(args.out, args.format)

    if args.format == "jsonl":
        write_jsonl(rows, out_path)
    else:
        write_csv(out_path, rows)

    print(f"[OK] Wrote {len(rows)} rows to {out_path}")
    print(f"[INFO] result_files={stats['result_files']} result_lines={stats['result_lines']}")
    print(
        "[INFO] "
        f"rows_written={stats['rows_written']} missing_config={stats['missing_config']} bad_json_lines={stats['bad_json_lines']}"
    )

    if args.aggregate:
        agg_rows = aggregate_rows(rows)
        agg_out = resolve_aggregate_out_path(out_path, args.aggregate_out)
        if args.format == "jsonl":
            write_jsonl(agg_rows, agg_out)
        else:
            write_csv(agg_out, agg_rows)
        print(f"[OK] Wrote {len(agg_rows)} aggregated rows to {agg_out}")


if __name__ == "__main__":
    main()
