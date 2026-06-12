#!/usr/bin/env python
"""Run BrainGNN main, ablation, and lambda-sweep feasibility experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--experiment", choices=["main", "loss_ablation", "conv_ablation", "lambda_sweep"],
                        default="main")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configs(experiment: str) -> list[dict[str, object]]:
    full = {"unit": 1.0, "tpk": 0.1, "glc": 0.1, "conv": "ra"}
    if experiment == "main":
        return [{"name": "paper_like", **full}]
    if experiment == "loss_ablation":
        return [
            {"name": "ce_only", "unit": 0.0, "tpk": 0.0, "glc": 0.0, "conv": "ra"},
            {"name": "ce_unit", "unit": 1.0, "tpk": 0.0, "glc": 0.0, "conv": "ra"},
            {"name": "ce_unit_tpk", "unit": 1.0, "tpk": 0.1, "glc": 0.0, "conv": "ra"},
            {"name": "ce_unit_glc", "unit": 1.0, "tpk": 0.0, "glc": 0.1, "conv": "ra"},
            {"name": "full", **full},
        ]
    if experiment == "conv_ablation":
        return [{"name": conv, **full, "conv": conv} for conv in ("ra", "vanilla")]
    values2 = [0.0, 0.1, 0.2, 0.5, 1.0]
    values1 = [0.0, 0.1, 0.2, 0.5]
    result = []
    for tpk in (0.0, 0.1):
        for glc in values2:
            result.append({"name": f"tpk_{tpk:g}_glc_{glc:g}", "unit": 1.0, "tpk": tpk, "glc": glc, "conv": "ra"})
    for tpk in values1:
        result.append({"name": f"tpk_{tpk:g}_glc_0.1", "unit": 1.0, "tpk": tpk, "glc": 0.1, "conv": "ra"})
    return list({item["name"]: item for item in result}.values())


def aggregate(output_root: Path, experiment: str, experiment_configs: list[dict[str, object]]) -> dict[str, object]:
    aggregate_result: dict[str, object] = {"experiment": experiment, "configs": {}}
    for config in experiment_configs:
        rows = []
        for summary_path in sorted((output_root / str(config["name"])).glob("fold_*/summary.json")):
            with summary_path.open() as handle:
                rows.append(json.load(handle))
        metrics: dict[str, dict[str, float]] = {}
        if rows:
            for key in rows[0]["metrics"]:
                values = np.asarray([float(row["metrics"][key]) for row in rows])
                metrics[key] = {"mean": float(values.mean()), "std": float(values.std())}
        aggregate_result["configs"][str(config["name"])] = {
            "parameters": config,
            "completed_folds": len(rows),
            "metrics": metrics,
        }
    with (output_root / "aggregate.json").open("w") as handle:
        json.dump(aggregate_result, handle, indent=2, sort_keys=True)
    return aggregate_result


def main() -> None:
    args = parse_args()
    experiment_configs = configs(args.experiment)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for config in experiment_configs:
        for fold in args.folds:
            output_dir = args.output_root / str(config["name"]) / f"fold_{fold}"
            if (output_dir / "summary.json").exists() and not args.force:
                print(f"skip completed {config['name']} fold={fold}", flush=True)
                continue
            command = [
                sys.executable,
                "-u",
                "07-main_hcp_subjectwise.py",
                "--dataroot", str(args.dataroot),
                "--split_manifest", str(args.split_manifest),
                "--fold", str(fold),
                "--n_epochs", str(args.epochs),
                "--batchSize", str(args.batch_size),
                "--edge_source", "pcorr",
                "--edge_top_percent", "0.1",
                "--positive_edges_only",
                "--lamb1", str(config["unit"]),
                "--lamb2", str(config["unit"]),
                "--lamb3", str(config["tpk"]),
                "--lamb4", str(config["tpk"]),
                "--lamb5", str(config["glc"]),
                "--conv_type", str(config["conv"]),
                "--class_weight", args.class_weight,
                "--best_metric", "balanced_acc",
                "--output_dir", str(output_dir),
            ]
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "stdout.log").open("w") as stdout:
                print(f"running {config['name']} fold={fold}", flush=True)
                subprocess.run(command, check=True, stdout=stdout, stderr=subprocess.STDOUT)
            aggregate(args.output_root, args.experiment, experiment_configs)
    print(json.dumps(aggregate(args.output_root, args.experiment, experiment_configs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
