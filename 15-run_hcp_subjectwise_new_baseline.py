#!/usr/bin/env python
"""Run and summarize the current HCP subject-wise BrainGNN baseline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATAROOT = PROJECT_ROOT / "data" / "HCP900_subjectwise_qc_allsub_7task_lrrl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "hcp900_subjectwise_new_baseline_pcorr_pos10_5fold",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_fold(args: argparse.Namespace, fold: int) -> None:
    fold_dir = args.output_dir / f"fold_{fold}"
    summary_path = fold_dir / "summary.json"
    if args.skip_completed and summary_path.exists():
        print(f"Skipping completed fold {fold}: {summary_path}", flush=True)
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "07-main_hcp_subjectwise.py"),
        "--dataroot", str(args.dataroot),
        "--fold", str(fold),
        "--n_epochs", str(args.n_epochs),
        "--batchSize", "64",
        "--lr", "0.001",
        "--edge_source", "pcorr",
        "--edge_top_percent", "0.10",
        "--positive_edges_only",
        "--best_metric", "balanced_acc",
        "--output_dir", str(fold_dir),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def summarize(args: argparse.Namespace) -> None:
    rows = []
    for fold in args.folds:
        summary_path = args.output_dir / f"fold_{fold}" / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as handle:
            summary = json.load(handle)
        rows.append({"fold": fold, **summary["metrics"]})
    if not rows:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "macro_recall", "macro_precision"]
    aggregate = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows])),
        }
        for metric in metric_names
    }
    payload = {
        "experiment": "hcp900_subjectwise_new_baseline",
        "completed_folds": len(rows),
        "folds": rows,
        "aggregate": aggregate,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", *metric_names], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    for fold in args.folds:
        run_fold(args, fold)
    summarize(args)


if __name__ == "__main__":
    main()
