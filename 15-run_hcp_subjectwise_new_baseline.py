#!/usr/bin/env python
"""Run and summarize the current HCP subject-wise BrainGNN experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATAROOT = PROJECT_ROOT / "data" / "HCP900_subjectwise_qc_allsub_7task_lrrl"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "hcp900_subjectwise_paper_reproduction_current_20260612"
)
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "macro_recall", "macro_precision")


def config(name: str, unit: float, tpk: float, glc: float, conv: str = "ra",
           fc_dim: int = 512) -> dict[str, object]:
    return {
        "name": name,
        "unit": unit,
        "tpk": tpk,
        "glc": glc,
        "conv": conv,
        "fc_dim": fc_dim,
    }


CANONICAL_CONFIGS = {
    item["name"]: item
    for item in [
        config("paper_like", 1.0, 0.1, 0.1),
        config("vanilla", 1.0, 0.1, 0.1, conv="vanilla"),
        config("ce_only", 0.0, 0.0, 0.0),
        config("ce_unit", 1.0, 0.0, 0.0),
        config("ce_unit_tpk", 1.0, 0.1, 0.0),
        config("ce_unit_glc", 1.0, 0.0, 0.1),
        config("tpk_0_glc_0.2", 1.0, 0.0, 0.2),
        config("tpk_0_glc_0.5", 1.0, 0.0, 0.5),
        config("tpk_0_glc_1", 1.0, 0.0, 1.0),
        config("tpk_0.1_glc_0.2", 1.0, 0.1, 0.2),
        config("tpk_0.1_glc_0.5", 1.0, 0.1, 0.5),
        config("tpk_0.1_glc_1", 1.0, 0.1, 1.0),
        config("tpk_0.2_glc_0.1", 1.0, 0.2, 0.1),
        config("tpk_0.5_glc_0.1", 1.0, 0.5, 0.1),
        config("paper_96k", 1.0, 0.1, 0.1, fc_dim=1472),
    ]
}

# Multiple paper tables contain identical settings. Each unique setting is trained
# once and referenced by every relevant aggregate.
EXPERIMENT_VIEWS = {
    "main": {
        "paper_like": "paper_like",
    },
    "loss_ablation": {
        "ce_only": "ce_only",
        "ce_unit": "ce_unit",
        "ce_unit_tpk": "ce_unit_tpk",
        "ce_unit_glc": "ce_unit_glc",
        "full": "paper_like",
    },
    "conv_ablation": {
        "ra": "paper_like",
        "vanilla": "vanilla",
    },
    "lambda_sweep": {
        "tpk_0_glc_0": "ce_unit",
        "tpk_0_glc_0.1": "ce_unit_glc",
        "tpk_0_glc_0.2": "tpk_0_glc_0.2",
        "tpk_0_glc_0.5": "tpk_0_glc_0.5",
        "tpk_0_glc_1": "tpk_0_glc_1",
        "tpk_0.1_glc_0": "ce_unit_tpk",
        "tpk_0.1_glc_0.1": "paper_like",
        "tpk_0.1_glc_0.2": "tpk_0.1_glc_0.2",
        "tpk_0.1_glc_0.5": "tpk_0.1_glc_0.5",
        "tpk_0.1_glc_1": "tpk_0.1_glc_1",
        "tpk_0.2_glc_0.1": "tpk_0.2_glc_0.1",
        "tpk_0.5_glc_0.1": "tpk_0.5_glc_0.1",
    },
    "capacity": {
        "paper_like_55k": "paper_like",
        "paper_96k": "paper_96k",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--experiment",
        choices=["all", *EXPERIMENT_VIEWS],
        default="all",
        help="paper experiment group to run; identical configurations are trained only once",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=1, help="concurrent fold-training processes")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_views(experiment: str) -> dict[str, dict[str, str]]:
    if experiment == "all":
        return EXPERIMENT_VIEWS
    return {experiment: EXPERIMENT_VIEWS[experiment]}


def selected_configs(experiment: str) -> list[dict[str, object]]:
    names = {
        canonical_name
        for view in selected_views(experiment).values()
        for canonical_name in view.values()
    }
    return [CANONICAL_CONFIGS[name] for name in sorted(names)]


def fold_command(args: argparse.Namespace, experiment_config: dict[str, object],
                 fold: int, fold_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "07-main_hcp_subjectwise.py"),
        "--dataroot", str(args.dataroot),
        "--fold", str(fold),
        "--n_epochs", str(args.n_epochs),
        "--batchSize", str(args.batch_size),
        "--lr", "0.001",
        "--stepsize", "20",
        "--gamma", "0.5",
        "--weightdecay", "0.005",
        "--edge_source", "pcorr",
        "--edge_top_percent", "0.10",
        "--positive_edges_only",
        "--best_metric", "balanced_acc",
        "--lamb1", str(experiment_config["unit"]),
        "--lamb2", str(experiment_config["unit"]),
        "--lamb3", str(experiment_config["tpk"]),
        "--lamb4", str(experiment_config["tpk"]),
        "--lamb5", str(experiment_config["glc"]),
        "--conv_type", str(experiment_config["conv"]),
        "--fc_dim", str(experiment_config["fc_dim"]),
        "--seed", str(args.seed),
        "--output_dir", str(fold_dir),
    ]
    command.append("--save_model" if args.save_model else "--no-save_model")
    return command


def run_fold(args: argparse.Namespace, experiment_config: dict[str, object], fold: int) -> str:
    fold_dir = args.output_dir / "runs" / str(experiment_config["name"]) / f"fold_{fold}"
    summary_path = fold_dir / "summary.json"
    if args.skip_completed and summary_path.exists():
        return f"skip completed {experiment_config['name']} fold={fold}"
    command = fold_command(args, experiment_config, fold, fold_dir)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return f"dry run {experiment_config['name']} fold={fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (fold_dir / "stdout.log").open("w") as stdout:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=stdout, stderr=subprocess.STDOUT)
    return f"completed {experiment_config['name']} fold={fold}"


def summarize_run(run_root: Path, folds: list[int]) -> dict[str, object]:
    rows = []
    for fold in folds:
        summary_path = run_root / f"fold_{fold}" / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as handle:
            summary = json.load(handle)
        rows.append({"fold": fold, **summary["metrics"]})
    metrics = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows])),
        }
        for metric in METRIC_NAMES
    } if rows else {}
    return {"completed_folds": len(rows), "folds": rows, "metrics": metrics}


def write_aggregates(args: argparse.Namespace) -> dict[str, object]:
    aggregate_dir = args.output_dir / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    all_aggregates: dict[str, object] = {}
    for experiment, view in selected_views(args.experiment).items():
        result: dict[str, object] = {"experiment": experiment, "configs": {}}
        for display_name, canonical_name in view.items():
            run_summary = summarize_run(args.output_dir / "runs" / canonical_name, args.folds)
            result["configs"][display_name] = {
                "source_run": canonical_name,
                "parameters": CANONICAL_CONFIGS[canonical_name],
                **run_summary,
            }
        (aggregate_dir / f"{experiment}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        all_aggregates[experiment] = result
    (args.output_dir / "summary.json").write_text(
        json.dumps(all_aggregates, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        fieldnames = ["experiment", "config", "source_run", "completed_folds"]
        fieldnames.extend(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for experiment, result in all_aggregates.items():
            for display_name, values in result["configs"].items():
                row = {
                    "experiment": experiment,
                    "config": display_name,
                    "source_run": values["source_run"],
                    "completed_folds": values["completed_folds"],
                }
                for metric in METRIC_NAMES:
                    for stat in ("mean", "std"):
                        row[f"{metric}_{stat}"] = values.get("metrics", {}).get(metric, {}).get(stat)
                writer.writerow(row)
    return all_aggregates


def write_protocol(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "runner": str(Path(__file__).name),
        "trainer": "07-main_hcp_subjectwise.py",
        "dataroot": str(args.dataroot),
        "experiment": args.experiment,
        "folds": args.folds,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "jobs": args.jobs,
        "seed": args.seed,
        "edge_source": "pcorr",
        "edge_top_percent": 0.10,
        "positive_edges_only": True,
        "best_metric": "balanced_acc",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "step_size": 20,
        "gamma": 0.5,
        "weight_decay": 0.005,
        "selected_canonical_configs": selected_configs(args.experiment),
        "experiment_views": selected_views(args.experiment),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def write_split_manifest(args: argparse.Namespace) -> None:
    subjects = sorted(path.stem for path in (args.dataroot / "subjects").glob("*.h5"))
    if len(subjects) < 3:
        raise ValueError(f"fewer than three subjects under {args.dataroot / 'subjects'}")
    shuffled = subjects[:]
    random.Random(args.seed).shuffle(shuffled)
    subject_folds = np.array_split(np.asarray(shuffled, dtype=object), 5)
    folds = []
    for fold, test_fold in enumerate(subject_folds):
        test_subjects = set(str(subject) for subject in test_fold.tolist())
        remaining = [subject for subject in shuffled if subject not in test_subjects]
        n_val = max(1, min(int(round(len(remaining) * 0.25)), len(remaining) - 1))
        folds.append({
            "fold": fold,
            "train_subjects": sorted(remaining[n_val:]),
            "val_subjects": sorted(remaining[:n_val]),
            "test_subjects": sorted(test_subjects),
        })
    manifest = {
        "dataroot": str(args.dataroot),
        "method": "matches 07-main_hcp_subjectwise.py deterministic random subject split",
        "n_subjects": len(subjects),
        "n_folds": 5,
        "val_ratio": 0.25,
        "seed": args.seed,
        "folds": folds,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    write_protocol(args)
    write_split_manifest(args)
    work = [
        (experiment_config, fold)
        for experiment_config in selected_configs(args.experiment)
        for fold in args.folds
    ]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_fold, args, experiment_config, fold):
                (experiment_config["name"], fold)
            for experiment_config, fold in work
        }
        for future in as_completed(futures):
            name, fold = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(f"failed {name} fold={fold}") from error
    aggregates = write_aggregates(args)
    print(json.dumps(aggregates, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
