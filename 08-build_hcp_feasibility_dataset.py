#!/usr/bin/env python
"""Merge local HCP LR/RL runs into one graph per subject-task pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py

from imports.hcp_feasibility import DEFAULT_TASK_ORDER, audit_subjectwise_dataset, merge_subject_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_label_map(source_root: Path) -> dict[str, int]:
    label_map_path = source_root / "task_label_map.json"
    if label_map_path.exists():
        with label_map_path.open() as handle:
            source_map = {str(key).upper(): int(value) for key, value in json.load(handle).items()}
        tasks = [task for task in DEFAULT_TASK_ORDER if task in source_map]
        tasks.extend(sorted(set(source_map) - set(tasks)))
        return {task: index for index, task in enumerate(tasks)}
    tasks: set[str] = set()
    for subject_file in sorted((source_root / "subjects").glob("*.h5")):
        with h5py.File(subject_file, "r") as handle:
            tasks.update(name.rsplit("_", 1)[0].upper() for name in handle["samples"])
    ordered = [task for task in DEFAULT_TASK_ORDER if task in tasks]
    ordered.extend(sorted(tasks - set(ordered)))
    return {task: index for index, task in enumerate(ordered)}


def main() -> None:
    args = parse_args()
    source_files = sorted((args.source_root / "subjects").glob("*.h5"))
    if not source_files:
        raise ValueError(f"no subject files under {args.source_root / 'subjects'}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_subjects = args.output_root / "subjects"
    output_subjects.mkdir(parents=True, exist_ok=True)
    label_map = infer_label_map(args.source_root)
    rows: list[dict[str, object]] = []
    for index, source_file in enumerate(source_files, start=1):
        output_file = output_subjects / source_file.name
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(f"{output_file} exists; use --overwrite")
        rows.extend(merge_subject_file(source_file, output_file, label_map))
        print(f"[{index}/{len(source_files)}] wrote {output_file}", flush=True)

    with (args.output_root / "task_label_map.json").open("w") as handle:
        json.dump(label_map, handle, indent=2, sort_keys=True)
    with (args.output_root / "sample_metadata.csv").open("w", newline="") as handle:
        fieldnames = ["subject", "task", "task_label", "n_source_runs", "timepoints", "rois", "source_samples"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "source_samples": json.dumps(row["source_samples"])})
    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "merge_method": "z-score each available run, concatenate time series, recompute corr and pcorr",
        "task_label_map": label_map,
        **audit_subjectwise_dataset(args.output_root),
    }
    with (args.output_root / "feasibility_dataset_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
