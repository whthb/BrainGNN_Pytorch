#!/usr/bin/env python
"""Build deterministic stratified subject-wise splits for local HCP graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from imports.hcp_splits import build_subjectwise_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_records(dataroot: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for subject_file in sorted((dataroot / "subjects").glob("*.h5")):
        with h5py.File(subject_file, "r") as handle:
            for sample_name in sorted(handle["samples"]):
                group = handle["samples"][sample_name]
                records.append(
                    {
                        "subject": subject_file.stem,
                        "sample_name": sample_name,
                        "task_label": int(group["task_label"][()][0]),
                    }
                )
    if not records:
        raise ValueError(f"no subject-wise samples under {dataroot / 'subjects'}")
    return records


def main() -> None:
    args = parse_args()
    records = load_records(args.dataroot)
    result = {
        "dataroot": str(args.dataroot),
        "method": "nested StratifiedGroupKFold with subjects as groups",
        "n_samples": len(records),
        "n_subjects": len({str(record["subject"]) for record in records}),
        "n_folds": args.n_folds,
        "val_folds": args.val_folds,
        "seed": args.seed,
        "folds": build_subjectwise_splits(records, args.n_folds, args.val_folds, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
