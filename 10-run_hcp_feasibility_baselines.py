#!/usr/bin/env python
"""Run the majority-class baseline on the merged local HCP feasibility dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
    }


def load_data(dataroot: Path) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    subjects: list[str] = []
    for subject_file in sorted((dataroot / "subjects").glob("*.h5")):
        with h5py.File(subject_file, "r") as handle:
            for sample_name in sorted(handle["samples"]):
                group = handle["samples"][sample_name]
                labels.append(int(group["task_label"][()][0]))
                subjects.append(subject_file.stem)
    return np.asarray(labels, dtype=np.int64), np.asarray(subjects, dtype=object)


def main() -> None:
    args = parse_args()
    y, subjects = load_data(args.dataroot)
    with args.split_manifest.open() as handle:
        split_manifest = json.load(handle)
    all_results: list[dict[str, object]] = []
    for split in split_manifest["folds"]:
        fold = int(split["fold"])
        train_mask = np.isin(subjects, split["train_subjects"])
        test_mask = np.isin(subjects, split["test_subjects"])
        train_y = y[train_mask]
        test_y = y[test_mask]
        majority = int(np.bincount(train_y).argmax())
        pred = np.full(test_y.shape, majority, dtype=np.int64)
        result = {"fold": fold, "model": "majority", **metrics(test_y, pred)}
        all_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    aggregate: dict[str, dict[str, float]] = {}
    for model_name in sorted({str(row["model"]) for row in all_results}):
        rows = [row for row in all_results if row["model"] == model_name]
        aggregate[model_name] = {}
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_recall", "macro_precision"):
            values = np.asarray([float(row[key]) for row in rows])
            aggregate[model_name][f"{key}_mean"] = float(values.mean())
            aggregate[model_name][f"{key}_std"] = float(values.std())
    result = {
        "dataroot": str(args.dataroot),
        "split_manifest": str(args.split_manifest),
        "feature": "none; predicts the majority class from each training fold",
        "folds": all_results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
