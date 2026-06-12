"""Utilities for the local HCP BrainGNN feasibility dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


DEFAULT_TASK_ORDER = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")


def standardize_run(timeseries: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each ROI within one acquisition run."""
    timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mean = timeseries.mean(axis=0, keepdims=True)
    std = timeseries.std(axis=0, keepdims=True)
    return np.nan_to_num((timeseries - mean) / np.maximum(std, eps), nan=0.0).astype(np.float32)


def connectivity(timeseries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pearson and partial-correlation connectivity matrices."""
    corr = np.corrcoef(timeseries, rowvar=False).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    cov = np.cov(timeseries, rowvar=False).astype(np.float64)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
    precision = np.linalg.pinv(cov)
    denominator = np.sqrt(np.maximum(np.outer(np.diag(precision), np.diag(precision)), 1e-12))
    pcorr = np.nan_to_num(-precision / denominator, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(pcorr, 1.0)
    return corr, pcorr


def sample_task(sample_name: str, group: h5py.Group) -> str:
    task = group.attrs.get("task")
    if task is not None:
        return str(task).upper()
    return sample_name.rsplit("_", 1)[0].upper()


def merge_subject_file(source_file: Path, output_file: Path, task_label_map: dict[str, int]) -> list[dict[str, object]]:
    """Merge available LR/RL runs into one graph for each subject-task pair."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with h5py.File(source_file, "r") as source, h5py.File(output_file, "w") as output:
        subject = str(source.attrs.get("subject", source_file.stem))
        output.attrs["subject"] = subject
        output.attrs["task_label_map_json"] = json.dumps(task_label_map, sort_keys=True)
        samples_out = output.create_group("samples")
        by_task: dict[str, list[tuple[str, h5py.Group]]] = {}
        for sample_name in sorted(source["samples"].keys()):
            group = source["samples"][sample_name]
            by_task.setdefault(sample_task(sample_name, group), []).append((sample_name, group))

        for task in sorted(by_task, key=lambda value: task_label_map[value]):
            source_groups = by_task[task]
            run_series = [standardize_run(group["timeseries"][()]) for _, group in source_groups]
            roi_counts = {series.shape[1] for series in run_series}
            if len(roi_counts) != 1:
                raise ValueError(f"{source_file}: inconsistent ROI counts for {task}: {sorted(roi_counts)}")
            merged_series = np.concatenate(run_series, axis=0)
            corr, pcorr = connectivity(merged_series)
            group_out = samples_out.create_group(task)
            label = int(task_label_map[task])
            source_names = [name for name, _ in source_groups]
            group_out.attrs["task"] = task
            group_out.attrs["task_label"] = label
            group_out.attrs["source_samples_json"] = json.dumps(source_names)
            group_out.create_dataset("timeseries", data=merged_series, compression="gzip", compression_opts=4)
            group_out.create_dataset("corr", data=corr, compression="gzip", compression_opts=4)
            group_out.create_dataset("pcorr", data=pcorr, compression="gzip", compression_opts=4)
            group_out.create_dataset("task_label", data=np.array([label], dtype=np.int64))
            rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "task_label": label,
                    "source_samples": source_names,
                    "n_source_runs": len(source_names),
                    "timepoints": int(merged_series.shape[0]),
                    "rois": int(merged_series.shape[1]),
                }
            )
    return rows


def audit_subjectwise_dataset(root: Path) -> dict[str, object]:
    subjects = sorted((root / "subjects").glob("*.h5"))
    task_counts: Counter[str] = Counter()
    roi_counts: Counter[int] = Counter()
    source_run_counts: Counter[int] = Counter()
    finite = True
    n_samples = 0
    for subject_file in subjects:
        with h5py.File(subject_file, "r") as handle:
            for sample_name in handle["samples"]:
                group = handle["samples"][sample_name]
                task = sample_task(sample_name, group)
                corr = group["corr"][()]
                pcorr = group["pcorr"][()]
                n_samples += 1
                task_counts[task] += 1
                roi_counts[int(corr.shape[0])] += 1
                finite = finite and bool(np.isfinite(corr).all() and np.isfinite(pcorr).all())
                sources = json.loads(group.attrs.get("source_samples_json", "[]"))
                source_run_counts[len(sources)] += 1
    return {
        "n_subjects": len(subjects),
        "n_samples": n_samples,
        "task_counts": dict(sorted(task_counts.items())),
        "roi_counts": {str(key): value for key, value in sorted(roi_counts.items())},
        "source_run_counts": {str(key): value for key, value in sorted(source_run_counts.items())},
        "all_connectivity_finite": finite,
    }
