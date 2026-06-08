#!/usr/bin/env python
"""Stream HCP task CIFTI files into subject-wise ROI HDF5 files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import h5py
import nibabel as nib
import numpy as np


DEFAULT_TASK_ORDER = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--aws-profile", default="hcp")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hcp_task_stream"))
    parser.add_argument("--task-order", nargs="+", default=list(DEFAULT_TASK_ORDER))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of subjects to process in parallel; each subject is written by one worker")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject", "task", "run", "source"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def fetch_source(source: str, dst_dir: Path, aws_profile: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(source)
    name = Path(parsed.path).name or "source.dtseries.nii"
    partial = dst_dir / f"{name}.partial"
    final = dst_dir / name

    if source.startswith("s3://"):
        subprocess.run(
            ["aws", "--profile", aws_profile, "s3", "cp", source, str(partial), "--only-show-errors"],
            check=True,
        )
        partial.rename(final)
        return final

    if source.startswith("http://") or source.startswith("https://"):
        import requests

        with requests.get(source, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.rename(final)
        return final

    local = Path(source)
    if not local.exists():
        raise FileNotFoundError(source)
    return local


def axis_lookup(axis) -> dict[tuple, int]:
    lookup: dict[tuple, int] = {}
    for structure, slc, model in axis.iter_structures():
        vertices = getattr(model, "vertex", None)
        voxels = getattr(model, "voxel", None)
        indices = range(len(axis))[slc]
        for offset, index in enumerate(indices):
            if vertices is not None and len(vertices) > offset and int(vertices[offset]) >= 0:
                lookup[(structure, "vertex", int(vertices[offset]))] = index
            elif voxels is not None and len(voxels) > offset:
                lookup[(structure, "voxel", tuple(int(v) for v in voxels[offset]))] = index
    return lookup


def load_atlas_labels(atlas_path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    atlas_img = nib.load(str(atlas_path))
    labels = np.asarray(atlas_img.get_fdata()).squeeze().astype(np.int32)
    if labels.ndim != 1:
        raise ValueError(f"atlas labels must be 1D after squeeze, got shape {labels.shape}")
    brain_axis = atlas_img.header.get_axis(1)
    roi_ids = np.array(sorted(int(x) for x in np.unique(labels) if int(x) != 0), dtype=np.int32)
    if len(roi_ids) == 0:
        raise ValueError("atlas does not contain non-zero ROI labels")
    return labels, roi_ids, brain_axis


def align_labels(atlas_labels: np.ndarray, atlas_axis, data_axis) -> np.ndarray:
    if len(atlas_labels) == len(data_axis):
        return atlas_labels
    atlas_positions = axis_lookup(atlas_axis)
    data_positions = axis_lookup(data_axis)
    aligned = np.zeros(len(data_axis), dtype=np.int32)
    for key, atlas_index in atlas_positions.items():
        data_index = data_positions.get(key)
        if data_index is not None:
            aligned[data_index] = atlas_labels[atlas_index]
    return aligned


def roi_timeseries(dtseries_path: Path, atlas_labels: np.ndarray, roi_ids: np.ndarray, atlas_axis) -> np.ndarray:
    img = nib.load(str(dtseries_path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"dtseries must be 2D, got {data.shape}")
    labels = align_labels(atlas_labels, atlas_axis, img.header.get_axis(1))
    series = np.zeros((data.shape[0], len(roi_ids)), dtype=np.float32)
    for out_index, roi_id in enumerate(roi_ids):
        mask = labels == roi_id
        if mask.any():
            series[:, out_index] = data[:, mask].mean(axis=1)
    return np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)


def connectivity(timeseries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corr = np.corrcoef(timeseries, rowvar=False).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    cov = np.cov(timeseries, rowvar=False).astype(np.float64)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
    precision = np.linalg.pinv(cov)
    denom = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    pcorr = -precision / denom
    pcorr = np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(pcorr, 1.0)
    return corr, pcorr


def existing_sample(subject_file: Path, task: str, run: str) -> bool:
    if not subject_file.exists():
        return False
    with h5py.File(subject_file, "r") as handle:
        return f"samples/{task}_{run}" in handle


def write_subject_sample(subject_file: Path, subject: str, task: str, run: str, task_label: int,
                         task_label_map: dict[str, int], timeseries: np.ndarray, corr: np.ndarray,
                         pcorr: np.ndarray, source: str) -> None:
    subject_file.parent.mkdir(parents=True, exist_ok=True)
    sample_id = f"{task}_{run}"
    with h5py.File(subject_file, "a") as handle:
        handle.attrs["subject"] = subject
        handle.attrs["task_label_map_json"] = json.dumps(task_label_map, sort_keys=True)
        samples = handle.require_group("samples")
        if sample_id in samples:
            del samples[sample_id]
        group = samples.create_group(sample_id)
        group.attrs["task"] = task
        group.attrs["run"] = run
        group.attrs["task_label"] = int(task_label)
        group.attrs["source"] = source
        group.create_dataset("timeseries", data=timeseries, compression="gzip", compression_opts=4)
        group.create_dataset("corr", data=corr, compression="gzip", compression_opts=4)
        group.create_dataset("pcorr", data=pcorr, compression="gzip", compression_opts=4)
        group.create_dataset("task_label", data=np.array([task_label], dtype=np.int64))


def process_subject_rows(subject: str, rows: list[dict[str, object]], atlas: str,
                         output_root: str, work_dir: str, task_label_map: dict[str, int],
                         aws_profile: str, skip_existing: bool, keep_work: bool) -> dict[str, object]:
    atlas_labels, roi_ids, atlas_axis = load_atlas_labels(Path(atlas))
    subjects_dir = Path(output_root) / "subjects"
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    counts = Counter()

    for row in rows:
        index = int(row["__index"])
        total = int(row["__total"])
        task = str(row["task"]).upper()
        run = str(row["run"]).upper()
        source = str(row["source"])
        task_label = int(task_label_map[task])
        subject_file = subjects_dir / f"{subject}.h5"
        sample_id = f"{subject}_{task}_{run}"
        base_meta = {
            "row_index": index,
            "sample_id": sample_id,
            "subject": subject,
            "task": task,
            "run": run,
            "task_label": task_label,
            "source": source,
            "subject_file": str(subject_file),
        }

        if skip_existing and existing_sample(subject_file, task, run):
            metadata_rows.append({**base_meta, "status": "skipped_existing", "timepoints": "", "rois": ""})
            counts[task] += 1
            print(f"[{index}/{total}] skipped existing {sample_id}", flush=True)
            continue

        source_dir = Path(tempfile.mkdtemp(prefix=f"source_{index:04d}_{subject}_", dir=work_dir))
        try:
            dtseries = fetch_source(source, source_dir, aws_profile)
            timeseries = roi_timeseries(dtseries, atlas_labels, roi_ids, atlas_axis)
            corr, pcorr = connectivity(timeseries)
            write_subject_sample(subject_file, subject, task, run, task_label, task_label_map,
                                 timeseries, corr, pcorr, source)
            metadata_rows.append(
                {**base_meta, "status": "ok", "timepoints": int(timeseries.shape[0]), "rois": int(timeseries.shape[1])}
            )
            counts[task] += 1
            print(
                f"[{index}/{total}] wrote {subject_file} sample={task}_{run} "
                f"timepoints={timeseries.shape[0]} rois={timeseries.shape[1]}",
                flush=True,
            )
        except Exception as exc:
            failure = {**base_meta, "status": "failed", "error": str(exc)}
            metadata_rows.append({**failure, "timepoints": "", "rois": ""})
            failures.append(failure)
            print(f"[{index}/{total}] failed {sample_id}: {exc}", flush=True)
        finally:
            if not keep_work:
                shutil.rmtree(source_dir, ignore_errors=True)

    return {
        "subject": subject,
        "metadata_rows": metadata_rows,
        "failures": failures,
        "counts": dict(counts),
    }


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    indexed_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        row = dict(row)
        row["__index"] = index
        row["__total"] = len(rows)
        indexed_rows.append(row)

    task_order = [task.upper() for task in args.task_order]
    manifest_tasks = sorted({row["task"].upper() for row in rows})
    ordered_tasks = [task for task in task_order if task in manifest_tasks]
    ordered_tasks += [task for task in manifest_tasks if task not in ordered_tasks]
    task_label_map = {task: index for index, task in enumerate(ordered_tasks)}

    subjects_dir = args.output_root / "subjects"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    counts = Counter()

    subject_rows: dict[str, list[dict[str, object]]] = {}
    for row in indexed_rows:
        subject_rows.setdefault(str(row["subject"]), []).append(row)

    workers = max(1, int(args.workers))
    if workers == 1:
        results = [
            process_subject_rows(subject, subject_rows[subject], str(args.atlas), str(args.output_root),
                                 str(args.work_dir), task_label_map, args.aws_profile,
                                 args.skip_existing, args.keep_work)
            for subject in subject_rows
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_subject = {
                executor.submit(
                    process_subject_rows,
                    subject,
                    subject_rows[subject],
                    str(args.atlas),
                    str(args.output_root),
                    str(args.work_dir),
                    task_label_map,
                    args.aws_profile,
                    args.skip_existing,
                    args.keep_work,
                ): subject
                for subject in subject_rows
            }
            for future in as_completed(future_to_subject):
                subject = future_to_subject[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    subject_failures = []
                    subject_metadata = []
                    for row in subject_rows[subject]:
                        task = str(row["task"]).upper()
                        run = str(row["run"]).upper()
                        task_label = int(task_label_map[task])
                        failure = {
                            "row_index": int(row["__index"]),
                            "sample_id": f"{subject}_{task}_{run}",
                            "subject": subject,
                            "task": task,
                            "run": run,
                            "task_label": task_label,
                            "source": str(row["source"]),
                            "subject_file": str(subjects_dir / f"{subject}.h5"),
                            "status": "failed",
                            "timepoints": "",
                            "rois": "",
                            "error": f"subject worker failed: {exc}",
                        }
                        subject_failures.append(failure)
                        subject_metadata.append(failure)
                    results.append(
                        {
                            "subject": subject,
                            "metadata_rows": subject_metadata,
                            "failures": subject_failures,
                            "counts": {},
                        }
                    )

    for result in results:
        metadata_rows.extend(result["metadata_rows"])
        failures.extend(result["failures"])
        counts.update(result["counts"])
    metadata_rows.sort(key=lambda item: int(item.get("row_index", 0)))
    failures.sort(key=lambda item: int(item.get("row_index", 0)))

    metadata_path = args.output_root / "sample_metadata.csv"
    with metadata_path.open("w", newline="") as handle:
        fieldnames = [
            "sample_id",
            "subject",
            "task",
            "run",
            "task_label",
            "status",
            "timepoints",
            "rois",
            "source",
            "subject_file",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (args.output_root / "task_label_map.json").open("w") as handle:
        json.dump(task_label_map, handle, indent=2, sort_keys=True)
    with (args.output_root / "subject_IDs.txt").open("w") as handle:
        for path in sorted(subjects_dir.glob("*.h5")):
            handle.write(f"{path.stem}\n")

    summary = {
        "manifest": str(args.manifest),
        "atlas": str(args.atlas),
        "n_manifest_samples": len(rows),
        "n_subject_files": len(list(subjects_dir.glob("*.h5"))),
        "n_success_or_skipped": int(sum(counts.values())),
        "n_failures": len(failures),
        "task_counts": dict(sorted(counts.items())),
        "task_label_map": task_label_map,
        "failures": failures,
    }
    with (args.output_root / "hcp_subjectwise_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
