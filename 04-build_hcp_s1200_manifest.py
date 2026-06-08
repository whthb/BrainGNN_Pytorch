#!/usr/bin/env python
"""Build an HCP task-fMRI manifest from the open-access S3 bucket.

The manifest columns are:
    subject,task,run,source

When motion QC thresholds are provided, each task/run is checked against the
HCP `Movement_RelativeRMS.txt` file in the same Results directory. This is the
closest available S1200 open-access file for frame-to-frame displacement style
QC. Rows are kept only when mean(values) < threshold and max(values) < threshold.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


DEFAULT_TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")


def run_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout


def list_subjects(aws_profile: str, bucket_prefix: str) -> list[str]:
    stdout = run_capture(["aws", "--profile", aws_profile, "s3", "ls", bucket_prefix])
    subjects: list[str] = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] == "PRE" and parts[1].endswith("/"):
            subject = parts[1].strip("/")
            if subject.isdigit():
                subjects.append(subject)
    return sorted(subjects)


def s3_motion_source(dtseries_source: str, motion_file: str) -> str:
    return str(Path(dtseries_source).parent / motion_file).replace("s3:/", "s3://")


def load_motion_values(source: str, aws_profile: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="hcp_motion_qc_") as tmpdir:
        dst = Path(tmpdir) / Path(source).name
        subprocess.run(
            ["aws", "--profile", aws_profile, "s3", "cp", source, str(dst), "--only-show-errors"],
            check=True,
        )
        values = np.loadtxt(dst, dtype=np.float64)
    return np.atleast_1d(values).reshape(-1)


def qc_pass(values: np.ndarray, mean_threshold: float | None, max_threshold: float | None) -> tuple[bool, float, float]:
    mean_value = float(np.mean(values))
    max_value = float(np.max(values))
    passed = True
    if mean_threshold is not None:
        passed = passed and mean_value < mean_threshold
    if max_threshold is not None:
        passed = passed and max_value < max_threshold
    return passed, mean_value, max_value


def build_candidate_rows(subjects: list[str], tasks: list[str], runs: list[str],
                         bucket_prefix: str, dtseries_suffix: str,
                         motion_file: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for task in tasks:
            for run in runs:
                task_run = f"tfMRI_{task}_{run}"
                source = (
                    f"{bucket_prefix.rstrip('/')}/{subject}/MNINonLinear/Results/"
                    f"{task_run}/{task_run}{dtseries_suffix}"
                )
                rows.append(
                    {
                        "subject": subject,
                        "task": task,
                        "run": run,
                        "source": source,
                        "motion_source": s3_motion_source(source, motion_file),
                        "qc_status": "not_applied",
                        "mean_fd": "",
                        "max_fd": "",
                        "keep": True,
                        "error": "",
                    }
                )
    return rows


def apply_motion_qc(row: dict[str, object], aws_profile: str,
                    mean_threshold: float | None, max_threshold: float | None,
                    exclude_missing_qc: bool) -> dict[str, object]:
    row = dict(row)
    try:
        values = load_motion_values(str(row["motion_source"]), aws_profile)
        keep, mean_fd, max_fd = qc_pass(values, mean_threshold, max_threshold)
        row.update({"qc_status": "ok", "mean_fd": mean_fd, "max_fd": max_fd, "keep": keep})
    except Exception as exc:
        row.update({"qc_status": "failed", "keep": not exclude_missing_qc, "error": str(exc)})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aws-profile", default="hcp")
    parser.add_argument("--release", choices=["HCP_1200", "HCP_900"], default="HCP_1200")
    parser.add_argument("--bucket-prefix", default=None,
                        help="override S3 release prefix; default is derived from --release")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--runs", nargs="+", default=["LR"])
    parser.add_argument("--dtseries-suffix", default="_Atlas_MSMAll.dtseries.nii")
    parser.add_argument("--motion-file", default="Movement_RelativeRMS.txt")
    parser.add_argument("--mean-fd-threshold", type=float, default=None)
    parser.add_argument("--max-fd-threshold", type=float, default=None)
    parser.add_argument("--qc-report", type=Path, default=None)
    parser.add_argument("--exclude-missing-qc", action="store_true")
    parser.add_argument("--qc-workers", type=int, default=8,
                        help="parallel workers for downloading motion QC files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bucket_prefix = args.bucket_prefix or f"s3://hcp-openaccess/{args.release}/"
    tasks = [task.upper() for task in args.tasks]
    runs = [run.upper() for run in args.runs]

    if args.subjects:
        subjects = [str(subject) for subject in args.subjects]
    else:
        subjects = list_subjects(args.aws_profile, bucket_prefix)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    apply_qc = args.mean_fd_threshold is not None or args.max_fd_threshold is not None
    qc_rows = build_candidate_rows(subjects, tasks, runs, bucket_prefix,
                                   args.dtseries_suffix, args.motion_file)
    if apply_qc:
        workers = max(1, int(args.qc_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            qc_rows = list(
                executor.map(
                    lambda row: apply_motion_qc(
                        row,
                        args.aws_profile,
                        args.mean_fd_threshold,
                        args.max_fd_threshold,
                        args.exclude_missing_qc,
                    ),
                    qc_rows,
                )
            )

    manifest_rows = [
        {
            "subject": str(row["subject"]),
            "task": str(row["task"]),
            "run": str(row["run"]),
            "source": str(row["source"]),
        }
        for row in qc_rows
        if bool(row["keep"])
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "task", "run", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    if args.qc_report is not None:
        args.qc_report.parent.mkdir(parents=True, exist_ok=True)
        with args.qc_report.open("w", newline="") as handle:
            fieldnames = [
                "subject",
                "task",
                "run",
                "source",
                "motion_source",
                "qc_status",
                "mean_fd",
                "max_fd",
                "keep",
                "error",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qc_rows)

    print(
        f"wrote {args.output} subjects={len(subjects)} tasks={len(tasks)} "
        f"runs={len(runs)} rows={len(manifest_rows)} qc_applied={apply_qc}"
    )


if __name__ == "__main__":
    main()
