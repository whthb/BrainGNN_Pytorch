import json
from pathlib import Path

import h5py
import numpy as np

from imports.hcp_feasibility import audit_subjectwise_dataset, merge_subject_file, standardize_run
from imports.hcp_splits import build_subjectwise_splits


def _write_source_subject(path, subject, tasks):
    with h5py.File(path, "w") as handle:
        handle.attrs["subject"] = subject
        samples = handle.create_group("samples")
        for task, runs in tasks.items():
            for run, timeseries in runs.items():
                group = samples.create_group(f"{task}_{run}")
                group.attrs["task"] = task
                group.create_dataset("timeseries", data=timeseries)


def test_standardize_run():
    timeseries = np.asarray([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]], dtype=np.float32)
    standardized = standardize_run(timeseries)
    np.testing.assert_allclose(standardized[:, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized[:, 0].std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(standardized[:, 1], 0.0, atol=1e-6)


def test_merge_subject_file(tmp_path):
    source = tmp_path / "source.h5"
    output_root = tmp_path / "output"
    output = output_root / "subjects" / "100001.h5"
    rng = np.random.default_rng(123)
    _write_source_subject(
        source,
        "100001",
        {
            "EMOTION": {"LR": rng.normal(size=(20, 4)), "RL": rng.normal(size=(25, 4))},
            "MOTOR": {"LR": rng.normal(size=(30, 4))},
        },
    )
    rows = merge_subject_file(source, output, {"EMOTION": 0, "MOTOR": 1})
    assert len(rows) == 2
    with h5py.File(output, "r") as handle:
        assert set(handle["samples"]) == {"EMOTION", "MOTOR"}
        assert handle["samples"]["EMOTION"]["timeseries"].shape == (45, 4)
        assert handle["samples"]["MOTOR"]["timeseries"].shape == (30, 4)
        assert json.loads(handle["samples"]["EMOTION"].attrs["source_samples_json"]) == [
            "EMOTION_LR",
            "EMOTION_RL",
        ]
    audit = audit_subjectwise_dataset(output_root)
    assert audit["n_samples"] == 2
    assert audit["roi_counts"] == {"4": 2}
    assert audit["all_connectivity_finite"]


def test_subjectwise_splits_have_no_leakage_and_all_classes():
    records = [
        {"subject": f"{subject:03d}", "task_label": task}
        for subject in range(30)
        for task in range(3)
        if (subject + task) % 5 != 0
    ]
    folds = build_subjectwise_splits(records, n_folds=5, val_folds=4, seed=123)
    assert len(folds) == 5
    for fold in folds:
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])
        test = set(fold["test_subjects"])
        assert not train & val
        assert not train & test
        assert not val & test
        assert set(fold["train_class_counts"]) == {"0", "1", "2"}
        assert set(fold["val_class_counts"]) == {"0", "1", "2"}
        assert set(fold["test_class_counts"]) == {"0", "1", "2"}
