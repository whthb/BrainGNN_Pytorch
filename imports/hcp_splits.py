"""Deterministic stratified subject-wise splits for HCP task graphs."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(labels[indices].tolist()).items())}


def _subjects(groups: np.ndarray, indices: np.ndarray) -> list[str]:
    return sorted(set(str(value) for value in groups[indices].tolist()))


def build_subjectwise_splits(
    records: list[dict[str, object]],
    n_folds: int = 5,
    val_folds: int = 4,
    seed: int = 123,
) -> list[dict[str, object]]:
    labels = np.asarray([int(record["task_label"]) for record in records], dtype=np.int64)
    groups = np.asarray([str(record["subject"]) for record in records], dtype=object)
    dummy = np.zeros(len(records), dtype=np.int8)
    all_classes = set(labels.tolist())
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, object]] = []

    for fold, (train_val_index, test_index) in enumerate(outer.split(dummy, labels, groups)):
        inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed + fold + 1)
        inner_train, inner_val = next(
            inner.split(dummy[train_val_index], labels[train_val_index], groups[train_val_index])
        )
        train_index = train_val_index[inner_train]
        val_index = train_val_index[inner_val]
        split_indices = {"train": train_index, "val": val_index, "test": test_index}
        split_subjects = {name: _subjects(groups, index) for name, index in split_indices.items()}
        subject_sets = {name: set(values) for name, values in split_subjects.items()}
        if subject_sets["train"] & subject_sets["val"] or subject_sets["train"] & subject_sets["test"] \
                or subject_sets["val"] & subject_sets["test"]:
            raise ValueError(f"subject leakage in fold {fold}")
        for name, index in split_indices.items():
            if set(labels[index].tolist()) != all_classes:
                raise ValueError(f"fold {fold} {name} does not contain every class")
        folds.append(
            {
                "fold": fold,
                "train_subjects": split_subjects["train"],
                "val_subjects": split_subjects["val"],
                "test_subjects": split_subjects["test"],
                "n_train_samples": int(len(train_index)),
                "n_val_samples": int(len(val_index)),
                "n_test_samples": int(len(test_index)),
                "train_class_counts": _class_counts(labels, train_index),
                "val_class_counts": _class_counts(labels, val_index),
                "test_class_counts": _class_counts(labels, test_index),
            }
        )
    return folds
