#!/usr/bin/env python
"""Run the retained RBF-SVM baseline using the same inputs as BrainGNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BRAINGNN_PARAMETER_COUNT = 55_719


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--edge-source", choices=["corr", "pcorr"], default="pcorr")
    parser.add_argument("--edge-top-percent", type=float, default=0.10)
    return parser.parse_args()


def select_edge_matrix(matrix: np.ndarray, top_percent: float) -> np.ndarray:
    """Match BrainGNN's positive top-percent edge construction."""
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.where(matrix > 0, matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    if top_percent <= 0 or top_percent > 1:
        raise ValueError("--edge-top-percent must be in (0, 1]")
    num_nodes = matrix.shape[0]
    edge_topk = max(1, min(int(np.ceil((num_nodes - 1) * top_percent)), num_nodes - 1))
    topk_cols = np.argpartition(matrix, -edge_topk, axis=1)[:, -edge_topk:]
    rows = np.arange(num_nodes)[:, None]
    mask = np.zeros_like(matrix, dtype=bool)
    mask[rows, topk_cols] = matrix[rows, topk_cols] > 0
    mask = mask | mask.T
    return matrix * mask


def same_input_vector(corr: np.ndarray, edge_matrix: np.ndarray) -> np.ndarray:
    """Flatten BrainGNN node features and weighted graph in fixed ROI order."""
    return np.concatenate([corr.reshape(-1), edge_matrix.reshape(-1)]).astype(np.float32)


def load_data(dataroot: Path, edge_source: str, edge_top_percent: float):
    features, labels, subjects, samples = [], [], [], []
    node_feature_dimension = None
    edge_dimension = None
    for subject_file in sorted((dataroot / "subjects").glob("*.h5")):
        with h5py.File(subject_file, "r") as handle:
            for sample_name in sorted(handle["samples"]):
                group = handle["samples"][sample_name]
                corr = np.nan_to_num(group["corr"][()], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                edge_matrix = select_edge_matrix(group[edge_source][()], edge_top_percent)
                features.append(same_input_vector(corr, edge_matrix))
                labels.append(int(group["task_label"][()][0]))
                subjects.append(subject_file.stem)
                samples.append(sample_name)
                node_feature_dimension = int(corr.size)
                edge_dimension = int(edge_matrix.size)
    if not features:
        raise ValueError(f"no samples under {dataroot / 'subjects'}")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects, dtype=object),
        np.asarray(samples, dtype=object),
        node_feature_dimension,
        edge_dimension,
    )


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
    }


def model_complexity(model) -> dict[str, object]:
    estimator = model.steps[-1][1]
    support_vectors = int(estimator.support_vectors_.shape[0])
    return {
        "measure": "support_vectors_and_stored_scalar_proxy",
        "support_vectors": support_vectors,
        "stored_scalar_proxy": int(
            estimator.support_vectors_.size + estimator.dual_coef_.size + estimator.intercept_.size
        ),
    }


def build_model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", cache_size=4096),
    )


def main() -> None:
    args = parse_args()
    x, y, subjects, samples, node_dim, edge_dim = load_data(
        args.dataroot, args.edge_source, args.edge_top_percent
    )
    manifest = json.loads(args.split_manifest.read_text())
    fold_results: list[dict[str, object]] = []
    for split in manifest["folds"]:
        fold = int(split["fold"])
        train_mask = np.isin(subjects, split["train_subjects"])
        test_mask = np.isin(subjects, split["test_subjects"])
        train_x, train_y = x[train_mask], y[train_mask]
        test_x, test_y = x[test_mask], y[test_mask]
        model = build_model()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        result = {
            "fold": fold,
            "model": "rbf_svm",
            "n_train_graphs": int(train_mask.sum()),
            "n_test_graphs": int(test_mask.sum()),
            "complexity": model_complexity(model),
            **metrics(test_y, pred),
        }
        fold_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "macro_recall", "macro_precision")
    aggregate = {}
    for model_name in sorted({str(row["model"]) for row in fold_results}):
        rows = [row for row in fold_results if row["model"] == model_name]
        aggregate[model_name] = {}
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in rows])
            aggregate[model_name][f"{metric_name}_mean"] = float(values.mean())
            aggregate[model_name][f"{metric_name}_std"] = float(values.std())
        aggregate[model_name]["complexity_mean"] = {
            key: float(np.mean([float(row["complexity"][key]) for row in rows]))
            for key in rows[0]["complexity"]
            if key not in {"measure"} and isinstance(rows[0]["complexity"][key], (int, float))
        }

    result = {
        "dataroot": str(args.dataroot),
        "split_manifest": str(args.split_manifest),
        "input": {
            "description": (
                "BrainGNN Pearson node-feature matrix concatenated with the weighted positive "
                "top-10% partial-correlation adjacency matrix in fixed ROI order"
            ),
            "feature_dimension": int(x.shape[1]),
            "node_feature_dimension": int(node_dim),
            "edge_dimension": int(edge_dim),
            "edge_source": args.edge_source,
            "edge_top_percent": args.edge_top_percent,
            "roi_identity_note": (
                "BrainGNN's constant identity pos matrix is represented implicitly by fixed vector positions"
            ),
        },
        "training_note": (
            "The retained unweighted RBF-SVM baseline is fit on the exact BrainGNN training graphs only; "
            "validation graphs are not added."
        ),
        "reference": {
            "brain_gnn_parameter_count": BRAINGNN_PARAMETER_COUNT,
            "brain_gnn_balanced_accuracy_mean": 0.752528492015674,
            "brain_gnn_balanced_accuracy_std": 0.028057246307531116,
        },
        "folds": fold_results,
        "aggregate": aggregate,
        "sample_count": int(len(samples)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
