#!/usr/bin/env python
"""Summarize BrainGNN feasibility classification and interpretability outputs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-rois", type=int, default=17)
    return parser.parse_args()


def mean_jaccard(rows: np.ndarray, top_rois: int) -> float:
    selections = [set(np.argpartition(row, -top_rois)[-top_rois:].tolist()) for row in rows]
    values = [
        len(first & second) / len(first | second)
        for first, second in itertools.combinations(selections, 2)
    ]
    return float(np.mean(values)) if values else 1.0


def aligned_community_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = first.T
    second = second.T
    denominator = np.linalg.norm(first, axis=1)[:, None] * np.linalg.norm(second, axis=1)[None, :]
    similarity = (first @ second.T) / np.maximum(denominator, 1e-12)
    row, col = linear_sum_assignment(-similarity)
    return float(similarity[row, col].mean())


def main() -> None:
    args = parse_args()
    fold_dirs = sorted(path.parent for path in args.experiment_root.glob("fold_*/summary.json"))
    if not fold_dirs:
        raise ValueError(f"no completed fold summaries under {args.experiment_root}")
    summaries = [json.loads((fold_dir / "summary.json").read_text()) for fold_dir in fold_dirs]
    aggregate_metrics = {}
    for key in summaries[0]["metrics"]:
        values = np.asarray([float(summary["metrics"][key]) for summary in summaries])
        aggregate_metrics[key] = {"mean": float(values.mean()), "std": float(values.std())}

    scores = []
    labels = []
    fold_jaccard: list[dict[str, object]] = []
    score_separation = []
    communities = []
    for fold_dir in fold_dirs:
        values = np.load(fold_dir / "test_pool1_scores.npz")
        fold_scores = values["scores"]
        fold_labels = values["true"]
        scores.append(fold_scores)
        labels.append(fold_labels)
        task_jaccard = {
            str(label): mean_jaccard(fold_scores[fold_labels == label], args.top_rois)
            for label in sorted(set(fold_labels.tolist()))
        }
        fold_jaccard.append({"fold": fold_dir.name, "task_jaccard": task_jaccard})
        sorted_scores = np.sort(fold_scores, axis=1)
        half = fold_scores.shape[1] // 2
        score_separation.append(float(sorted_scores[:, -half:].mean() - sorted_scores[:, :half].mean()))
        community_path = fold_dir / "community_membership_alpha_positive.npy"
        if community_path.exists():
            communities.append(np.load(community_path))

    all_scores = np.concatenate(scores)
    all_labels = np.concatenate(labels)
    roi_importance = {}
    for label in sorted(set(all_labels.tolist())):
        task_scores = all_scores[all_labels == label]
        mean_score = task_scores.mean(axis=0)
        top = np.argsort(mean_score)[-args.top_rois:][::-1]
        selected = np.zeros_like(task_scores, dtype=bool)
        selected_rows = np.arange(task_scores.shape[0])[:, None]
        selected_cols = np.argpartition(task_scores, -args.top_rois, axis=1)[:, -args.top_rois:]
        selected[selected_rows, selected_cols] = True
        roi_importance[str(label)] = {
            "mean_scores": mean_score.tolist(),
            "selection_frequency": selected.mean(axis=0).tolist(),
            "top_roi_indices_zero_based": top.tolist(),
        }
    community_similarities = [
        aligned_community_similarity(first, second)
        for first, second in itertools.combinations(communities, 2)
    ]
    result = {
        "completed_folds": len(fold_dirs),
        "aggregate_metrics": aggregate_metrics,
        "chance_accuracy": 1.0 / len(set(all_labels.tolist())),
        "pool_score_top_bottom_mean_gap": {
            "mean": float(np.mean(score_separation)),
            "std": float(np.std(score_separation)),
        },
        "within_task_selected_roi_jaccard_by_fold": fold_jaccard,
        "roi_importance": roi_importance,
        "community_aligned_cosine_similarity": {
            "mean": float(np.mean(community_similarities)) if community_similarities else None,
            "std": float(np.std(community_similarities)) if community_similarities else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
