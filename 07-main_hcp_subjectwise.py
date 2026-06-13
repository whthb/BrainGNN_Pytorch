#!/usr/bin/env python
"""BrainGNN training entry for HCP subject-wise task classification."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

from net.braingnn import Network


EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--edge_source", choices=["corr", "pcorr"], default="pcorr")
    parser.add_argument("--edge_topk", type=int, default=None)
    parser.add_argument("--edge_top_percent", type=float, default=0.10)
    parser.add_argument("--positive_edges_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.25,
                        help="validation fraction within non-test subjects; 0.25 gives about 60/20/20 with 5 folds")
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--stepsize", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--weightdecay", type=float, default=5e-3)
    parser.add_argument("--lamb0", type=float, default=1)
    parser.add_argument("--lamb1", type=float, default=1)
    parser.add_argument("--lamb2", type=float, default=1)
    parser.add_argument("--lamb3", type=float, default=0.1)
    parser.add_argument("--lamb4", type=float, default=0.1)
    parser.add_argument("--lamb5", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--dim1", type=int, default=32)
    parser.add_argument("--dim2", type=int, default=32)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--conv_type", choices=["ra", "vanilla"], default="ra")
    parser.add_argument("--best_metric", choices=["loss", "acc", "balanced_acc"], default="balanced_acc")
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    parser.add_argument(
        "--direction_protocol",
        choices=["mixed", "lr_to_rl", "rl_to_lr"],
        default="mixed",
        help="mixed uses both directions; cross-direction protocols train/validate on the source and test on the target",
    )
    parser.add_argument(
        "--direction_adv_weight",
        type=float,
        default=0.0,
        help="weight of the LR/RL adversarial direction-classification loss",
    )
    parser.add_argument(
        "--pair_consistency_weight",
        type=float,
        default=0.0,
        help="weight of the JS consistency loss between matched LR/RL task predictions",
    )
    parser.add_argument(
        "--pair_balance",
        choices=["none", "sqrt_inverse"],
        default="none",
        help="optional task-frequency balancing for LR/RL pair losses",
    )
    parser.add_argument("--pair_warmup_start", type=int, default=0)
    parser.add_argument("--pair_warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["primary", "pair_noninferiority"],
        default="primary",
    )
    parser.add_argument("--checkpoint_bacc_tolerance", type=float, default=0.005)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_epochs", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--optim", choices=["Adam", "SGD"], default="Adam")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_path", type=Path, default=Path("./model_hcp_subjectwise"))
    parser.add_argument("--log_path", type=Path, default=Path("./log_hcp_subjectwise"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def select_edge_matrix(matrix: np.ndarray, edge_topk: int | None, edge_top_percent: float | None,
                       positive_edges_only: bool) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.where(matrix > 0, matrix, 0.0) if positive_edges_only else np.abs(matrix)
    np.fill_diagonal(matrix, 0.0)
    if edge_topk is None and edge_top_percent is None:
        return matrix
    if edge_topk is not None and edge_top_percent is not None:
        raise ValueError("Use only one of --edge_topk and --edge_top_percent")
    num_nodes = matrix.shape[0]
    if edge_top_percent is not None:
        if edge_top_percent <= 0 or edge_top_percent > 1:
            raise ValueError("--edge_top_percent must be in (0, 1]")
        edge_topk = int(np.ceil((num_nodes - 1) * edge_top_percent))
    edge_topk = max(1, min(int(edge_topk), num_nodes - 1))
    topk_cols = np.argpartition(matrix, -edge_topk, axis=1)[:, -edge_topk:]
    rows = np.arange(num_nodes)[:, None]
    mask = np.zeros_like(matrix, dtype=bool)
    mask[rows, topk_cols] = matrix[rows, topk_cols] > 0
    mask = mask | mask.T
    return matrix * mask


def matrix_to_edges(matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    row, col = np.nonzero(matrix)
    edge_index = torch.from_numpy(np.stack([row, col])).long()
    edge_attr = torch.from_numpy(matrix[row, col]).float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, matrix.shape[0], matrix.shape[1])
    return edge_index.long(), edge_attr.float().view(-1, 1)


def load_hcp_subjectwise_graphs(args: argparse.Namespace) -> tuple[list[Data], dict[str, int]]:
    with (args.dataroot / "task_label_map.json").open() as handle:
        task_label_map = {str(k): int(v) for k, v in json.load(handle).items()}
    graphs: list[Data] = []
    for subject_file in sorted((args.dataroot / "subjects").glob("*.h5")):
        with h5py.File(subject_file, "r") as handle:
            for sample_name in sorted(handle["samples"].keys()):
                group = handle["samples"][sample_name]
                x = np.nan_to_num(group["corr"][()], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                edge_matrix = select_edge_matrix(
                    group[args.edge_source][()], args.edge_topk, args.edge_top_percent, args.positive_edges_only
                )
                edge_index, edge_attr = matrix_to_edges(edge_matrix)
                num_nodes = x.shape[0]
                data = Data(
                    x=torch.from_numpy(x).float(),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(int(group["task_label"][()][0]), dtype=torch.long),
                    pos=torch.eye(num_nodes, dtype=torch.float32),
                )
                data.subject = subject_file.stem
                data.sample_name = sample_name
                data.direction = torch.tensor(
                    0 if sample_name.endswith("_LR") else 1, dtype=torch.long
                )
                graphs.append(data)
    if not graphs:
        raise ValueError(f"no HCP subject-wise files found under {args.dataroot / 'subjects'}")
    return graphs, task_label_map


def split_subjects(graphs: list[Data], n_folds: int, fold: int, val_ratio: float,
                   seed: int) -> tuple[list[Data], list[Data], list[Data], dict[str, list[str]]]:
    subjects = sorted({graph.subject for graph in graphs})
    if len(subjects) < 3:
        raise ValueError("subject-wise train/val/test split needs at least 3 subjects")
    if fold < 0 or fold >= n_folds:
        raise ValueError("--fold must be in [0, n_folds)")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("--val_ratio must be in (0, 1)")
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    subject_folds = np.array_split(np.array(shuffled, dtype=object), n_folds)
    test_subjects = set(str(subject) for subject in subject_folds[fold].tolist())
    remaining = [subject for subject in shuffled if subject not in test_subjects]
    n_val = max(1, min(int(round(len(remaining) * val_ratio)), len(remaining) - 1))
    val_subjects = set(remaining[:n_val])
    train_subjects = set(remaining[n_val:])
    split_info = {
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
    }
    return (
        [g for g in graphs if g.subject in train_subjects],
        [g for g in graphs if g.subject in val_subjects],
        [g for g in graphs if g.subject in test_subjects],
        split_info,
    )


def split_subjects_from_manifest(
    graphs: list[Data], manifest_path: Path, fold: int
) -> tuple[list[Data], list[Data], list[Data], dict[str, list[str]]]:
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    matching = [item for item in manifest["folds"] if int(item["fold"]) == fold]
    if len(matching) != 1:
        raise ValueError(f"expected one fold {fold} in {manifest_path}, found {len(matching)}")
    selected = matching[0]
    split_info = {
        name: sorted(str(subject) for subject in selected[f"{name}_subjects"])
        for name in ("train", "val", "test")
    }
    split_sets = {name: set(subjects) for name, subjects in split_info.items()}
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] \
            or split_sets["val"] & split_sets["test"]:
        raise ValueError(f"subject leakage in fold {fold} from {manifest_path}")
    return (
        [graph for graph in graphs if graph.subject in split_sets["train"]],
        [graph for graph in graphs if graph.subject in split_sets["val"]],
        [graph for graph in graphs if graph.subject in split_sets["test"]],
        {f"{name}_subjects": subjects for name, subjects in split_info.items()},
    )


def label_counts(graphs: list[Data]) -> dict[int, int]:
    return dict(sorted(Counter(int(graph.y.item()) for graph in graphs).items()))


def direction_counts(graphs: list[Data]) -> dict[str, int]:
    labels = {0: "LR", 1: "RL"}
    return {
        labels[key]: value
        for key, value in sorted(Counter(int(graph.direction.item()) for graph in graphs).items())
    }


def apply_direction_protocol(
    train_graphs: list[Data],
    val_graphs: list[Data],
    test_graphs: list[Data],
    protocol: str,
) -> tuple[list[Data], list[Data], list[Data]]:
    if protocol == "mixed":
        return train_graphs, val_graphs, test_graphs
    source, target = (0, 1) if protocol == "lr_to_rl" else (1, 0)
    return (
        [graph for graph in train_graphs if int(graph.direction.item()) == source],
        [graph for graph in val_graphs if int(graph.direction.item()) == source],
        [graph for graph in test_graphs if int(graph.direction.item()) == target],
    )


def complete_direction_pairs(graphs: list[Data]) -> list[tuple[Data, Data]]:
    grouped: dict[tuple[str, str], dict[int, Data]] = {}
    for graph in graphs:
        task, _ = str(graph.sample_name).rsplit("_", 1)
        key = str(graph.subject), task
        direction = int(graph.direction.item())
        if direction in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate direction {direction} for subject-task pair {key}")
        grouped[key][direction] = graph
    pairs = []
    for key in sorted(grouped):
        directions = grouped[key]
        if set(directions) != {0, 1}:
            continue
        lr, rl = directions[0], directions[1]
        if int(lr.y.item()) != int(rl.y.item()):
            raise ValueError(f"task-label mismatch for LR/RL pair {key}")
        pairs.append((lr, rl))
    return pairs


def partition_direction_pairs(
    pairs: list[tuple[Data, Data]], n_batches: int, shuffle: bool = True
) -> list[list[tuple[Data, Data]]]:
    if not pairs or n_batches < 1:
        return []
    ordered = pairs[:]
    if shuffle:
        random.shuffle(ordered)
    n_batches = min(n_batches, len(ordered))
    return [ordered[index::n_batches] for index in range(n_batches)]


def pair_js_divergence(
    lr_log_probs: torch.Tensor, rl_log_probs: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    if lr_log_probs.shape != rl_log_probs.shape:
        raise ValueError("LR and RL prediction tensors must have matching shapes")
    log_middle = torch.logsumexp(
        torch.stack([lr_log_probs, rl_log_probs]), dim=0
    ) - np.log(2.0)
    values = 0.5 * (
        F.kl_div(log_middle, lr_log_probs, reduction="none", log_target=True).sum(dim=1)
        + F.kl_div(log_middle, rl_log_probs, reduction="none", log_target=True).sum(dim=1)
    )
    if reduction == "none":
        return values
    if reduction != "mean":
        raise ValueError(f"unsupported pair JS reduction: {reduction}")
    return values.mean()


def pair_task_weights(
    pairs: list[tuple[Data, Data]], mode: str
) -> dict[int, float] | None:
    if mode == "none":
        return None
    if mode != "sqrt_inverse":
        raise ValueError(f"unsupported pair balance mode: {mode}")
    counts = Counter(int(lr.y.item()) for lr, _ in pairs)
    raw = {label: 1 / np.sqrt(count) for label, count in counts.items()}
    normalizer = sum(raw[int(lr.y.item())] for lr, _ in pairs) / len(pairs)
    return {label: float(weight / normalizer) for label, weight in raw.items()}


def pair_weight_at_epoch(target: float, epoch: int, start: int, warmup_epochs: int) -> float:
    if target == 0 or epoch < start:
        return 0.0
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, (epoch - start + 1) / warmup_epochs)


def pair_consistency_loss(
    model: Network,
    pairs: list[tuple[Data, Data]],
    device: torch.device,
    task_weights: dict[int, float] | None = None,
) -> torch.Tensor:
    if not pairs:
        return torch.tensor(0.0, device=device)
    pair_batch = Batch.from_data_list(
        [lr for lr, _ in pairs] + [rl for _, rl in pairs]
    ).to(device)
    was_training = model.training
    model.eval()
    try:
        output, _, _, _, _ = model(
            pair_batch.x,
            pair_batch.edge_index,
            pair_batch.batch,
            pair_batch.edge_attr,
            pair_batch.pos,
        )
    finally:
        model.train(was_training)
    n_pairs = len(pairs)
    losses = pair_js_divergence(output[:n_pairs], output[n_pairs:], reduction="none")
    if task_weights is None:
        return losses.mean()
    weights = torch.tensor(
        [task_weights[int(lr.y.item())] for lr, _ in pairs],
        dtype=losses.dtype,
        device=device,
    )
    return (losses * weights).mean()


def mean_pair_consistency_loss(
    model: Network,
    pairs: list[tuple[Data, Data]],
    device: torch.device,
    pair_batch_size: int,
    task_weights: dict[int, float] | None = None,
) -> float:
    if not pairs:
        return 0.0
    weighted_loss = 0.0
    for start in range(0, len(pairs), pair_batch_size):
        batch = pairs[start:start + pair_batch_size]
        loss = pair_consistency_loss(model, batch, device, task_weights)
        weighted_loss += float(loss.item()) * len(batch)
    return weighted_loss / len(pairs)


def topk_loss(s: torch.Tensor, ratio: float, device: torch.device) -> torch.Tensor:
    if len(s) == 0:
        return torch.tensor(0.0, device=device)
    if ratio > 0.5:
        ratio = 1 - ratio
    k = max(1, int(s.size(1) * ratio))
    s = s.sort(dim=1).values
    return -torch.log(s[:, -k:] + EPS).mean() - torch.log(1 - s[:, :k] + EPS).mean()


def consist_loss(s: torch.Tensor, device: torch.device) -> torch.Tensor:
    if len(s) == 0:
        return torch.tensor(0.0, device=device)
    w = torch.ones(s.shape[0], s.shape[0], device=device)
    d = torch.eye(s.shape[0], device=device) * torch.sum(w, dim=1)
    laplacian = d - w
    return torch.trace(torch.transpose(s, 0, 1) @ laplacian @ s) / (s.shape[0] * s.shape[0])


def prediction_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
    }


def model_forward(
    model: Network, data: Data, direction_adversarial: bool
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if direction_adversarial:
        output, direction_output, w1, w2, s1, s2, _ = model.forward_with_direction(
            data.x, data.edge_index, data.batch, data.edge_attr, data.pos
        )
        return output, direction_output, w1, w2, s1, s2
    output, w1, w2, s1, s2 = model(
        data.x, data.edge_index, data.batch, data.edge_attr, data.pos
    )
    return output, None, w1, w2, s1, s2


def collect_model_outputs(
    model: Network,
    loader: DataLoader,
    device: torch.device,
    collect_scores: bool = False,
    collect_embeddings: bool = False,
    collect_direction_head: bool = False,
) -> dict[str, object]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    direction_preds: list[np.ndarray] = []
    subjects: list[str] = []
    sample_names: list[str] = []
    scores: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if collect_direction_head and model.direction_adversarial:
                output, direction_output, _, _, s1, _, graph_embedding = model.forward_with_direction(
                    data.x, data.edge_index, data.batch, data.edge_attr, data.pos,
                    reversal_scale=0.0,
                )
                direction_preds.append(direction_output.argmax(dim=1).cpu().numpy())
            else:
                output, _, _, s1, _ = model(
                    data.x, data.edge_index, data.batch, data.edge_attr, data.pos
                )
                graph_embedding = (
                    model.graph_embedding(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                    if collect_embeddings else None
                )
            preds.append(output.argmax(dim=1).cpu().numpy())
            log_probs.append(output.cpu().numpy())
            trues.append(data.y.cpu().numpy())
            directions.append(data.direction.cpu().numpy())
            subjects.extend(str(value) for value in data.subject)
            sample_names.extend(str(value) for value in data.sample_name)
            if collect_scores:
                scores.append(s1.cpu().numpy())
            if collect_embeddings:
                if graph_embedding is None:
                    graph_embedding = model.graph_embedding(
                        data.x, data.edge_index, data.batch, data.edge_attr, data.pos
                    )
                embeddings.append(graph_embedding.cpu().numpy())
    return {
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
        "direction": np.concatenate(directions),
        "direction_pred": np.concatenate(direction_preds) if direction_preds else None,
        "subjects": subjects,
        "sample_names": sample_names,
        "scores": np.concatenate(scores) if scores else None,
        "embeddings": np.concatenate(embeddings) if embeddings else None,
        "log_probs": np.concatenate(log_probs),
    }


def evaluate(
    model: Network, loader: DataLoader, device: torch.device, collect_scores: bool = False
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str], list[str], np.ndarray | None]:
    outputs = collect_model_outputs(model, loader, device, collect_scores=collect_scores)
    return (
        prediction_metrics(outputs["true"], outputs["pred"]),
        outputs["pred"],
        outputs["true"],
        outputs["subjects"],
        outputs["sample_names"],
        outputs["scores"],
    )


def direction_subgroup_metrics(outputs: dict[str, object]) -> dict[str, dict[str, float]]:
    result = {}
    for label, name in ((0, "LR"), (1, "RL")):
        mask = outputs["direction"] == label
        if mask.any():
            result[name] = prediction_metrics(outputs["true"][mask], outputs["pred"][mask])
    return result


def bootstrap_interval(
    values: np.ndarray, subjects: np.ndarray, seed: int, n_samples: int
) -> dict[str, float | int] | None:
    unique_subjects = np.unique(subjects)
    if n_samples <= 0 or len(unique_subjects) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_samples):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
        estimates.append(float(values[indices].mean()))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "n_bootstrap": n_samples}


def paired_direction_metrics(
    outputs: dict[str, object], topk: int, seed: int = 123, bootstrap_samples: int = 0
) -> dict[str, object]:
    pairs: dict[tuple[str, str], dict[str, int]] = {}
    scores = outputs["scores"]
    for index, (subject, sample_name) in enumerate(zip(outputs["subjects"], outputs["sample_names"])):
        task, direction = str(sample_name).rsplit("_", 1)
        pairs.setdefault((str(subject), task), {})[direction] = index
    complete = [
        (subject, task, pair)
        for (subject, task), pair in pairs.items()
        if set(pair) == {"LR", "RL"}
    ]
    if not complete:
        return {"n_pairs": 0}
    agreements = []
    both_correct = []
    one_correct = []
    score_jaccards = []
    pair_js = []
    ensemble_true = []
    ensemble_pred = []
    pair_subjects = []
    pair_tasks = []
    log_probs = outputs["log_probs"]
    for subject, task, pair in complete:
        lr, rl = pair["LR"], pair["RL"]
        lr_correct = bool(outputs["pred"][lr] == outputs["true"][lr])
        rl_correct = bool(outputs["pred"][rl] == outputs["true"][rl])
        agreements.append(outputs["pred"][lr] == outputs["pred"][rl])
        both_correct.append(lr_correct and rl_correct)
        one_correct.append(lr_correct != rl_correct)
        lr_probs = np.exp(log_probs[lr])
        rl_probs = np.exp(log_probs[rl])
        middle = np.clip((lr_probs + rl_probs) / 2, EPS, 1)
        pair_js.append(
            0.5 * (
                np.sum(lr_probs * (log_probs[lr] - np.log(middle)))
                + np.sum(rl_probs * (log_probs[rl] - np.log(middle)))
            )
        )
        ensemble_true.append(int(outputs["true"][lr]))
        ensemble_pred.append(int(np.argmax(middle)))
        pair_subjects.append(subject)
        pair_tasks.append(task)
        if scores is not None:
            lr_top = set(np.argpartition(scores[lr], -topk)[-topk:].tolist())
            rl_top = set(np.argpartition(scores[rl], -topk)[-topk:].tolist())
            score_jaccards.append(len(lr_top & rl_top) / len(lr_top | rl_top))
    agreements_array = np.asarray(agreements, dtype=float)
    both_correct_array = np.asarray(both_correct, dtype=float)
    one_correct_array = np.asarray(one_correct, dtype=float)
    ensemble_true_array = np.asarray(ensemble_true)
    ensemble_pred_array = np.asarray(ensemble_pred)
    subjects_array = np.asarray(pair_subjects)
    per_task = {}
    for task in sorted(set(pair_tasks)):
        mask = np.asarray(pair_tasks) == task
        per_task[task] = {
            "n_pairs": int(mask.sum()),
            "prediction_agreement": float(agreements_array[mask].mean()),
            "both_correct": float(both_correct_array[mask].mean()),
            "ensemble_accuracy": float((ensemble_true_array[mask] == ensemble_pred_array[mask]).mean()),
            "mean_prediction_js": float(np.asarray(pair_js)[mask].mean()),
        }
    result = {
        "n_pairs": len(complete),
        "prediction_agreement": float(agreements_array.mean()),
        "macro_task_prediction_agreement": float(np.mean([
            values["prediction_agreement"] for values in per_task.values()
        ])),
        "both_correct": float(both_correct_array.mean()),
        "one_correct": float(one_correct_array.mean()),
        "both_wrong": float(1 - both_correct_array.mean() - one_correct_array.mean()),
        "mean_prediction_js": float(np.mean(pair_js)),
        "ensemble_metrics": prediction_metrics(ensemble_true_array, ensemble_pred_array),
        "per_task": per_task,
        "pool1_topk_jaccard": float(np.mean(score_jaccards)) if score_jaccards else None,
        "pool1_topk": topk,
    }
    result["subject_bootstrap_95ci"] = {
        "prediction_agreement": bootstrap_interval(
            agreements_array, subjects_array, seed, bootstrap_samples
        ),
        "both_correct": bootstrap_interval(
            both_correct_array, subjects_array, seed + 1, bootstrap_samples
        ),
        "ensemble_accuracy": bootstrap_interval(
            (ensemble_true_array == ensemble_pred_array).astype(float),
            subjects_array,
            seed + 2,
            bootstrap_samples,
        ),
    }
    return result


def direction_probe_metrics(
    train_outputs: dict[str, object], test_outputs: dict[str, object], seed: int
) -> dict[str, float] | None:
    if len(np.unique(train_outputs["direction"])) < 2 or len(np.unique(test_outputs["direction"])) < 2:
        return None
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    probe.fit(train_outputs["embeddings"], train_outputs["direction"])
    pred = probe.predict(test_outputs["embeddings"])
    return prediction_metrics(test_outputs["direction"], pred)


def compute_loss(output: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor,
                 s2: torch.Tensor, y: torch.Tensor, args: argparse.Namespace,
                 device: torch.device, class_weights: torch.Tensor | None = None,
                 direction_output: torch.Tensor | None = None,
                 direction: torch.Tensor | None = None) -> torch.Tensor:
    loss_c = F.nll_loss(output, y, weight=class_weights)
    loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
    loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
    loss_tpk1 = topk_loss(s1, args.ratio, device)
    loss_tpk2 = topk_loss(s2, args.ratio, device)
    loss_consist = torch.tensor(0.0, device=device)
    for cls in range(args.nclass):
        loss_consist = loss_consist + consist_loss(s1[y == cls], device)
    loss = (
        args.lamb0 * loss_c + args.lamb1 * loss_p1 + args.lamb2 * loss_p2
        + args.lamb3 * loss_tpk1 + args.lamb4 * loss_tpk2 + args.lamb5 * loss_consist
    )
    if args.direction_adv_weight > 0:
        if direction_output is None or direction is None:
            raise ValueError("direction-adversarial loss requires direction predictions and labels")
        loss = loss + args.direction_adv_weight * F.nll_loss(direction_output, direction)
    return loss


def main() -> None:
    args = parse_args()
    if args.direction_adv_weight < 0:
        raise ValueError("--direction_adv_weight must be non-negative")
    if args.pair_consistency_weight < 0:
        raise ValueError("--pair_consistency_weight must be non-negative")
    if args.pair_warmup_start < 0 or args.pair_warmup_epochs < 0:
        raise ValueError("pair warm-up values must be non-negative")
    if args.checkpoint_bacc_tolerance < 0:
        raise ValueError("--checkpoint_bacc_tolerance must be non-negative")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping values must be non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap_samples must be non-negative")
    if args.direction_adv_weight > 0 and args.direction_protocol != "mixed":
        raise ValueError("direction-adversarial training requires --direction_protocol mixed")
    if args.pair_consistency_weight > 0 and args.direction_protocol != "mixed":
        raise ValueError("LR/RL pair consistency training requires --direction_protocol mixed")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.output_dir is not None:
        args.save_path = args.output_dir / "model"
        args.log_path = args.output_dir / "tensorboard"
    args.save_path.mkdir(parents=True, exist_ok=True)
    args.log_path.mkdir(parents=True, exist_ok=True)

    graphs, task_label_map = load_hcp_subjectwise_graphs(args)
    args.nclass = len(task_label_map)
    args.nroi = int(graphs[0].num_nodes)
    args.indim = args.nroi
    model = Network(args.indim, args.ratio, args.nclass, R=args.nroi, dim1=args.dim1,
                    dim2=args.dim2, fc_dim=args.fc_dim, dropout=args.dropout,
                    roi_aware=args.conv_type == "ra",
                    direction_adversarial=args.direction_adv_weight > 0)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.split_manifest is None:
        train_graphs, val_graphs, test_graphs, split_info = split_subjects(
            graphs, args.n_folds, args.fold, args.val_ratio, args.seed
        )
    else:
        train_graphs, val_graphs, test_graphs, split_info = split_subjects_from_manifest(
            graphs, args.split_manifest, args.fold
        )
    train_graphs, val_graphs, test_graphs = apply_direction_protocol(
        train_graphs, val_graphs, test_graphs, args.direction_protocol
    )
    train_pairs = complete_direction_pairs(train_graphs)
    val_pairs = complete_direction_pairs(val_graphs)
    test_pairs = complete_direction_pairs(test_graphs)
    train_pair_task_weights = pair_task_weights(train_pairs, args.pair_balance)
    if not train_graphs or not val_graphs or not test_graphs:
        raise ValueError(f"empty split after applying direction protocol {args.direction_protocol}")
    if args.pair_consistency_weight > 0 and not train_pairs:
        raise ValueError("pair consistency training requires at least one complete LR/RL training pair")
    for split_name, split_graphs in (
        ("train", train_graphs), ("validation", val_graphs), ("test", test_graphs)
    ):
        if len(label_counts(split_graphs)) != args.nclass:
            raise ValueError(
                f"{split_name} split lacks one or more task classes after applying "
                f"direction protocol {args.direction_protocol}: {label_counts(split_graphs)}"
            )
    run_info = {
        "n_graphs": len(graphs),
        "n_subjects": len({graph.subject for graph in graphs}),
        "n_train_graphs": len(train_graphs),
        "n_val_graphs": len(val_graphs),
        "n_test_graphs": len(test_graphs),
        "train_label_counts": label_counts(train_graphs),
        "val_label_counts": label_counts(val_graphs),
        "test_label_counts": label_counts(test_graphs),
        "train_direction_counts": direction_counts(train_graphs),
        "val_direction_counts": direction_counts(val_graphs),
        "test_direction_counts": direction_counts(test_graphs),
        "n_train_direction_pairs": len(train_pairs),
        "n_val_direction_pairs": len(val_pairs),
        "n_test_direction_pairs": len(test_pairs),
        "direction_protocol": args.direction_protocol,
        "direction_adv_weight": args.direction_adv_weight,
        "pair_consistency_weight": args.pair_consistency_weight,
        "pair_balance": args.pair_balance,
        "pair_task_weights": train_pair_task_weights,
        "pair_warmup_start": args.pair_warmup_start,
        "pair_warmup_epochs": args.pair_warmup_epochs,
        "checkpoint_selection": args.checkpoint_selection,
        "checkpoint_bacc_tolerance": args.checkpoint_bacc_tolerance,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
        "nclass": args.nclass,
        "nroi": args.nroi,
        "model_parameter_count": model_parameter_count,
        "task_label_map": task_label_map,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        **split_info,
    }
    print(json.dumps(run_info, indent=2, sort_keys=True))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "run_info.json").open("w") as handle:
            json.dump(run_info, handle, indent=2, sort_keys=True)
        with (args.output_dir / "args.json").open("w") as handle:
            json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                      handle, indent=2, sort_keys=True)
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(args.log_path, str(args.fold)))
    train_loader = DataLoader(train_graphs, batch_size=args.batchSize, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batchSize, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batchSize, shuffle=False)
    model = model.to(device)
    print(model)
    class_weights = None
    if args.class_weight == "balanced":
        counts = label_counts(train_graphs)
        class_weights = torch.tensor(
            [len(train_graphs) / (args.nclass * counts[class_index]) for class_index in range(args.nclass)],
            dtype=torch.float32,
            device=device,
        )
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weightdecay, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = -1e10
    best_primary_epoch = -1
    selected_epoch = -1
    checkpoint_candidates: list[dict[str, object]] = []
    epochs_without_improvement = 0
    epochs_trained = 0
    stopped_early = False
    pair_topk = max(1, int(round(args.nroi * args.ratio * args.ratio)))
    for epoch in range(args.n_epochs):
        since = time.time()
        model.train()
        base_loss_all = 0.0
        pair_loss_all = 0.0
        pair_count = 0
        pair_batches = (
            partition_direction_pairs(train_pairs, len(train_loader))
            if args.pair_consistency_weight > 0 else []
        )
        effective_pair_weight = pair_weight_at_epoch(
            args.pair_consistency_weight,
            epoch,
            args.pair_warmup_start,
            args.pair_warmup_epochs,
        )
        s1_list: list[np.ndarray] = []
        s2_list: list[np.ndarray] = []
        for step, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output, direction_output, w1, w2, s1, s2 = model_forward(
                model, data, args.direction_adv_weight > 0
            )
            base_loss = compute_loss(
                output, w1, w2, s1, s2, data.y, args, device, class_weights,
                direction_output, data.direction,
            )
            pair_loss = (
                pair_consistency_loss(
                    model, pair_batches[step], device, train_pair_task_weights
                )
                if step < len(pair_batches) else torch.tensor(0.0, device=device)
            )
            loss = base_loss + effective_pair_weight * pair_loss
            loss.backward()
            optimizer.step()
            base_loss_all += float(base_loss.item()) * data.num_graphs
            if step < len(pair_batches):
                pair_loss_all += float(pair_loss.item()) * len(pair_batches[step])
                pair_count += len(pair_batches[step])
            s1_list.append(s1.view(-1).detach().cpu().numpy())
            s2_list.append(s2.view(-1).detach().cpu().numpy())
            writer.add_scalar("train/loss", loss, epoch * len(train_loader) + step)
        train_pair_loss = pair_loss_all / pair_count if pair_count else 0.0
        train_loss = base_loss_all / len(train_graphs) + effective_pair_weight * train_pair_loss
        train_metrics, _, _, _, _, _ = evaluate(model, train_loader, device)
        val_outputs_epoch = collect_model_outputs(model, val_loader, device)
        val_metrics = prediction_metrics(val_outputs_epoch["true"], val_outputs_epoch["pred"])
        val_pair_metrics_epoch = paired_direction_metrics(val_outputs_epoch, pair_topk)
        model.eval()
        val_loss_all = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output, direction_output, w1, w2, s1, s2 = model_forward(
                    model, data, args.direction_adv_weight > 0
                )
                val_loss = compute_loss(
                    output, w1, w2, s1, s2, data.y, args, device, class_weights,
                    direction_output, data.direction,
                )
                val_loss_all += float(val_loss.item()) * data.num_graphs
            val_pair_loss = mean_pair_consistency_loss(
                model,
                val_pairs,
                device,
                max(1, args.batchSize // 2),
                train_pair_task_weights,
            )
        val_loss_mean = (
            val_loss_all / len(val_graphs)
            + effective_pair_weight * val_pair_loss
        )
        writer.add_scalars("Acc", {"train_acc": train_metrics["accuracy"], "val_acc": val_metrics["accuracy"],
                                   "train_bacc": train_metrics["balanced_accuracy"],
                                   "val_bacc": val_metrics["balanced_accuracy"]}, epoch)
        writer.add_scalars("Loss", {"train_loss": train_loss, "val_loss": val_loss_mean}, epoch)
        writer.add_scalars(
            "PairConsistency",
            {"train_js": train_pair_loss, "val_js": val_pair_loss},
            epoch,
        )
        writer.add_scalar("PairConsistency/effective_weight", effective_pair_weight, epoch)
        if s1_list:
            writer.add_histogram("Hist/hist_s1", np.hstack(s1_list), epoch)
            writer.add_histogram("Hist/hist_s2", np.hstack(s2_list), epoch)
        current_score = -val_loss_mean if args.best_metric == "loss" else (
            val_metrics["accuracy"] if args.best_metric == "acc" else val_metrics["balanced_accuracy"]
        )
        if current_score > best_score:
            best_score = current_score
            best_primary_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.checkpoint_selection == "primary":
            if current_score >= best_score:
                best_model_wts = copy.deepcopy(model.state_dict())
                selected_epoch = epoch
        else:
            candidate = {
                "epoch": epoch,
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "pair_agreement": val_pair_metrics_epoch.get("prediction_agreement", 0.0),
                "pair_js": val_pair_metrics_epoch.get("mean_prediction_js", float("inf")),
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            checkpoint_candidates.append(candidate)
            max_bacc = max(item["balanced_accuracy"] for item in checkpoint_candidates)
            checkpoint_candidates = [
                item for item in checkpoint_candidates
                if item["balanced_accuracy"] >= max_bacc - args.checkpoint_bacc_tolerance
            ]
            selected = max(
                checkpoint_candidates,
                key=lambda item: (
                    item["pair_agreement"],
                    -item["pair_js"],
                    item["balanced_accuracy"],
                ),
            )
            best_model_wts = selected["state_dict"]
            selected_epoch = int(selected["epoch"])
        scheduler.step()
        epochs_trained = epoch + 1
        elapsed = time.time() - since
        print(
            f"Epoch: {epoch:03d}, {elapsed // 60:.0f}m {elapsed % 60:.0f}s, "
            f"Train Loss: {train_loss:.7f}, Train Acc: {train_metrics['accuracy']:.7f}, "
            f"Train Balanced Acc: {train_metrics['balanced_accuracy']:.7f}, Val Loss: {val_loss_mean:.7f}, "
            f"Val Acc: {val_metrics['accuracy']:.7f}, Val Balanced Acc: {val_metrics['balanced_accuracy']:.7f}, "
            f"Train Pair JS: {train_pair_loss:.7f}, Val Pair JS: {val_pair_loss:.7f}, "
            f"Pair Weight: {effective_pair_weight:.7f}, Selected Epoch: {selected_epoch:03d}",
            flush=True,
        )
        if (
            args.early_stopping_patience > 0
            and epochs_trained >= args.early_stopping_min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch:03d}; best primary epoch {best_primary_epoch:03d}",
                flush=True,
            )
            break

    if args.save_model:
        torch.save(best_model_wts, args.save_path / f"{args.fold}.pth")

    model.load_state_dict(best_model_wts)
    train_outputs = collect_model_outputs(
        model, train_loader, device, collect_embeddings=True
    )
    val_outputs = collect_model_outputs(model, val_loader, device)
    test_outputs = collect_model_outputs(
        model,
        test_loader,
        device,
        collect_scores=True,
        collect_embeddings=True,
        collect_direction_head=args.direction_adv_weight > 0,
    )
    test_metrics = prediction_metrics(test_outputs["true"], test_outputs["pred"])
    val_selected_metrics = prediction_metrics(val_outputs["true"], val_outputs["pred"])
    preds = test_outputs["pred"]
    trues = test_outputs["true"]
    subjects = test_outputs["subjects"]
    sample_names = test_outputs["sample_names"]
    pool1_scores = test_outputs["scores"]
    paired_metrics = paired_direction_metrics(
        test_outputs, pair_topk, args.seed, args.bootstrap_samples
    )
    val_paired_metrics = paired_direction_metrics(val_outputs, pair_topk)
    probe_metrics = direction_probe_metrics(train_outputs, test_outputs, args.seed)
    direction_head_metrics = (
        prediction_metrics(test_outputs["direction"], test_outputs["direction_pred"])
        if test_outputs["direction_pred"] is not None else None
    )
    print("===========================")
    print(f"Test Acc: {test_metrics['accuracy']:.7f}, Test Balanced Acc: {test_metrics['balanced_accuracy']:.7f}")
    print("Confusion matrix")
    print(confusion_matrix(trues, preds))
    print(classification_report(trues, preds, zero_division=0))
    print(args)
    if args.output_dir is not None:
        summary = {
            "fold": args.fold,
            "metrics": test_metrics,
            "confusion_matrix": confusion_matrix(trues, preds).tolist(),
            "classification_report": classification_report(trues, preds, output_dict=True, zero_division=0),
            "best_validation_score": best_score,
            "best_primary_epoch": best_primary_epoch,
            "selected_epoch": selected_epoch,
            "epochs_trained": epochs_trained,
            "stopped_early": stopped_early,
            "selected_checkpoint_validation_metrics": val_selected_metrics,
            "selected_checkpoint_validation_paired_direction_metrics": val_paired_metrics,
            "model_parameter_count": model_parameter_count,
            "direction_subgroup_metrics": direction_subgroup_metrics(test_outputs),
            "paired_direction_metrics": paired_metrics,
            "direction_probe_metrics": probe_metrics,
            "direction_head_metrics": direction_head_metrics,
        }
        with (args.output_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
            writer_csv = csv.DictWriter(
                handle,
                fieldnames=["subject", "sample_name", "direction", "true", "pred"],
                lineterminator="\n",
            )
            writer_csv.writeheader()
            writer_csv.writerows(
                {
                    "subject": subject,
                    "sample_name": sample_name,
                    "direction": "LR" if int(direction) == 0 else "RL",
                    "true": int(true),
                    "pred": int(pred),
                }
                for subject, sample_name, direction, true, pred
                in zip(subjects, sample_names, test_outputs["direction"], trues, preds)
            )
        np.savez_compressed(
            args.output_dir / "test_pool1_scores.npz",
            scores=pool1_scores,
            subjects=np.asarray(subjects),
            sample_names=np.asarray(sample_names),
            direction=test_outputs["direction"],
            true=trues,
            pred=preds,
        )
        np.savez_compressed(
            args.output_dir / "test_graph_embeddings.npz",
            embeddings=test_outputs["embeddings"],
            subjects=np.asarray(subjects),
            sample_names=np.asarray(sample_names),
            direction=test_outputs["direction"],
            true=trues,
            pred=preds,
        )
        community_membership = model.community_membership()
        if community_membership is not None:
            np.save(args.output_dir / "community_membership_alpha_positive.npy",
                    community_membership.detach().cpu().numpy())


if __name__ == "__main__":
    main()
