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
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch_geometric.data import Data
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


def evaluate(
    model: Network, loader: DataLoader, device: torch.device, collect_scores: bool = False
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str], list[str], np.ndarray | None]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    subjects: list[str] = []
    sample_names: list[str] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _, _, s1, _ = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
            preds.append(output.argmax(dim=1).cpu().numpy())
            trues.append(data.y.cpu().numpy())
            subjects.extend(str(value) for value in data.subject)
            sample_names.extend(str(value) for value in data.sample_name)
            if collect_scores:
                scores.append(s1.cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    score_array = np.concatenate(scores) if scores else None
    return prediction_metrics(true, pred), pred, true, subjects, sample_names, score_array


def compute_loss(output: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor,
                 s2: torch.Tensor, y: torch.Tensor, args: argparse.Namespace,
                 device: torch.device, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    loss_c = F.nll_loss(output, y, weight=class_weights)
    loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
    loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
    loss_tpk1 = topk_loss(s1, args.ratio, device)
    loss_tpk2 = topk_loss(s2, args.ratio, device)
    loss_consist = torch.tensor(0.0, device=device)
    for cls in range(args.nclass):
        loss_consist = loss_consist + consist_loss(s1[y == cls], device)
    return (
        args.lamb0 * loss_c + args.lamb1 * loss_p1 + args.lamb2 * loss_p2
        + args.lamb3 * loss_tpk1 + args.lamb4 * loss_tpk2 + args.lamb5 * loss_consist
    )


def main() -> None:
    args = parse_args()
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
    if args.split_manifest is None:
        train_graphs, val_graphs, test_graphs, split_info = split_subjects(
            graphs, args.n_folds, args.fold, args.val_ratio, args.seed
        )
    else:
        train_graphs, val_graphs, test_graphs, split_info = split_subjects_from_manifest(
            graphs, args.split_manifest, args.fold
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
        "nclass": args.nclass,
        "nroi": args.nroi,
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
    model = Network(args.indim, args.ratio, args.nclass, R=args.nroi, dim1=args.dim1,
                    dim2=args.dim2, fc_dim=args.fc_dim, dropout=args.dropout,
                    roi_aware=args.conv_type == "ra").to(device)
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
    for epoch in range(args.n_epochs):
        since = time.time()
        model.train()
        loss_all = 0.0
        s1_list: list[np.ndarray] = []
        s2_list: list[np.ndarray] = []
        for step, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
            loss = compute_loss(output, w1, w2, s1, s2, data.y, args, device, class_weights)
            loss.backward()
            optimizer.step()
            loss_all += float(loss.item()) * data.num_graphs
            s1_list.append(s1.view(-1).detach().cpu().numpy())
            s2_list.append(s2.view(-1).detach().cpu().numpy())
            writer.add_scalar("train/loss", loss, epoch * len(train_loader) + step)
        train_loss = loss_all / len(train_graphs)
        train_metrics, _, _, _, _, _ = evaluate(model, train_loader, device)
        val_metrics, _, _, _, _, _ = evaluate(model, val_loader, device)
        model.eval()
        val_loss_all = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                val_loss = compute_loss(output, w1, w2, s1, s2, data.y, args, device, class_weights)
                val_loss_all += float(val_loss.item()) * data.num_graphs
        val_loss_mean = val_loss_all / len(val_graphs)
        writer.add_scalars("Acc", {"train_acc": train_metrics["accuracy"], "val_acc": val_metrics["accuracy"],
                                   "train_bacc": train_metrics["balanced_accuracy"],
                                   "val_bacc": val_metrics["balanced_accuracy"]}, epoch)
        writer.add_scalars("Loss", {"train_loss": train_loss, "val_loss": val_loss_mean}, epoch)
        if s1_list:
            writer.add_histogram("Hist/hist_s1", np.hstack(s1_list), epoch)
            writer.add_histogram("Hist/hist_s2", np.hstack(s2_list), epoch)
        current_score = -val_loss_mean if args.best_metric == "loss" else (
            val_metrics["accuracy"] if args.best_metric == "acc" else val_metrics["balanced_accuracy"]
        )
        if current_score > best_score:
            best_score = current_score
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.save_model:
                torch.save(best_model_wts, args.save_path / f"{args.fold}.pth")
        scheduler.step()
        elapsed = time.time() - since
        print(
            f"Epoch: {epoch:03d}, {elapsed // 60:.0f}m {elapsed % 60:.0f}s, "
            f"Train Loss: {train_loss:.7f}, Train Acc: {train_metrics['accuracy']:.7f}, "
            f"Train Balanced Acc: {train_metrics['balanced_accuracy']:.7f}, Val Loss: {val_loss_mean:.7f}, "
            f"Val Acc: {val_metrics['accuracy']:.7f}, Val Balanced Acc: {val_metrics['balanced_accuracy']:.7f}",
            flush=True,
        )

    model.load_state_dict(best_model_wts)
    test_metrics, preds, trues, subjects, sample_names, pool1_scores = evaluate(
        model, test_loader, device, collect_scores=True
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
        }
        with (args.output_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
            writer_csv = csv.DictWriter(
                handle,
                fieldnames=["subject", "sample_name", "true", "pred"],
                lineterminator="\n",
            )
            writer_csv.writeheader()
            writer_csv.writerows(
                {"subject": subject, "sample_name": sample_name, "true": int(true), "pred": int(pred)}
                for subject, sample_name, true, pred in zip(subjects, sample_names, trues, preds)
            )
        np.savez_compressed(
            args.output_dir / "test_pool1_scores.npz",
            scores=pool1_scores,
            subjects=np.asarray(subjects),
            sample_names=np.asarray(sample_names),
            true=trues,
            pred=preds,
        )
        community_membership = model.community_membership()
        if community_membership is not None:
            np.save(args.output_dir / "community_membership_alpha_positive.npy",
                    community_membership.detach().cpu().numpy())


if __name__ == "__main__":
    main()
