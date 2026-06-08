#!/usr/bin/env python
"""Run a small BrainGNN smoke test on HCP subject-wise task labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

from net.braingnn import Network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--edge-source", choices=["corr", "pcorr"], default="corr")
    parser.add_argument("--edge-top-percent", type=float, default=0.10)
    parser.add_argument("--positive-edges-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def select_edges(matrix: np.ndarray, top_percent: float, positive_only: bool) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.where(matrix > 0, matrix, 0.0) if positive_only else np.abs(matrix)
    np.fill_diagonal(matrix, 0.0)
    if top_percent <= 0 or top_percent > 1:
        raise ValueError("--edge-top-percent must be in (0, 1]")
    num_nodes = matrix.shape[0]
    topk = max(1, int(np.ceil((num_nodes - 1) * top_percent)))
    topk_cols = np.argpartition(matrix, -topk, axis=1)[:, -topk:]
    rows = np.arange(num_nodes)[:, None]
    mask = np.zeros_like(matrix, dtype=bool)
    mask[rows, topk_cols] = matrix[rows, topk_cols] > 0
    mask = mask | mask.T
    return matrix * mask


def matrix_to_edge_tensors(matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    row, col = np.nonzero(matrix)
    edge_index = torch.from_numpy(np.stack([row, col])).long()
    edge_attr = torch.from_numpy(matrix[row, col]).float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, matrix.shape[0], matrix.shape[1])
    return edge_index.long(), edge_attr.float().view(-1, 1)


def load_graphs(data_root: Path, edge_source: str, edge_top_percent: float,
                positive_only: bool) -> tuple[list[Data], dict[str, int]]:
    with (data_root / "task_label_map.json").open() as handle:
        label_map = json.load(handle)

    graphs: list[Data] = []
    for subject_file in sorted((data_root / "subjects").glob("*.h5")):
        with h5py.File(subject_file, "r") as handle:
            for sample_name in sorted(handle["samples"].keys()):
                group = handle["samples"][sample_name]
                x = np.nan_to_num(group["corr"][()], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                edge_matrix = select_edges(group[edge_source][()], edge_top_percent, positive_only)
                edge_index, edge_attr = matrix_to_edge_tensors(edge_matrix)
                num_nodes = x.shape[0]
                graph = Data(
                    x=torch.from_numpy(x).float(),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(int(group["task_label"][()][0]), dtype=torch.long),
                    pos=torch.eye(num_nodes, dtype=torch.float32),
                )
                graph.subject = subject_file.stem
                graph.sample_name = sample_name
                graphs.append(graph)
    if not graphs:
        raise ValueError(f"no subject-wise samples found under {data_root / 'subjects'}")
    return graphs, {str(key): int(value) for key, value in label_map.items()}


def subject_split(graphs: list[Data], seed: int) -> tuple[list[Data], list[Data]]:
    subjects = sorted({graph.subject for graph in graphs})
    if len(subjects) < 2:
        raise ValueError("need at least two subjects for subject-wise smoke split")
    random.Random(seed).shuffle(subjects)
    n_train = max(1, min(int(round(len(subjects) * 0.5)), len(subjects) - 1))
    train_subjects = set(subjects[:n_train])
    return [g for g in graphs if g.subject in train_subjects], [g for g in graphs if g.subject not in train_subjects]


def evaluate(model: Network, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)[0]
            preds.append(output.argmax(dim=1).cpu().numpy())
            trues.append(data.y.cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    return float((pred == true).mean()), float(balanced_accuracy_score(true, pred))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    graphs, label_map = load_graphs(args.data_root, args.edge_source, args.edge_top_percent, args.positive_edges_only)
    train_graphs, test_graphs = subject_split(graphs, args.seed)
    num_nodes = int(graphs[0].num_nodes)
    num_classes = len(label_map)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(indim=num_nodes, ratio=0.5, nclass=num_classes, R=num_nodes,
                    dim1=16, dim2=16, fc_dim=64, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(json.dumps({
        "device": str(device),
        "n_graphs": len(graphs),
        "n_train": len(train_graphs),
        "n_test": len(test_graphs),
        "n_subjects": len({graph.subject for graph in graphs}),
        "n_roi": num_nodes,
        "n_classes": num_classes,
        "task_label_map": label_map,
        "train_subjects": sorted({graph.subject for graph in train_graphs}),
        "test_subjects": sorted({graph.subject for graph in test_graphs}),
    }, indent=2, sort_keys=True))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)[0]
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * data.num_graphs
            total_graphs += data.num_graphs
        train_acc, train_bacc = evaluate(model, train_loader, device)
        test_acc, test_bacc = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch} loss={total_loss / total_graphs:.6f} "
            f"train_acc={train_acc:.4f} train_bacc={train_bacc:.4f} "
            f"test_acc={test_acc:.4f} test_bacc={test_bacc:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
