"""Paper-style ROI top-k pooling."""

from __future__ import annotations

import torch
from torch.nn import Parameter
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import scatter

from net.inits import uniform


class ROITopKPooling(torch.nn.Module):
    """Top-k pooling with per-graph standardized projection scores."""

    def __init__(self, in_channels: int, ratio: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.eps = eps
        self.weight = Parameter(torch.empty(1, in_channels))
        self.connect = FilterEdges()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        projection = (x * self.weight).sum(dim=-1) / self.weight.norm(p=2).clamp_min(self.eps)
        mean = scatter(projection, batch, reduce="mean")
        centered = projection - mean[batch]
        variance = scatter(centered.square(), batch, reduce="mean")
        normalized = centered / variance[batch].add(self.eps).sqrt()
        scores = normalized.sigmoid()
        perm = topk(normalized, self.ratio, batch)
        selected_scores = scores[perm]
        pooled_x = x[perm] * selected_scores.view(-1, 1)
        cluster_index = torch.arange(perm.numel(), device=perm.device)
        select_output = SelectOutput(
            node_index=perm,
            num_nodes=x.size(0),
            cluster_index=cluster_index,
            num_clusters=perm.numel(),
            weight=selected_scores,
        )
        connect_output = self.connect(select_output, edge_index, edge_attr, batch)
        return (
            pooled_x,
            connect_output.edge_index,
            connect_output.edge_attr,
            connect_output.batch,
            perm,
            selected_scores,
            scores,
        )
