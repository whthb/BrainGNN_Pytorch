import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm

from net.braingraphconv import MyNNConv
from net.roi_pool import ROITopKPooling


class GradientReversal(torch.autograd.Function):
    """Pass values through unchanged and reverse their encoder gradients."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


class SharedKernel(nn.Module):
    """Return the same vectorized graph-convolution kernel for every ROI."""

    def __init__(self, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, pseudo):
        return self.weight.expand(pseudo.size(0), -1)


##########################################################################################################################
class Network(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, k=8, R=200, dim1=32,
                 dim2=32, fc_dim=512, dropout=0.5, roi_aware=True,
                 direction_adversarial=False):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network, self).__init__()

        self.indim = indim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = fc_dim
        self.dim4 = 256
        self.dim5 = 8
        self.k = k
        self.R = R
        self.dropout = dropout
        self.roi_aware = roi_aware
        self.direction_adversarial = direction_adversarial

        self.n1 = (
            nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(),
                          nn.Linear(self.k, self.dim1 * self.indim))
            if roi_aware else SharedKernel(self.dim1 * self.indim)
        )
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = ROITopKPooling(self.dim1, ratio=ratio)
        self.n2 = (
            nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(),
                          nn.Linear(self.k, self.dim2 * self.dim1))
            if roi_aware else SharedKernel(self.dim2 * self.dim1)
        )
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = ROITopKPooling(self.dim2, ratio=ratio)

        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, nclass)
        self.direction_head = (
            nn.Sequential(
                nn.Linear(self.dim3, min(128, self.dim3)),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(min(128, self.dim3), 2),
            )
            if direction_adversarial else None
        )


    def encode_graph(self, x, edge_index, batch, edge_attr, pos):
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, _, score1 = self.pool1(x, edge_index, edge_attr, batch)

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, _, score2 = self.pool2(x, edge_index,edge_attr, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x= F.dropout(x, p=self.dropout, training=self.training)
        return x, self.pool1.weight, self.pool2.weight, score1, score2

    def forward(self, x, edge_index, batch, edge_attr, pos):
        graph_embedding, w1, w2, score1, score2 = self.encode_graph(
            x, edge_index, batch, edge_attr, pos
        )
        output = F.log_softmax(self.fc3(graph_embedding), dim=-1)
        return (
            output,
            w1,
            w2,
            score1.view(output.size(0), -1),
            score2.view(output.size(0), -1),
        )

    def forward_with_direction(self, x, edge_index, batch, edge_attr, pos,
                               reversal_scale=1.0):
        if self.direction_head is None:
            raise RuntimeError("direction-adversarial head is not enabled")
        graph_embedding, w1, w2, score1, score2 = self.encode_graph(
            x, edge_index, batch, edge_attr, pos
        )
        output = F.log_softmax(self.fc3(graph_embedding), dim=-1)
        reversed_embedding = GradientReversal.apply(graph_embedding, reversal_scale)
        direction_output = F.log_softmax(self.direction_head(reversed_embedding), dim=-1)
        return (
            output,
            direction_output,
            w1,
            w2,
            score1.view(output.size(0), -1),
            score2.view(output.size(0), -1),
            graph_embedding,
        )

    def graph_embedding(self, x, edge_index, batch, edge_attr, pos):
        return self.encode_graph(x, edge_index, batch, edge_attr, pos)[0]

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def community_membership(self):
        if not self.roi_aware:
            return None
        return torch.relu(self.n1[0].weight).transpose(0, 1)
