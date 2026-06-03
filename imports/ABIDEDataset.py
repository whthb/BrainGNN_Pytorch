import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from imports.read_abide_stats_parall import read_data


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 processed_filename='data.pt', edge_source='pcorr',
                 edge_topk=None, edge_top_percent=None,
                 positive_edges_only=False):
        self.root = root
        self.name = name
        self.processed_filename = processed_filename
        self.edge_source = edge_source
        self.edge_topk = edge_topk
        self.edge_top_percent = edge_top_percent
        self.positive_edges_only = positive_edges_only
        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return self.processed_filename

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir,
                                           edge_source=self.edge_source,
                                           edge_topk=self.edge_topk,
                                           edge_top_percent=self.edge_top_percent,
                                           positive_edges_only=self.positive_edges_only)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        data = self._data if hasattr(self, '_data') else self.data
        torch.save((data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
