import itertools as it
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from constants import ONE_HOT_FEATURE, WDEGREE_FEATURE

__all__ = ['CouplingGraphsDataset']


class CouplingGraphsDataset(Dataset):
    def __init__(self, coupling_matrices: np.ndarray, coupling_matrices_ids: List[int], rng: np.random.Generator,
                 device, feature_type, seed=None):
        self.rng = np.random.default_rng(seed=seed) if rng is None else rng
        self._coupling_matrices_dict = {k: v for k, v in zip(coupling_matrices_ids, coupling_matrices)}
        self._coupling_matrices_ids = coupling_matrices_ids

        self._coupling_matrices_tensor = torch.from_numpy(
            np.array([self._coupling_matrices_dict[k] for k in coupling_matrices_ids])
        ).type(torch.float32)

        self._device = device

        if len(np.unique(list(cm.shape[0] for cm in coupling_matrices))) == 1:
            self._num_qubits = coupling_matrices[0].shape[0]
        else:
            self._num_qubits = np.nan

        self._edge_indices, self._edge_weights, self._vertices_deg = self._build_graphs(coupling_matrices)

        if feature_type == ONE_HOT_FEATURE:
            assert not np.isnan(self._num_qubits), 'one_hot features only available for systems of equal size!'

            # use one hot encoded node features
            self._data_list = [
                # Data(x=torch.eye(self._num_qubits, dtype=torch.float32).to(device), edge_index=e, edge_weight=w)
                Data(x=torch.eye(self._num_qubits, dtype=torch.float32), edge_index=e, edge_weight=w)
                for e, w in zip(self._edge_indices, self._edge_weights)
            ]
            self._node_dim = self._num_qubits

        elif feature_type == WDEGREE_FEATURE:
            # use weighted degree as node features
            self._data_list = [
                Data(x=d, edge_index=e, edge_weight=w)
                for e, w, d in zip(self._edge_indices, self._edge_weights, self._vertices_deg)
            ]
            self._node_dim = 1
        else:
            raise ValueError(f'unknown graph node feature type {feature_type}')

    def _build_graphs(self, coupling_matrices):
        # read graph topology from coupling matrices
        graphs_edge_indices = [
            [
                [(i, j), (j, i)] for i, j in it.combinations(range(self._num_qubits), 2) if mat[i, j] != 0.0
            ] for mat in coupling_matrices
        ]

        # convert edge weights to tensors
        graphs_edge_weights = [
            torch.from_numpy(np.array([mat[i, j] for i, j in edges], dtype=np.float32).flatten())
            for edges, mat in zip(graphs_edge_indices, coupling_matrices)
        ]

        # convert to tensor in COO format for torch_geometric
        graphs_edge_indices = [
            torch.from_numpy(np.array([x for x in it.chain(*edge_indices)]).T)
            for edge_indices in graphs_edge_indices
        ]

        # compute weighted degree for each vertex
        graphs_vertices_degrees = [
            torch.FloatTensor([
                torch.sum((edge_indices[0] == q) * edge_weights) for q in range(self._num_qubits)
            ]).reshape(-1, 1)
            for edge_indices, edge_weights in zip(graphs_edge_indices, graphs_edge_weights)
        ]

        return graphs_edge_indices, graphs_edge_weights, graphs_vertices_degrees

    def __getitem__(self, item) -> (Data, int, torch.Tensor):
        return self._data_list[item], self._coupling_matrices_ids[item], self._coupling_matrices_tensor[item]

    def __len__(self):
        return len(self._data_list)

    @property
    def coupling_matrices_ids(self):
        return self._coupling_matrices_ids

    @property
    def coupling_matrices_dict(self):
        return self._coupling_matrices_dict

    @property
    def coupling_matrices_list(self):
        return self._coupling_matrices_tensor

    @property
    def data_list(self):
        return self._data_list

    @property
    def node_dim(self):
        return self._node_dim


def batch_generator(ds: CouplingGraphsDataset, batch_size) -> Batch:
    data_list = ds.data_list
    num_graphs = len(data_list)

    num_full_batches = num_graphs // batch_size
    batch_sizes = [batch_size] * num_full_batches

    if num_full_batches * batch_size != num_graphs:
        batch_sizes = batch_sizes + [num_graphs % batch_size]

    for i, bs in enumerate(batch_sizes):
        data_list_batch = data_list[(i * bs):((i + 1) * bs)]
        yield Batch.from_data_list(data_list=data_list_batch)
