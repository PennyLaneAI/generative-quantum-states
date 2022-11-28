import numpy as np
from typing import List

from torch_geometric.data import Data as GraphData
import torch
from torch.utils.data import Dataset

from src.data.loading.dataset_coupling_graphs import CouplingGraphsDataset

__all__ = [
    'normalize_coupling_matrices',
    'DatasetGCTransformer'
]


def map_to_sentence(data, start_token, end_token, token_shift):
    num_data = data.shape[0]
    data = data + token_shift
    data = np.concatenate([start_token * np.ones(shape=(num_data, 1)), data], axis=1)

    if end_token > 0:
        data = np.concatenate([data, end_token * np.ones(shape=(num_data, 1))], axis=1)

    data = data.astype(int)

    return data


def normalize_coupling_matrices(coupling_matrices: np.ndarray) -> np.ndarray:
    """
    function normalizes coupling matrices and adds a batch dimension if a single matrix is given
    Args:
        coupling_matrices (np.ndarray): either of shape (nq, nq) or (nb, nq, nq)

    Returns (np.ndarray): normalized coupling matrices (on instance level)

    """
    # normalize coupling matrix
    if coupling_matrices.ndim == 2:
        coupling_matrices = np.expand_dims(coupling_matrices, axis=0)

    cmax = np.max(coupling_matrices, axis=(1, 2)).reshape((-1, 1, 1))
    cmin = np.min(coupling_matrices, axis=(1, 2)).reshape((-1, 1, 1))
    coupling_matrices = (coupling_matrices - cmin) / (cmax - cmin)

    return coupling_matrices


class DatasetGCTransformer(Dataset):
    """ Dataset with randomized pauli measurements prepared for the transformer network.

    The data is read from a *.npy file and has the shape N x Nq. Each row is assumed to be a measurement outcome
    of the  Nq-qubit Pauli-6 POVM with the assignments 0,1 <-> X; 2,3 <-> Y; and 4,5 <-> Z basis
    measurements.

    The tokens {0, 1, ..., 5} are then mapped to {3, ..., 8}. The <start> character is 1, the <end> character is 2, and
    the padding is 0.

    The resulting data is then in the shape N x (Nq + 2) where the first and last entry correspond to start and end
    character.

    TODO: update docstring
    """

    def __init__(self, measurements, coupling_matrices, hamiltonian_ids, rng: np.random.Generator, device,
                 nodes_feature_type, start_token, end_token, token_shift):
        self._coupling_matrices = coupling_matrices
        self._hamiltonian_ids = hamiltonian_ids
        self._device = device

        assert measurements.shape[0] == self._coupling_matrices.shape[0]
        assert hamiltonian_ids.shape[0] == self._coupling_matrices.shape[0]

        self._num_data, self._num_qubits = measurements.shape

        # random number generator
        self.rng = rng

        # adjust data
        measurements = map_to_sentence(
            measurements, start_token=start_token, end_token=end_token, token_shift=token_shift
        )

        self._measurements_numpy = measurements
        # self._measurements = torch.from_numpy(measurements).to(device)
        self._measurements = torch.from_numpy(measurements)

        # normalize coupling matrix
        coupling_matrices = normalize_coupling_matrices(coupling_matrices)

        # init coupling graphs dataset
        self._coupling_graphs_dataset = CouplingGraphsDataset(
            coupling_matrices=coupling_matrices, coupling_matrices_ids=self._hamiltonian_ids, rng=rng,
            device=device, feature_type=nodes_feature_type
        )

    def __getitem__(self, item):
        pass

    def __len__(self):
        return self._measurements_numpy.shape[0]

    @property
    def measurements(self) -> torch.Tensor:
        return self._measurements

    @property
    def measurements_numpy(self) -> np.ndarray:
        return self._measurements_numpy

    @property
    def hamiltonian_ids(self) -> np.ndarray:
        return self._hamiltonian_ids

    @property
    def coupling_matrices(self) -> np.ndarray:
        return self._coupling_matrices

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def coupling_graphs_data(self) -> List[GraphData]:
        return self._coupling_graphs_dataset.data_list

    @property
    def coupling_graphs_dataset(self) -> CouplingGraphsDataset:
        return self._coupling_graphs_dataset
