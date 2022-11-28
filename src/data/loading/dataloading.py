import itertools as it
from operator import itemgetter
from typing import List

import numpy as np
from torch_geometric.data import Batch as GraphBatch

from src.data.loading import DatasetGCTransformer
from src.data.loading.utils import TgtBatch


def batch_generator_epoch_mixed(
        ds_list: List[DatasetGCTransformer], batch_size, pad_token, rng, shuffle=False
):
    """
    Data loader
    TODO: docstring
    """

    dataset_sizes = [len(ds) for ds in ds_list]

    # get a list (shuffled) batch indices for each dataset
    dataset_indices = [
        np.array_split(
            rng.choice(np.arange(sz), sz, replace=False) if shuffle else np.arange(sz),
            indices_or_sections=sz // batch_size
        ) for sz in dataset_sizes
    ]

    # add dataset id and flatten
    dataset_indices = [[(b, i) for b in idx_list] for i, idx_list in enumerate(dataset_indices)]
    dataset_indices = list(it.chain(*dataset_indices))

    # shuffle datasets
    if shuffle:
        rng.shuffle(dataset_indices)

    for indices, ds_id in dataset_indices:
        cgs = ds_list[ds_id].coupling_graphs_data
        mts = ds_list[ds_id].measurements
        hids = ds_list[ds_id].hamiltonian_ids

        graphs_batch = GraphBatch.from_data_list(data_list=list(itemgetter(*indices)(cgs)))
        tgt_batch = TgtBatch(mts[indices], pad=pad_token)
        hamiltonian_ids_batch = hids[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch


def batch_generator_epoch(
        ds: DatasetGCTransformer, batch_size, pad_token, rng, device, shuffle=False
):
    """
    Data loader
    TODO: docstring
    """
    coupling_graphs = ds.coupling_graphs_data
    measurements_tensor = ds.measurements
    hamiltonian_ids = ds.hamiltonian_ids

    num_samples = len(ds)

    num_full_batches = num_samples // batch_size
    batch_sizes = [batch_size] * num_full_batches

    if num_full_batches * batch_size != num_samples:
        batch_sizes = batch_sizes + [num_samples % batch_size]

    if shuffle:
        shuffled_indices = rng.choice(num_samples, num_samples, replace=False)
        coupling_graphs = [coupling_graphs[i] for i in shuffled_indices]
        hamiltonian_ids = [hamiltonian_ids[i] for i in shuffled_indices]
        measurements_tensor = measurements_tensor[shuffled_indices]

    for i, bs in enumerate(batch_sizes):
        graphs_batch = GraphBatch.from_data_list(data_list=coupling_graphs[(i * bs): ((i + 1) * bs)]).to(device)
        tgt_batch = TgtBatch(measurements_tensor[(i * bs):((i + 1) * bs)].to(device), pad=pad_token)
        hamiltonian_ids_batch = hamiltonian_ids[(i * bs):((i + 1) * bs)]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch


def batch_generator_iterations_mixed(
        ds_list: List[DatasetGCTransformer], batch_size, pad_token, rng, iterations
):
    """
    Data loader
    TODO: docstring
    """
    for step in range(1, iterations + 1):
        ds = rng.choice(ds_list, 1)[0]

        cgs = ds.coupling_graphs_data
        mts = ds.measurements
        hids = ds.hamiltonian_ids

        indices = ds.rng.choice(len(ds), batch_size, replace=True)
        graphs_batch = GraphBatch.from_data_list(data_list=[cgs[i] for i in indices])
        tgt_batch = TgtBatch(mts[indices], pad=pad_token)
        hamiltonian_ids_batch = hids[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch


def batch_generator_iterations(
        ds: DatasetGCTransformer, batch_size, pad_token, rng, iterations
):
    """
    Data loader
    TODO: docstring
    """
    cgs = ds.coupling_graphs_data
    mts = ds.measurements
    hids = ds.hamiltonian_ids

    for step in range(1, iterations + 1):
        indices = rng.choice(len(ds), batch_size, replace=True)
        graphs_batch = GraphBatch.from_data_list(data_list=[cgs[i] for i in indices])
        tgt_batch = TgtBatch(mts[indices], pad=pad_token)
        hamiltonian_ids_batch = hids[indices]

        yield tgt_batch, graphs_batch, hamiltonian_ids_batch


def _get_batch_sizes(b, n):
    num_full_batches = n // b
    batch_sizes = [b] * num_full_batches

    if num_full_batches * b != n:
        batch_sizes = batch_sizes + [n % b]

    return batch_sizes
