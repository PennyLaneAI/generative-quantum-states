import numpy as np
import torch
from torch_geometric.data import Batch as GraphBatch
from tqdm import tqdm
from typing import List

from src.data.loading import CouplingGraphsDataset
from src.data.loading.dataset_graph_conditional_transformer import \
    normalize_coupling_matrices
from src.models.gctransformer import GCTransformer


def generate_samples(
        transformer: GCTransformer,
        coupling_matrices: np.ndarray,
        coupling_matrices_ids: List[int],
        shots: int,
        device: torch.device,
        rng: np.random.Generator,
        graph_feature_type: str,
):
    # padding for start token
    coupling_matrices = normalize_coupling_matrices(coupling_matrices=coupling_matrices)

    coupling_graphs_dataset = CouplingGraphsDataset(
        coupling_matrices, coupling_matrices_ids, rng, device=device,
        feature_type=graph_feature_type
    )

    generated_samples = {}

    pbar = tqdm(coupling_graphs_dataset, total=len(coupling_graphs_dataset))
    qubits = coupling_matrices.shape[-1]

    for coupling_graph, cm_id, _ in pbar:
        pbar.set_description_str(f'model sampling for hamiltonian {cm_id}')

        coupling_graph_batch = GraphBatch.from_data_list(data_list=[coupling_graph]).to(
            device)

        with torch.no_grad():
            samples = transformer.sample(shots, coupling_graph_batch, qubits,
                                         print_progress=False)

        generated_samples[cm_id] = samples

    return generated_samples
