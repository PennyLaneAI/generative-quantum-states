import itertools as it
import json
import numpy as np
import os
import torch
from torch_geometric.data import Batch, Data

from constants import ROOT_DIR, TF_ARCHS, GCN_ARCHS, NUM_MMT_OUTCOMES
from src.data.loading.dataset_graph_conditional_transformer import \
    normalize_coupling_matrices
from src.models.gctransformer import init_gctransformer, get_sample_structure
from src.models.graph_encoder import get_graph_encoder

_MODEL_DIR = os.path.join(ROOT_DIR, 'heisenberg_tutorial', 'model_weights', '4x4')
_QUBITS = 16


def initialize():
    # get and parse model args
    with open(os.path.join(_MODEL_DIR, 'args.json'), 'r') as f:
        model_args = json.load(f)

    # get and parse hyperparams
    with open(os.path.join(_MODEL_DIR, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    # structure of mmt sequence
    sample_struct = get_sample_structure(version=model_args['sample_structure'])

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model architectures
    tf_arch = model_args['tf_arch']
    gcn_arch = model_args['gcn_arch']

    # initialize graph encoder
    graph_encoder = get_graph_encoder(
        arch=gcn_arch,
        in_node_dim=_QUBITS,
        gcn_dim=GCN_ARCHS[gcn_arch]['gcn_dim'],
        gcn_layers=GCN_ARCHS[gcn_arch]['gcn_layers'],
        d_model=TF_ARCHS[tf_arch]['d_model'],
        qubits=_QUBITS,
    )

    # initialize transformer
    transformer = init_gctransformer(
        n_outcomes=NUM_MMT_OUTCOMES,
        encoder=graph_encoder,
        n_layers=TF_ARCHS[tf_arch]['n_layers'],
        d_model=TF_ARCHS[tf_arch]['d_model'],
        d_ff=4 * TF_ARCHS[tf_arch]['d_model'],
        n_heads=TF_ARCHS[tf_arch]['n_head'],
        dropout=hparams['dropout'],
        pad_token=sample_struct.pad_token,
        start_token=sample_struct.start_token,
        end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift
    )

    # load weights
    ckpt_fp = os.path.join(_MODEL_DIR, f'checkpoint_final.pth.tar')
    ckpt = torch.load(ckpt_fp, map_location=device)

    transformer.load_state_dict(ckpt['model_state_dict'], strict=True)
    transformer.to(device)

    return transformer


def preprocess_coupling_matrix(coupling_matrix: np.ndarray) -> Batch:
    """ convert coupling matrix to torch_geometric.data object """
    assert coupling_matrix.ndim == 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    qubits = coupling_matrix.shape[0]
    coup_mat_padded = normalize_coupling_matrices(coupling_matrix)

    # convert weights to torch tensor
    edge_indices = [[(i, j), (j, i)] for i, j in it.combinations(range(qubits), 2) if
                    coupling_matrix[i, j] != 0.0]
    edge_weights = torch.from_numpy(
        np.array([coup_mat_padded[0, i, j] for i, j in edge_indices],
                 dtype=np.float32).flatten()
    ).to(device)

    # flatten edge indices and convert to torch tensor
    edge_indices = torch.from_numpy(
        np.array([x for x in it.chain(*edge_indices)]).T).to(device)

    # compute weighted node degree
    vertices_degrees = torch.FloatTensor(
        [torch.sum((edge_indices[0] == q) * edge_weights) for q in range(qubits)]
    ).reshape(-1, 1).to(device)

    # convert to torch_geometric Data object
    data = Data(torch.eye(
        qubits, dtype=torch.float32), edge_indices, edge_weight=edge_weights
    )

    # convert to Graph batch
    data_batch = Batch.from_data_list(data_list=[data]).to(device)

    return data_batch
