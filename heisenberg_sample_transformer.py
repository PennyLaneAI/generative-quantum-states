"""
script to sample from conditional transformers for Heisenberg data
"""

import argparse
import json
import numpy as np
import os
import torch

from constants import *
from src.models.gctransformer.generate_samples import generate_samples
from src.models.gctransformer import init_gctransformer, get_sample_structure
from src.models.graph_encoder import get_graph_encoder
from src.utils import filter_folder_for_files

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default=None, required=True)
parser.add_argument('--snapshots', type=int, default=20000,
                    help='number of generated samples for evaluation')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()


def main(results_dir, seed, snapshots):
    model_args, hparams = load_model_args(results_dir)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # lattice size
    rows, cols = tuple(map(int, str(model_args['train_size']).split('x')))
    qubits = rows * cols

    # intialize model
    ckpt_fp = os.path.join(results_dir, 'checkpoints', f'checkpoint_final.pth.tar')
    transformer = load_model(
        ckpt_fp,
        sample_structure=model_args['sample_structure'],
        qubits=qubits,
        tf_arch=model_args['tf_arch'],
        gcn_arch=model_args['gcn_arch'],
        gcn_features=model_args['gcn_features'],
        dropout=hparams['dropout'],
        device=device
    )

    gcn_features = model_args['gcn_features']
    train_size = model_args['train_size']
    data_root = model_args['data_root']

    # generator for random numbers
    rng = np.random.default_rng(seed=seed)

    run_sampling(transformer, results_dir, data_root, train_size, snapshots, rng,
                 gcn_features, device)


def load_model_args(results_dir):
    # get and parse model args
    with open(os.path.join(results_dir, 'args.json'), 'r') as f:
        model_args = json.load(f)

    # get and parse hyperparams
    with open(os.path.join(results_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    return model_args, hparams


def load_model(checkpoint_fp, sample_structure, qubits, tf_arch, gcn_arch, gcn_features,
               dropout, device):
    # initialize graph encoder
    encoder = get_graph_encoder(
        gcn_arch,
        in_node_dim=1 if gcn_features == WDEGREE_FEATURE else qubits,
        gcn_dim=GCN_ARCHS[gcn_arch]['gcn_dim'],
        gcn_layers=GCN_ARCHS[gcn_arch]['gcn_layers'],
        d_model=TF_ARCHS[tf_arch]['d_model'],
        qubits=qubits
    )

    # structure of mmt sequence
    sample_struct = get_sample_structure(version=sample_structure)

    # initialize transformer
    transformer = init_gctransformer(
        n_outcomes=NUM_MMT_OUTCOMES,
        encoder=encoder,
        n_layers=TF_ARCHS[tf_arch]['n_layers'],
        d_model=TF_ARCHS[tf_arch]['d_model'],
        d_ff=4 * TF_ARCHS[tf_arch]['d_model'],
        n_heads=TF_ARCHS[tf_arch]['n_head'],
        dropout=dropout,
        pad_token=sample_struct.pad_token,
        start_token=sample_struct.start_token,
        end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift
    )

    # load weights
    ckpt = torch.load(checkpoint_fp, map_location=device)

    transformer.load_state_dict(ckpt['model_state_dict'], strict=True)
    transformer.to(device)
    transformer.eval()
    print(f'loaded weights from {checkpoint_fp}')

    return transformer


def run_sampling(transformer, results_dir, data_dir, train_size, snapshots, rng,
                 gcn_features, device):
    # generate samples for test hamiltonians
    test_data_dir = os.path.join(data_dir, train_size, 'test')
    couplings, couplings_ids = load_couplings(test_data_dir)
    samples_test = generate_samples(transformer, couplings, couplings_ids, snapshots,
                                    device, rng, gcn_features)

    # save samples
    test_save_dir = os.path.join(results_dir, 'samples', 'test', 'model')

    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    for k, s in samples_test.items():
        np.save(os.path.join(test_save_dir, f'model_samples_{snapshots}_id{k}.npy'), s)

    # generate samples for train hamiltonians
    train_data_dir = os.path.join(data_dir, train_size, 'train')
    couplings, couplings_ids = load_couplings(train_data_dir)
    samples_train = generate_samples(transformer, couplings, couplings_ids, snapshots,
                                     device, rng, gcn_features)

    # save samples
    train_save_dir = os.path.join(results_dir, 'samples', 'train', 'model')

    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)

    for k, s in samples_train.items():
        np.save(os.path.join(train_save_dir, f'model_samples_{snapshots}_id{k}.npy'), s)

    return samples_train, samples_test


def load_couplings(data_dir):
    # get data filepaths
    coupling_mats_fps = filter_folder_for_files(data_dir,
                                                file_pattern=f'coupling_matrix_id*.npy')
    coupling_mats_ids = [int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in
                         coupling_mats_fps]

    coupling_matrices_array = []
    coupling_matrices_ids_list = []

    for cid, cfp in zip(coupling_mats_ids, coupling_mats_fps):
        # load coupling matrices
        coup_mat = np.load(os.path.join(data_dir, f'coupling_matrix_id{cid}.npy'))
        coupling_matrices_array.append(coup_mat)
        coupling_matrices_ids_list.append(cid)

    coupling_matrices_array = np.stack(coupling_matrices_array, axis=0)

    return coupling_matrices_array, coupling_matrices_ids_list


if __name__ == '__main__':
    main(results_dir=args.results_dir, seed=args.seed, snapshots=args.snapshots)
