"""
script to train conditional transformers for Heisenberg data
"""
import argparse
from datetime import datetime as dt
import json
import numpy as np
from operator import itemgetter
import os
import shutil
import sys
import torch

from constants import *
from src.data.loading import DatasetGCTransformer
from src.training import GCTransformerTrainer
from src.models.graph_encoder import get_graph_encoder
from src.models.gctransformer import init_gctransformer, get_sample_structure
from src.utils import timestamp, filter_folder_for_files
from src.training.utils import dir_setup, Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--data-root', type=str,
                        default='./data/2d_heisenberg/')
    parser.add_argument('--train-size', type=str, default=None)
    parser.add_argument('--tf-arch', type=str, default='transformer_l4_d128_h4')
    parser.add_argument('--gcn-arch', type=str, default='gcn_proj_3_16')
    parser.add_argument('--gcn-features', type=str, default=ONE_HOT_FEATURE,
                        choices=[WDEGREE_FEATURE, ONE_HOT_FEATURE])
    parser.add_argument('--hamiltonians', type=int, default=None,
                        help='number of training hamiltonians; set to None to use all '
                             'hamiltonians in the train split')
    parser.add_argument('--train-samples', type=int, default=1000,
                        help='number of train samples per hamiltonian')
    parser.add_argument('--iterations', type=int, default=100,
                        help='number of epochs if epoch_mode = 1, else number of steps')
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--eval-test', type=int, default=1, choices=[0, 1])
    parser.add_argument('--k', type=int, default=1,
                        help='number of buckets for median of means estimation')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--epoch_mode', type=int, default=1, choices=[0, 1])
    parser.add_argument('--sample-structure', type=int, default=2, choices=[0, 1, 2])
    args = parser.parse_args()
    return args


def get_hyperparams(**kwargs):
    """ this is the default set of hyperparams """
    hparams = argparse.Namespace(
        lr=1e-3,
        wd=0.0,
        bs=100,
        dropout=0.1,
        final_lr=1e-7,
        lr_scheduler=WARMUP_COSINE_SCHEDULER,
        warmup_frac=0.05,
    )

    for k, v in kwargs.items():
        setattr(hparams, k, v)

    return hparams


def train_transformer(args, hparams):
    if args.debug:
        args.data_root = './data/conditional_heisenberg/'
        args.results_dir = './results-debug-local'
        args.train_samples = 20
        args.iterations = 2
        hparams.bs = 20

    state_type = 'conditional_heisenberg'

    # convert strings to integers
    rows, cols = tuple(map(int, args.train_size.split('x')))

    # setup results dir structure
    model_id = f'{args.gcn_arch}-{args.tf_arch}_feat{args.gcn_features}'

    # train id based on hyperparams
    train_id = f'iter{args.iterations}_lr{hparams.lr}_wd{hparams.wd}_bs{hparams.bs}_dropout{hparams.dropout}'
    train_id = train_id + f'_samplestruct{args.sample_structure}_lrschedule{hparams.lr_scheduler}'
    train_id = train_id + dt.now().strftime('%d%m%Y-%H%M%S')

    # dataset id
    dataset_id = f'{state_type}_' + f'{rows}x{cols}'

    results_dir, _, train_id = dir_setup(results_root=args.results_dir,
                                         dataset_id=dataset_id,
                                         model_id=model_id,
                                         num_train=f'ns{args.train_samples}',
                                         train_id=train_id,
                                         verbose=args.verbose)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'train_out.txt'))

    # save args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # save hparams
    with open(os.path.join(results_dir, 'hparams.json'), 'w') as f:
        json.dump(vars(hparams), f)

    if args.verbose:
        print_dict(vars(args))
        print_dict(vars(hparams))

    # generator for random numbers
    rng = np.random.default_rng(seed=args.seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # structure of mmt sequence
    sample_struct = get_sample_structure(version=args.sample_structure)

    d_model = TF_ARCHS[args.tf_arch]['d_model']
    n_head = TF_ARCHS[args.tf_arch]['n_head']
    n_layers = TF_ARCHS[args.tf_arch]['n_layers']

    gcn_dim = GCN_ARCHS[args.gcn_arch]['gcn_dim']
    gcn_layers = GCN_ARCHS[args.gcn_arch]['gcn_layers']

    assert d_model % n_head == 0, 'd_model must be integer multiple of n_head!'

    # initialize graph encoder
    qubits = rows * cols
    in_node_dim = 1 if args.gcn_features == WDEGREE_FEATURE else qubits
    encoder = get_graph_encoder(
        args.gcn_arch, in_node_dim=in_node_dim, gcn_dim=gcn_dim, gcn_layers=gcn_layers,
        d_model=d_model, qubits=qubits
    )

    # initialize transformer
    transformer = init_gctransformer(
        n_outcomes=NUM_MMT_OUTCOMES,
        encoder=encoder,
        n_layers=n_layers,
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=n_head,
        dropout=hparams.dropout,
        pad_token=sample_struct.pad_token,
        start_token=sample_struct.start_token,
        end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift
    )

    transformer.to(device)

    # train data
    train_data_dir = os.path.join(args.data_root, f'{args.train_size}', 'train')
    train_measurements, train_coupling_matrices, train_hamiltonians_ids = load_data(
        data_dir=train_data_dir, num_samples=args.train_samples,
        num_hamiltonians=args.hamiltonians, rng=rng
    )
    train_hamiltonians_ids_unique = list(map(int, np.unique(train_hamiltonians_ids)))
    train_dataset = DatasetGCTransformer(
        measurements=train_measurements, coupling_matrices=train_coupling_matrices,
        hamiltonian_ids=train_hamiltonians_ids, rng=rng, device=device,
        nodes_feature_type=args.gcn_features,
        start_token=sample_struct.start_token, end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift
    )

    # save data to results dir
    results_train_data_dir = os.path.join(results_dir, 'data', f'{args.train_size}',
                                          'train')
    save_data(results_train_data_dir, train_data_dir, train_hamiltonians_ids_unique)

    # test data
    test_data_dir = os.path.join(args.data_root, f'{args.train_size}', 'test')
    if args.eval_test:
        test_measurements, test_coupling_matrices, test_hamiltonians_ids = load_data(
            data_dir=test_data_dir, num_samples=args.train_samples,
            num_hamiltonians=None, rng=rng
        )
        test_dataset = DatasetGCTransformer(
            measurements=test_measurements, coupling_matrices=test_coupling_matrices,
            hamiltonian_ids=test_hamiltonians_ids, rng=rng, device=device,
            nodes_feature_type=args.gcn_features,
            start_token=sample_struct.start_token, end_token=sample_struct.end_token,
            token_shift=sample_struct.token_shift
        )
    else:
        test_dataset = None

    # save data to results dir
    results_test_data_dir = os.path.join(results_dir, 'data', f'{args.train_size}',
                                         'test')
    save_data(results_test_data_dir, test_data_dir, None)

    # save train ids
    with open(os.path.join(results_dir, 'train_hamiltonian_ids.json'), 'w') as f:
        json.dump(train_hamiltonians_ids_unique, f)

    print(f'* Training Hamiltonian Ids: {train_hamiltonians_ids_unique}')

    print(f'[{timestamp()}] start training, saving results to {results_dir}')
    trainer = GCTransformerTrainer(model=transformer,
                                   train_dataset=train_dataset,
                                   val_dataset=None,
                                   test_dataset=test_dataset,
                                   save_dir=results_dir,
                                   iterations=args.iterations,
                                   lr=hparams.lr,
                                   final_lr=hparams.final_lr,
                                   lr_scheduler=hparams.lr_scheduler,
                                   warmup_frac=hparams.warmup_frac,
                                   weight_decay=hparams.wd,
                                   device=device,
                                   batch_size=hparams.bs,
                                   rng=rng,
                                   eval_every=args.eval_every,
                                   epoch_mode=args.epoch_mode)

    train_total_loss, val_total_loss, test_total_loss = trainer.train()

    pstr = f'[{timestamp()}] training end, train total-loss: {train_total_loss:.4f}'
    pstr = pstr + f', test total-loss: {test_total_loss:.4f}, val total-loss: {val_total_loss:.4f}'
    print(pstr)

    trainer.save_model('final', is_best=False)


def print_dict(d, tag=None):
    """ helper function to print args """
    print(f'--------{tag or ""}----------')
    for k, v in d.items():
        print('{0:27}: {1}'.format(k, v))
    print(f'--------{tag or ""}----------\n')


def save_data(results_data_dir, data_dir, hamiltonian_ids):
    os.makedirs(results_data_dir)

    for fn in os.listdir(data_dir):
        _, ext = os.path.splitext(fn)

        if ext == '.json':
            shutil.copy(os.path.join(data_dir, fn), os.path.join(results_data_dir, fn))
            continue

        if ext not in ['.txt', '.npy']:
            continue

        hid = int(fn[(fn.find('id') + 2):fn.find(ext)])
        if hamiltonian_ids is None or hid in hamiltonian_ids:
            shutil.copy(os.path.join(data_dir, fn), os.path.join(results_data_dir, fn))


def load_data(data_dir, num_samples, num_hamiltonians, rng: np.random.Generator):
    """
    docstring goes here
    """
    # get data filepaths
    coupling_mats_fps = filter_folder_for_files(data_dir,
                                                file_pattern=f'coupling_matrix_id*.npy')
    coupling_mats_ids = [int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in
                         coupling_mats_fps]

    if num_hamiltonians is not None:
        # subsample num_hamiltoninas at random
        indices = rng.choice(range(len(coupling_mats_ids)), size=num_hamiltonians,
                             replace=False)
        if len(indices) > 1:
            coupling_mats_fps = list(itemgetter(*indices)(coupling_mats_fps))
            coupling_mats_ids = list(itemgetter(*indices)(coupling_mats_ids))
        else:
            coupling_mats_fps = [coupling_mats_fps[indices[0]]]
            coupling_mats_ids = [coupling_mats_ids[indices[0]]]

    measurements = []
    coupling_matrices = []
    hamiltonian_ids = []

    for cid, cfp in zip(coupling_mats_ids, coupling_mats_fps):
        ham_mmts = np.load(os.path.join(data_dir, f'data_id{cid}.npy'))
        num_measurements = ham_mmts.shape[0]
        ham_mmts = ham_mmts[rng.choice(num_measurements, num_samples, replace=False)]
        measurements.append(ham_mmts)

        coup_mat = np.load(os.path.join(data_dir, f'coupling_matrix_id{cid}.npy'))
        coup_mat = np.tile(coup_mat, reps=[num_samples, 1, 1])

        coupling_matrices.append(coup_mat)

        hamiltonian_ids.append(cid * np.ones(shape=num_samples))

    measurements = np.concatenate(measurements, axis=0)
    coupling_matrices = np.concatenate(coupling_matrices, axis=0)
    hamiltonian_ids = np.concatenate(hamiltonian_ids, axis=0)

    return measurements, coupling_matrices, hamiltonian_ids


if __name__ == '__main__':
    train_transformer(args=parse_args(), hparams=get_hyperparams())
