import argparse
import joblib
import json
import numpy as np
import os
from pennylane.utils import sparse_hamiltonian
from scipy.sparse import linalg as sparse_linalg
from tqdm import tqdm

from constants import DATA_DIR
from src.data.models.random_heisenberg.couplings import generate_coupling_matrix
from src.data.sample_shadow import sample_shadow
from src.data.utils import build_hamiltonian_from_couplings
from src.properties.correlation import compute_correlation_matrix_from_statevector
from src.properties.entropy import compute_entropies_from_statevector

parser = argparse.ArgumentParser()
parser.add_argument('--nh_train', default=80, type=int,
                    help='number hamiltonians to simulate')
parser.add_argument('--nh_test', default=20, type=int,
                    help='number hamiltonians to simulate')
parser.add_argument('--rows', default=2, type=int,
                    help='number of rows for the 2D lattice model')
parser.add_argument('--cols', default=10, type=int,
                    help='number of columns for the 2D lattice model')
parser.add_argument('--shots', default=1000, type=int,
                    help='number of samples to generate')
parser.add_argument('--threads', default=-1, type=int,
                    help='number of threads to use. -1 means all.')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--device', default='lightning.qubit', type=str,
                    help='Device to use for the quantum simulation. If using '
                         'lightning.gpu, need to run `pip install cuquantum '
                         'pennylane-lightning[gpu]` in advance',
                    choices=['default.qubit', 'lightning.gpu', 'lightning.qubit'])

args = parser.parse_args()

if args.threads <= 0:
    args.threads = joblib.cpu_count()


def generate_data():
    train_save_dir = os.path.join(DATA_DIR, '2d_heisenberg', f'{args.rows}x{args.cols}',
                                  'train')
    test_save_dir = os.path.join(DATA_DIR, '2d_heisenberg', f'{args.rows}x{args.cols}',
                                 'test')

    os.makedirs(train_save_dir)
    os.makedirs(test_save_dir)

    # random number generator
    rng = np.random.default_rng(seed=args.seed)

    with open(os.path.join(train_save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
        print(f'* saved args as {f.name}')

    with open(os.path.join(test_save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
        print(f'* saved args as {f.name}')

    for nh, save_dir in zip([args.nh_train, args.nh_test],
                            [train_save_dir, test_save_dir]):
        pbar = tqdm(range(nh), total=nh)
        for hid in pbar:
            # sample coupling matrix
            pbar.set_description(desc=f'generating coupling matrix...')
            coupling_matrix = generate_coupling_matrix(rows=args.rows, cols=args.cols,
                                                       rng=rng)

            # setup sparse Hamiltonian
            pbar.set_description(desc=f'setting up sparse Hamiltonian...')
            hamiltonian = build_hamiltonian_from_couplings(coupling_matrix)
            hamiltonian_sparse = sparse_hamiltonian(hamiltonian)

            # compute ground state using exact diagonalization
            pbar.set_description(desc=f'exact diagonalization...')
            eigvals, eigvecs = sparse_linalg.eigs(hamiltonian_sparse, which='SR', k=1)
            eigvals = eigvals.real
            ground_state = eigvecs[:, np.argmin(eigvals)]
            ground_state = ground_state / np.linalg.norm(ground_state)

            # sample from ground state
            wires = args.rows * args.cols
            pbar.set_description(desc=f'sampling {args.shots} shadows...')
            measurement_data = sample_shadow(ground_state, wires=wires,
                                             shots=args.shots, device_name=args.device)

            # compute correlation matrix
            if wires <= 20:
                pbar.set_description(desc=f'computing correlations...')
                correlations = compute_correlation_matrix_from_statevector(
                    ground_state, wires, device_name='default.qubit'
                )
            else:
                correlations = np.nan

            # compute entanglement entropies
            if wires <= 20:
                pbar.set_description(desc=f'computing entropies...')
                entropies = compute_entropies_from_statevector(ground_state,
                                                               wires=wires)
            else:
                entropies = np.nan

            # ground state energy
            ground_state_energy = np.min(eigvals)

            # save as numpy arrays
            np.save(os.path.join(save_dir, f'data_id{hid}.npy'), measurement_data)
            np.save(os.path.join(save_dir, f'coupling_matrix_id{hid}.npy'),
                    coupling_matrix)
            np.save(os.path.join(save_dir, f'correlation_matrix_id{hid}.npy'),
                    correlations)
            np.save(os.path.join(save_dir, f'entanglement_entropies_id{hid}.npy'),
                    entropies)

            # save as text
            with open(os.path.join(save_dir, f'ground_state_energy_id{hid}.txt'),
                      'w') as f:
                f.write(str(ground_state_energy))


if __name__ == '__main__':
    generate_data()
