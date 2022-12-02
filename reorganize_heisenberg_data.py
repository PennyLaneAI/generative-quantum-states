import argparse
import itertools as it
import json
import numpy as np
import os
import shutil
from tqdm import tqdm

from constants import DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument('heisenberg_data', type=str)
args = parser.parse_args()

# number of columns in the lattice
N_COLS = 5

# number of rows in the lattice
LATTICE_ROW_NUMBERS = [4, 5, 6, 7, 8, 9]

# names of the files containing the data
DATA_FILENAMES = [
    'correlation_matrix_id{}.npy',
    'coupling_matrix_id{}.npy',
    'data_id{}.npy',
    'ground_state_energy_id{}.npy'
]

TARGET_FOLDER = os.path.join(DATA_DIR, '2d_heisenberg')


def read_couplings(length, idx, coup_mat_fp):
    """ Read couplings from file. """
    with open(coup_mat_fp.format(length, idx), 'r') as f:
        couplings = []
        for line in f:
            for i, c in enumerate(line.split("\t")):
                v = float(c)
                couplings.append(v)
    return couplings


def load_coupling_matrix(length, idx, coup_mat_fp):
    """ load coupling matrix from file. """
    # read coupling matrix
    couplings = read_couplings(length, idx, coup_mat_fp=coup_mat_fp)
    edges = [
        (si, sj) for (si, sj) in it.combinations(range(length * N_COLS), 2)
        if ((sj % N_COLS > 0) and sj - si == 1) or sj - si == N_COLS
    ]
    mat = np.zeros(shape=(length * N_COLS, length * N_COLS))

    for (i, j), v in zip(edges, couplings):
        mat[i, j] = v

    mat += mat.T
    return mat


def load_correlation_matrix(length, idx, xx_corrs_fp, yy_corrs_fp, zz_corrs_fp):
    """ load correlation matrix from xx, yy and zz files. """
    corr_mat = np.zeros((length * N_COLS) ** 2)
    for fp in [xx_corrs_fp, yy_corrs_fp, zz_corrs_fp]:
        with open(fp.format(length, idx), 'r') as f:
            i = 0
            for line in f:
                for c in line.split("\t"):
                    v = float(c)
                    corr_mat[i] += v
                    i += 1

    # reshape and normalze correlation matrix
    corr_mat = np.array(corr_mat).reshape(length * N_COLS, length * N_COLS) / 3
    return corr_mat


def load_data(length, idx, samples_fp):
    """ load measurements from file. """
    with open(samples_fp.format(length, idx), 'r') as f:
        measurements = [[int(c) for i, c in enumerate(line.split("\t"))] for line in f]
    measurements = np.array(measurements)
    return measurements


def load_ground_state_energy(length, idx, energy_fp):
    """ load ground state energy. """
    with open(energy_fp.format(length, idx), 'r') as f:
        energy = float(f.readline())
    return energy


def process(
        lng, data_dir, indices, desc, xx_corrs_fp, yy_corrs_fp, zz_corrs_fp, samples_fp,
        energy_fp, coup_mat_fp
):
    pbar = tqdm(indices, total=len(indices), desc=desc)
    i = 0
    for idx in pbar:
        try:
            corr_mat = load_correlation_matrix(
                lng, idx, xx_corrs_fp=xx_corrs_fp, yy_corrs_fp=yy_corrs_fp,
                zz_corrs_fp=zz_corrs_fp
            )
            coup_mat = load_coupling_matrix(lng, idx, coup_mat_fp=coup_mat_fp)
            energy = load_ground_state_energy(lng, idx, energy_fp=energy_fp)
            data = load_data(lng, idx, samples_fp=samples_fp)
        except FileNotFoundError:
            # skip if at least one file is missing
            continue

        np.save(os.path.join(data_dir, f'correlation_matrix_id{i}.npy'), corr_mat)
        np.save(os.path.join(data_dir, f'coupling_matrix_id{i}.npy'), coup_mat)
        np.save(os.path.join(data_dir, f'data_id{i}.npy'), data)
        np.save(os.path.join(data_dir, f'ground_state_energy_id{i}.npy'), energy)
        i += 1

    return i


def main():
    samples_fp = os.path.join(args.heisenberg_data, 'heisenberg_{}x5_id{}_samples.txt')
    xx_corrs_fp = os.path.join(args.heisenberg_data, 'heisenberg_{}x5_id{}_XX.txt')
    yy_corrs_fp = os.path.join(args.heisenberg_data, 'heisenberg_{}x5_id{}_YY.txt')
    zz_corrs_fp = os.path.join(args.heisenberg_data, 'heisenberg_{}x5_id{}_ZZ.txt')
    cp_mat_fp = os.path.join(args.heisenberg_data, 'heisenberg_{}x5_id{}_couplings.txt')
    energy_fp = os.path.join(args.heisenberg_data, 'heisenberg_{}x5_id{}_E.txt')

    indices = np.arange(1, 101)

    for lng in LATTICE_ROW_NUMBERS:
        # process train
        data_dir = os.path.join(
            TARGET_FOLDER,
            f'{lng}x{N_COLS}',
            'data'
        )
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        n = process(
            lng, data_dir, indices,
            desc=f'processing {lng}x{N_COLS}',
            xx_corrs_fp=xx_corrs_fp,
            yy_corrs_fp=yy_corrs_fp,
            zz_corrs_fp=zz_corrs_fp,
            samples_fp=samples_fp,
            energy_fp=energy_fp,
            coup_mat_fp=cp_mat_fp
        )

        n_train = int(0.8 * n)
        n_test = n - n_train

        # train split
        train_dir = os.path.join(TARGET_FOLDER, f'{lng}x{N_COLS}', 'train')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        for i in np.arange(n_train):
            for fn_tag in DATA_FILENAMES:
                shutil.move(os.path.join(data_dir, fn_tag.format(i)),
                            os.path.join(train_dir, fn_tag.format(i)))

        with open(os.path.join(train_dir, 'args.json'), 'w') as f:
            json.dump({'n_rows': lng, 'n_cols': N_COLS}, f)

        print('saved data to {}'.format(train_dir))

        # test split
        test_dir = os.path.join(TARGET_FOLDER, f'{lng}x{N_COLS}', 'test')

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for i in np.arange(n_test):
            for fn_tag in DATA_FILENAMES:
                shutil.move(
                    os.path.join(data_dir, fn_tag.format(i + n_train)),
                    os.path.join(test_dir, fn_tag.format(i))
                )

        with open(os.path.join(test_dir, 'args.json'), 'w') as f:
            json.dump({'n_rows': lng, 'n_cols': N_COLS}, f)

        print('saved data to {}'.format(test_dir))

        os.rmdir(data_dir)


if __name__ == '__main__':
    main()
