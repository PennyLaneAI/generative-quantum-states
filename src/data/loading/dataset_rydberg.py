from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.eval.eval_rydberg import est_order_param_1D, est_order_param_2D, est_order_param_1D_from_measurements, \
    est_order_param_1D_fourier, est_order_param_1D_fourier_from_measurements, est_order_param_2D_from_measurements

min_Omega = 2 * np.pi * 4 - 0.1


def int2bitarray(xs, length):
    vals = np.empty(shape=(len(xs), length), dtype=np.bool)
    for i, x in enumerate(xs):
        vals[i, :] = np.flip(np.array(list(np.binary_repr(x, length)), dtype=np.uint8))
    return vals


def read_single_data_file(fname: str, variables: list, metrics: list, n_qubits: int, extra_vars: list = None,
                          ):
    result = dict(np.load(fname))
    # mask = result['Omega'] >= min_Omega
    length = len(result['Omega'])

    key_list = []
    for V in variables:
        v = result[V]
        if len(v.shape) == 0:
            v = np.repeat(v, length)
        key_list.append(v)
    # key_list = np.stack([result[v] for v in variables]).T
    key_list = np.stack(key_list).T
    dataset = {}
    for i, key in enumerate(key_list):
        # if not mask[i]: continue  # skip if omega <= 0
        if extra_vars is not None:
            key = extra_vars + list(key)
        key = tuple(key)
        assert key not in dataset
        dataset[key] = {
            'measurements': int2bitarray(result['measurements'][i], n_qubits),
        }
        for m in metrics:
            dataset[key][m] = result[m][i]
    return dataset


def read_data_files(folder: str, var_name: str, variables: list, metrics: list, nx: int, ny: int = 1,
                    n_threads: int = 20, ):
    dataset = {}
    fnames = glob(folder + f"/{var_name}-*.npz")
    values = [float(fname.split("-")[-1].rstrip('.npz')) for fname in fnames]
    N = len(fnames)
    common_kwargs = dict(variables=variables, metrics=metrics, n_qubits=nx * ny)
    all_data = Parallel(n_jobs=n_threads)(
        delayed(read_single_data_file)(fnames[i], extra_vars=[nx, ny, values[i]], **common_kwargs) for i in
        range(N))
    for data in all_data:
        dataset.update(data)
    return dataset


def unif_sample_on_grid(df, x_bins=8, y_bins=7, x_range=None, y_range=None, X='detuning', Y='interaction_range'):
    if x_range is None: x_range = df[X].min(), df[X].max() + 0.1
    if y_range is None: y_range = df[Y].min(), df[Y].max() + 0.1
    x_grid = np.linspace(*x_range, x_bins + 1)
    y_grid = np.linspace(*y_range, y_bins + 1)
    # print(x_grid.shape,y_grid.shape)
    xs = df[X].values
    ys = df[Y].values
    idxes = []
    for i in range(x_bins):
        for j in range(y_bins):
            x_min, x_max = x_grid[i], x_grid[i + 1]
            y_min, y_max = y_grid[j], y_grid[j + 1]
            valid_idxes = df.index[(xs >= x_min) * (xs < x_max) * (ys >= y_min) * (ys < y_max)]
            if len(valid_idxes) > 0:
                idx = np.random.choice(valid_idxes)
                idxes.append(idx)
    idxes = np.array(idxes)
    return idxes, df.loc[idxes]


class RydbergDataset(object):
    # variables = ['Omega', 'Delta', "detuning", 'sweep_rate', 'time', 'rel_time']
    variables = ['Omega', 'Delta', 'time', 'sweep_rate']
    # metrics = ['density_z', 'density_x', 'corr_z', 'corr_x', 'corr_y', 'Z2', 'Z3']
    metrics = ['density_z', ]
    order_params_1D = ['Z2', 'Z3', 'Z4']
    order_params_2D = ['Checkboard', 'Striated', 'Staggered']
    C6 = 2 * np.pi * 862690

    def __init__(self, nx: int, dim: int, folder: str, n_threads=20,
                 var_name='interaction_range',
                 variables: list = None,
                 ny: int = 1,
                 ):
        self.metrics = RydbergDataset.metrics
        self.dataset = {}
        self.dim = dim  # dimension of the lattice
        self.nx = nx  # number of qubits in x direction
        self.ny = ny  # number of qubits in y direction
        if dim == 1:
            assert ny == 1
        assert dim in [1, 2]

        self.dataset = read_data_files(folder, var_name=var_name,
                                       variables=variables if variables is not None else self.variables,
                                       metrics=self.metrics, nx=nx, ny=ny, n_threads=n_threads, )
        # Variables for each system
        self.variables = ['nx', 'ny', var_name]
        if variables is None:
            self.variables = self.variables + RydbergDataset.variables
        else:
            self.variables = self.variables + variables
        self.keys = np.stack(self.dataset.keys())
        self.info = self.gather_info()
        super().__init__()

    def gather_info(self):
        df = pd.DataFrame(self.keys, columns=self.variables)

        if 'sweep_rate' in df:
            df['inv_sweep_rate'] = 1. / df['sweep_rate'].values
        if not ('detuning' in df):
            df['detuning'] = df['Delta'] / df['Omega']
        # Rb = (RydbergDataset.C6 / df['Omega']) ** (1 / 6)
        # df['interaction_range'] = Rb / df['distance']

        # if self.dim == 1:
        #     for order_param in RydbergDataset.order_params:
        #         df[order_param] = np.array([self.dataset[key][order_param] for key in self.dataset])
        #     df['phase'] = determine_phase_1D(df['Z2'], df['Z3'])
        return df

    def est_order_params(self, order_params: list = None, est_from_measurements: bool = True):
        if order_params is None:
            order_params = RydbergDataset.order_params_1D if self.dim == 1 else RydbergDataset.order_params_2D
        if est_from_measurements:
            f_order_param = est_order_param_1D_from_measurements if self.dim == 1 else est_order_param_2D_from_measurements
        else:
            f_order_param = est_order_param_1D if self.dim == 1 else est_order_param_2D
        base_kwargs = {'nx': self.nx, 'ny': self.ny}
        for order_param in order_params:
            vals = np.full(len(self), np.nan)
            for idx in range(len(self)):
                if est_from_measurements:
                    measurements = self[idx]['measurements']
                    vals[idx] = f_order_param(measurements, order_param=order_param, **base_kwargs)
                else:
                    density = self[idx]['density_z']
                    vals[idx] = f_order_param(density=density, order_param=order_param, **base_kwargs)
            self.info[order_param] = vals
        return self.info

    def prepare_train_set(self, keys, n_measurements=1000):
        train_set = {}
        for key in keys:
            key = tuple(key)
            measurements = self.dataset[key]['measurements']
            if n_measurements <= 0:
                train_set[key] = measurements
            else:
                replace = n_measurements > len(measurements)
                idxes = np.random.choice(len(measurements), size=n_measurements, replace=replace)
                train_set[key] = measurements[idxes]
        return train_set

    def get_key(self, idx):
        key = self.keys[idx]
        return tuple(key)

    def __getitem__(self, idx):
        key = self.get_key(idx)
        return self.dataset[key]

    def __len__(self):
        return len(self.dataset)
