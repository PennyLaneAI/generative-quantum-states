'''
Adapted from https://pennylane.ai/qml/demos/tutorial_ml_classical_shadowss.html
'''

from glob import glob

import numpy as np
import pandas as pd
import pennylane as qml

from src.utils import measurement2readout_obs, one_hot
from .utils import get_grid_coordinates

PAULI_NAME_TO_OP = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ, }


def estimate_shadow_obs(shadow, observable, k=10):
    shadow_size = shadow[0].shape[0]

    # convert Pennylane observables to indices
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_obs = np.array([map_name_to_int[observable.name]])
        target_locs = np.array([observable.wires[0]])
    else:
        target_obs = np.array([map_name_to_int[o.name] for o in observable.obs])
        target_locs = np.array([o.wires[0] for o in observable.obs])

    # perform median of means to return the result
    means = []
    meas_list, obs_lists = shadow
    for i in range(0, shadow_size, shadow_size // k):
        meas_list_k, obs_lists_k = (
            meas_list[i: i + shadow_size // k],
            obs_lists[i: i + shadow_size // k],
        )
        indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)
        if sum(indices):
            means.append(
                np.sum(np.prod(meas_list_k[indices][:, target_locs], axis=1)) / sum(
                    indices)
            )
        else:
            means.append(0)

    return np.median(means)


def determine_phase_1D(Z2, Z3, threshold=None, z2_threshold=0.8, z3_threshold=0.8):
    phases = []
    if threshold is not None:
        z2_threshold, z3_threshold = threshold, threshold
    for i, (z2, z3) in enumerate(zip(Z2, Z3)):
        if z2 < z2_threshold and z3 < z3_threshold:
            phases.append('Disordered')
        elif z2 >= z2_threshold and z3 >= z3_threshold:
            if z2 > z3:
                phases.append('Z2')
            else:
                phases.append('Z3')
        elif z2 >= z2_threshold and z3 < z3_threshold:
            phases.append('Z2')
        elif z2 < z2_threshold and z3 >= z3_threshold:
            phases.append('Z3')
        else:
            print(f'{i}-th entry: Z2={z2}, Z3={z2}')
            raise ValueError('Something went wrong')
    return np.array(phases)


def determine_phase_2D(df: pd.DataFrame, order_params: np.ndarray,
                       threshold: float = 0.5):
    phases = []
    order_params = np.array(order_params)
    for i, row in enumerate((df[order_params].values > threshold)):
        phase_idxes = np.argwhere(row).flatten()
        if len(phase_idxes) == 0:
            phase = 'Disordered'
        elif len(phase_idxes) > 1:
            row_idx = df.index[i]
            # print(
            #     f'{row_idx}-th row: multiple phases, {order_params[phase_idxes]}\n {df.loc[row_idx]}')
            phase = order_params[df.iloc[i][order_params].values.argmax()]
        else:
            phase = order_params[phase_idxes[0]]
        phases.append(phase)
    return np.array(phases)


def est_order_param_1D_from_measurements(measurements, order_param='Z2', **kwargs):
    measurements = np.asarray(measurements, dtype=float)
    measurements += 1e-8  # to avoid division by zero
    # denominator
    Zs = measurements.sum(axis=1)
    if order_param == 'Z2':
        return np.mean(measurements[:, ::2].sum(axis=1) / Zs)
    elif order_param == 'Z3':
        return np.mean(measurements[:, ::3].sum(axis=1) / Zs)
    elif order_param == 'Z4':
        return np.mean(measurements[:, ::4].sum(axis=1) / Zs)
    elif order_param == 'Z5':
        return np.mean(measurements[:, ::5].sum(axis=1) / Zs)
    else:
        raise NotImplementedError(
            f"Order parameter {order_param} not supported.")


def est_order_param_1D(density, order_param='Z2', **kwargs):
    if density.sum() == 0.:
        return 0.
    if order_param == 'Z2':
        return density[::2].sum() / density.sum()
    elif order_param == 'Z3':
        return density[::3].sum() / density.sum()
    elif order_param == 'Z4':
        return density[::4].sum() / density.sum()
    elif order_param == 'Z5':
        return density[::5].sum() / density.sum()
    else:
        raise NotImplementedError(
            f"Order parameter {order_param} not supported.")


def est_order_param_1D_from_measurements_huang(Ms, order_param: str):
    # Hsin-Yuan Huang's version in https://arxiv.org/abs/2106.12627
    n_samples = len(Ms)
    n_qubits = Ms.shape[1]
    Ms = Ms.astype(int)
    assert Ms.min() >= 0 and Ms.max() <= 1
    if order_param == 'Z2':
        vals = []
        for i in range(n_qubits - 1):
            # |01><01| + |10><10|
            v = np.mean(np.isclose(Ms[:, i] + Ms[:, i + 1], 1))
            vals.append(v)
        return np.mean(vals)
    elif order_param == 'Z3':
        vals = []
        for i in range(n_qubits - 2):
            # |001><001| + |010><010| + |100><100|
            v = np.mean(np.isclose(Ms[:, i] + Ms[:, i + 1] + Ms[:, i + 2], 1))
            vals.append(v)
        return np.mean(vals)
    elif order_param == 'Z4':
        vals = []
        for i in range(n_qubits - 3):
            # |0001><0001| + |0010><0010| + |0100><0100| + |1000><1000|
            v = np.mean(np.isclose(
                Ms[:, i] + Ms[:, i + 1] + Ms[:, i + 2] + Ms[:, i + 3], 1))
            vals.append(v)
        return np.mean(vals)
    else:
        raise ValueError(f"Order parameter {order_param} not supported.")


def est_order_param_1D_fourier(density, nx, order_param: str, **kwargs):
    # Empirically/theoretically determined
    norm_consts = {'Z2': 1 / 2, 'Z3': 1 / 3, 'Z4': 1 / 4}
    if order_param == 'Z2':
        ks = np.array([np.pi, -np.pi])
    elif order_param == 'Z3':
        ks = np.array([np.pi * 2 / 3, -np.pi * 2 / 3, ])
    elif order_param == 'Z4':
        ks = np.array([np.pi * 1 / 2, -np.pi * 1 / 2, ])
    else:
        raise ValueError(f'Unknown order param {order_param}')
    F_ks = fourier_transform_1D(density=density, ks=ks, nx=nx)
    expval = np.mean(F_ks) / norm_consts[order_param]
    return expval


def est_order_param_1D_fourier_from_measurements(measurements, nx, order_param: str,
                                                 **kwargs):
    # Empirically/theoretically determined
    norm_consts = {'Z2': 1 / 2, 'Z3': 1 / 3, 'Z4': 1 / 4}
    if order_param == 'Z2':
        ks = np.array([np.pi, -np.pi])
    elif order_param == 'Z3':
        ks = np.array([np.pi * 2 / 3, -np.pi * 2 / 3, ])
    elif order_param == 'Z4':
        ks = np.array([np.pi * 1 / 2, -np.pi * 1 / 2, ])
    else:
        raise ValueError(f'Unknown order param {order_param}')
    F_ks = fourier_transform_1D_from_measurements(measurements, ks=ks, nx=nx)
    expval = np.mean(F_ks) / norm_consts[order_param]
    return expval


def est_order_param_2D(density, nx, ny, order_param: str, a: float = 1, **kwargs):
    order_param = order_param.lower()
    norm_consts = {'checkboard': 1.6, 'striated': 0.8,
                   'custom': 1., 'staggered': 1.}  # Empirically determined
    if order_param == 'checkboard':
        ks = np.array([[np.pi, np.pi],
                       [np.pi, 0],
                       [0, np.pi],
                       ])
        F_ks = fourier_transform_square_lattice(density, ks, nx, ny, a=a)
        expval = F_ks[0] - np.mean(F_ks[1:])
    elif order_param == 'striated':
        ks = np.array([[np.pi, 0],
                       [0, np.pi],
                       [np.pi / 2, np.pi],
                       [np.pi, np.pi / 2],
                       ])
        F_ks = fourier_transform_square_lattice(density, ks, nx, ny, a=a)
        expval = np.mean(F_ks[:2]) - np.mean(F_ks[1:])
    elif order_param == 'star':
        ks = np.array([[np.pi, np.pi / 2],
                       [np.pi / 2, np.pi],
                       [np.pi, 3 * np.pi / 2],
                       [3 * np.pi / 2, np.pi],
                       ])
        F_ks = fourier_transform_square_lattice(density, ks, nx, ny, a=a)
        expval = np.mean(F_ks)
    elif order_param == 'custom' or order_param == 'staggered':
        ks = np.array([[np.pi / 2, np.pi / 2],
                       [3 * np.pi / 2, np.pi / 2],
                       [np.pi / 2, 3 * np.pi / 2],
                       [3 * np.pi / 2, 3 * np.pi / 2],
                       ])
        F_ks = fourier_transform_square_lattice(density, ks, nx, ny, a=a)
        expval = np.mean(F_ks)
    else:
        raise ValueError(f'Unknown order param {order_param}')
    return expval / norm_consts[order_param]


def est_order_param_2D_from_measurements(measurements, nx, ny, order_param: str,
                                         a: float = 1, **kwargs):
    order_param = order_param.lower()
    norm_consts = {'checkboard': 1.6, 'striated': 0.8,
                   'custom': 1., 'staggered': 1.}  # Empirically determined
    if order_param == 'checkboard':
        ks = np.array([[np.pi, np.pi],
                       [np.pi, 0],
                       [0, np.pi],
                       ])
        F_ks = fourier_transform_square_lattice_from_measurements(
            measurements, ks, nx, ny, a=a)
        expval = F_ks[0] - np.mean(F_ks[1:])
    elif order_param == 'striated':
        ks = np.array([[np.pi, 0],
                       [0, np.pi],
                       [np.pi / 2, np.pi],
                       [np.pi, np.pi / 2],
                       ])
        F_ks = fourier_transform_square_lattice_from_measurements(
            measurements, ks, nx, ny, a=a)
        expval = np.mean(F_ks[:2]) - np.mean(F_ks[1:])
    elif order_param == 'star':
        ks = np.array([[np.pi, np.pi / 2],
                       [np.pi / 2, np.pi],
                       [np.pi, 3 * np.pi / 2],
                       [3 * np.pi / 2, np.pi],
                       ])
        F_ks = fourier_transform_square_lattice_from_measurements(
            measurements, ks, nx, ny, a=a)
        expval = np.mean(F_ks)
    elif order_param == 'custom' or order_param == 'staggered':
        # This is our custom order parameter of Staggered phase for 5x5 -- for larger systems
        # this order parameter needs to be modified
        ks = np.array([[np.pi / 2, np.pi / 2],
                       [3 * np.pi / 2, np.pi / 2],
                       [np.pi / 2, 3 * np.pi / 2],
                       [3 * np.pi / 2, 3 * np.pi / 2],
                       ])
        F_ks = fourier_transform_square_lattice_from_measurements(
            measurements, ks, nx, ny, a=a)
        expval = np.mean(F_ks)
    else:
        raise ValueError(f'Unknown order param {order_param}')
    return expval / norm_consts[order_param]


def fourier_transform_square_lattice(density, ks, nx: int, ny: int, a: float = 1., ):
    N = nx * ny
    assert len(density.shape) == 1 and density.shape[0] == N
    if not isinstance(ks, np.ndarray):
        ks = np.array(ks)
    xs = get_grid_coordinates(np.arange(nx) * a, np.arange(ny) * a)
    ft = np.exp(-1j * ks @ (xs.T) / a) * density / np.sqrt(N)
    F_ks = np.abs(np.sum(ft, axis=1))
    return F_ks


def fourier_transform_square_lattice_from_measurements(Ms, ks, nx: int, ny: int,
                                                       a: float = 1., ):
    N = nx * ny
    assert len(Ms.shape) == 2
    assert Ms.shape[1] == N
    if not isinstance(ks, np.ndarray):
        ks = np.array(ks)
    xs = get_grid_coordinates(np.arange(nx) * a, np.arange(ny) * a)
    ft = np.exp(-1j * ks @ (xs.T) / a) @ Ms.T / np.sqrt(N)
    F_ks = np.mean(np.abs(ft), axis=1)
    return F_ks


def fourier_transform_1D(density, ks, nx: int, a: float = 1., ):
    assert len(density.shape) == 1 and density.shape[0] == nx
    assert len(ks.shape) == 1
    xs = np.arange(nx) * a
    ft = np.exp(-1j * np.outer(ks, xs) / a) * density / nx
    ffs = np.abs(np.sum(ft, axis=1))
    return ffs


def fourier_transform_1D_from_measurements(Ms, ks, nx: int, a: float = 1., ):
    assert len(Ms.shape) == 2 and Ms.shape[1] == nx
    assert len(ks.shape) == 1
    Ms = Ms.astype(int)
    n_samples = Ms.shape[0]
    xs = np.arange(nx) * a
    ft = np.exp(-1j * np.outer(ks, xs) / a) @ Ms.T / nx
    ffs = np.mean(np.abs(ft), axis=1)
    return ffs


PHASE_NORM_CONSTANTS = {"checkboard": 1.6, "striated": 0.8, "custom": 1.,
                        "Z2": 1 - 1 / 2, "Z3": 1 - 1 / 3, "Z4": 1 - 1 / 4, None: 1.}
PHASE_OFFSETS = {"checkboard": 0., "striated": 0., "custom": 0.,
                 "Z2": 1 / 2, "Z3": 1 / 3, "Z4": 1 / 4, None: 0.}


def est_phase_diagram(folder, dim=2, nx=5, ny=5,
                      ranges=None, suffix='', order_params=None):
    if ranges is None:
        ranges = []
        for fname in glob(folder + "/interaction_range-*.npz"):
            ranges.append(float(fname.split("-")[-1].rstrip('.npz')))
    if len(ranges) == 0:
        print("No data in", folder)
        return None

    assert dim in [1, 2]
    if order_params is None:
        order_params = ["Z2", "Z3", "Z4"] if dim == 1 else [
            "checkboard", "striated", "custom"]
    assert len(order_params) <= 3
    if len(order_params) < 3:
        order_params += [None] * (3 - len(order_params))
    ranges = np.sort(ranges)
    order_1 = []
    order_2 = []
    order_3 = []

    NONE_ENTRY = 0.
    for r in ranges:
        order_1.append([])
        order_2.append([])
        order_3.append([])
        result = dict(np.load(folder + f"/interaction_range-{r}.npz"))
        for density in result["density_z"]:
            if dim == 1:
                order_1[-1].append(
                    est_order_param_1D(density, order_param=order_params[0], ) if
                    order_params[
                        0] is not None else NONE_ENTRY)
                order_2[-1].append(
                    est_order_param_1D(density, order_param=order_params[1], ) if
                    order_params[
                        1] is not None else NONE_ENTRY)
                order_3[-1].append(
                    est_order_param_1D(density, order_param=order_params[2], ) if
                    order_params[
                        2] is not None else NONE_ENTRY)
            elif dim == 2:
                order_1[-1].append(
                    est_order_param_2D(density, nx, ny, order_param=order_params[0])
                    if order_params[0] is not None else NONE_ENTRY)
                order_2[-1].append(
                    est_order_param_2D(density, nx, ny, order_param=order_params[1])
                    if order_params[1] is not None else NONE_ENTRY)
                order_3[-1].append(
                    est_order_param_2D(density, nx, ny, order_param=order_params[2])
                    if order_params[2] is not None else NONE_ENTRY)
            else:
                raise ValueError(f"Dimension {dim} not supported.")
    # results[r] = result

    order_1 = (np.array(
        order_1) - PHASE_OFFSETS[order_params[0]]) / PHASE_NORM_CONSTANTS[
                  order_params[0]]
    order_2 = (np.array(
        order_2) - PHASE_OFFSETS[order_params[1]]) / PHASE_NORM_CONSTANTS[
                  order_params[1]]
    order_3 = (np.array(
        order_3) - PHASE_OFFSETS[order_params[2]]) / PHASE_NORM_CONSTANTS[
                  order_params[2]]

    channel_1 = np.flip(order_1, axis=0)
    channel_2 = np.flip(order_2, axis=0)
    channel_3 = np.flip(order_3, axis=0)
    im = np.stack([channel_1, channel_2, channel_3]).transpose([1, 2, 0])
    return im


def phase2img(df, order_params, xlabel='detuning', ylabel='interaction_range',
              xticks=None, yticks=None):
    if xticks is None:
        xticks = np.sort(df[xlabel].unique())
    if yticks is None:
        yticks = np.flip(np.sort(df[ylabel].unique()))
    im = np.full((len(yticks), len(xticks), len(order_params)), np.nan)
    for i, x in enumerate(xticks):
        for j, y in enumerate(yticks):
            row = df.loc[(df[xlabel] == x) & (df[ylabel] == y)]
            assert len(row) == 1
            row = row.iloc[0]
            im[j, i, :] = np.array([row[order_param]
                                    for order_param in order_params])
    return im, xticks, yticks


def get_neighbors(x, y, nx, ny, neighbor_order=1):
    # as neighbor_order = 1, consider the nearest neighbors
    # as neighbor_order = 2, consider the nearest neighbors and the second nearest neighbors
    assert neighbor_order == 1, 'Only neighbor_order = 1 is supported so far'
    neighbors = []
    for i in range(max(x - neighbor_order, 0), 1 + min(x + neighbor_order, nx - 1)):
        for j in range(max(y - neighbor_order, 0), 1 + min(y + neighbor_order, ny - 1)):
            shift_x = i - x
            shift_y = j - y
            if np.abs(shift_x) + np.abs(shift_y) == 1:
                neighbors.append([i, j])
    return np.array(neighbors)


def single_site_staggered_magnetization(densities, x, y, nx, ny, neighbor_order=1):
    '''
    Compute the staggered magnetization of a system at sites (x, y)
    densities: a list of density matrices, shape = (batch, nx, ny)
    x, y: the coordinates of the site
    nx, ny: the dimension of the 2D system
    neighbor_order: the order of the nearest neighbors to consider
    return: the staggered magnetization of all sites across the batch, shape = (batch, )
    '''
    neighbors = get_neighbors(x, y, nx, ny, neighbor_order=neighbor_order)
    n_i = densities[:, x, y]
    n_js = densities[:, neighbors[:, 0], neighbors[:, 1]]

    sign = (-1.) ** ((x + 1) + (y + 1))
    m_i = sign * (n_i - np.mean(n_js, axis=1))
    return m_i


def est_staggered_magnetization(densities, nx, ny, neighbor_order=1):
    ms = np.zeros((len(densities), nx, ny))
    for x in range(nx):
        for y in range(ny):
            ms[:, x, y] = single_site_staggered_magnetization(
                densities, x, y, nx, ny, neighbor_order=neighbor_order)
    return ms


def est_avg_corr_cov_2D(Ms, nx, ny, ):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    xs, ys = xs.flatten(), ys.flatten()
    N = len(xs)
    # Cov = np.zeros(2 * (nx - 1), 2 * (ny - 1))
    Cov = {}
    for i in range(N):
        x_i, y_i = xs[i], ys[i]
        for j in range(N):
            x_j, y_j = xs[j], ys[j]
            shift_x = x_j - x_i
            shift_y = y_j - y_i
            key = (shift_x, shift_y)
            # print(x_i, x_j, y_i, y_j)
            cov = np.mean(Ms[:, x_i, y_i] * Ms[:, x_j, y_j]) - \
                  Ms[:, x_i, y_i].mean() * Ms[:, x_j, y_j].mean()
            if not (key in Cov):
                Cov[key] = [cov]
            else:
                Cov[key].append(cov)
    for k, v in Cov.items():
        Cov[k] = np.mean(v)
    return Cov


def n_groups_median_of_means(n_obs):
    return int(2 * np.log(2 * n_obs)) + 1


def corr_function_single_pauli(i: int, j: int, pauli_idx: str):
    assert len(pauli_idx) == 1
    pauli_idx = pauli_idx.capitalize()
    op = PAULI_NAME_TO_OP[pauli_idx]
    return op(i) @ op(j)


def est_density_from_shadows(n_qubits, shadows, k=None, pauli_idx: str = 'Z',
                             n_basis=True):
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits)  # group size

    def est_single_site(i):
        o = PAULI_NAME_TO_OP[pauli_idx.capitalize()](i)
        expval = estimate_shadow_obs(shadows, o, k=k)
        if n_basis:
            # from Z-basis to the basis of number operator
            expval = (expval + 1) / 2
        return expval

    vals = []
    for i in range(n_qubits):
        vals.append(est_single_site(i))
    return np.array(vals)


def est_order_param_from_shadows(n_qubits, shadows, k=None):
    density = est_density_from_shadows(n_qubits, shadows, k=k)
    return np.mean(density[::2]) - np.mean(density[1::2])


def est_corr_from_shadows(n_qubits, shadows, pauli_idx: str, k=None):
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits - 1)  # group size
    corr_mat = np.ones((n_qubits, n_qubits))
    # diagonal entries have all ones
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            obs = corr_function_single_pauli(i, j, pauli_idx=pauli_idx)
            expval_ij = estimate_shadow_obs(shadows, obs, k=k)
            corr_mat[i, j] = corr_mat[j, i] = expval_ij
    return corr_mat


def est_ccorr_from_shadows(n_qubits, shadows, pauli_idx: str, s=1, k=None):
    density = est_density_from_shadows(
        n_qubits, shadows, k=k, n_basis=False, pauli_idx=pauli_idx)
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits - 1)  # group size
    terms = []
    for i in range(n_qubits - s):
        j = i + s
        obs = corr_function_single_pauli(i, j, pauli_idx=pauli_idx)
        expval_ij = estimate_shadow_obs(shadows, obs, k=k)
        terms.append(expval_ij - density[i] * density[j])
    return np.mean(terms)


def bitstr2prod_state(bitstrs):
    # for each bitstr, convert it to a product state
    n_qubits = bitstrs.shape[1]
    multiplier = np.ones(n_qubits)
    for i in range(n_qubits):
        multiplier[i] *= 2 ** (n_qubits - i - 1)
    locs = np.sum(bitstrs * multiplier, axis=1).astype(int)
    return one_hot(locs, 2 ** n_qubits)


def bitstrs2pure_state(bitstrs):
    prod_states = bitstr2prod_state(bitstrs)
    return np.sqrt(np.mean(prod_states, axis=0))


def est_obs_from_z_measurements(measurements: np.ndarray,
                                observable: qml.operation.Observable, k: int = 1):
    all_bitstrs = (measurements + 1) / 2  # map from {-1,1} to {0,1}
    n_qubits = measurements.shape[1]
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_locs = np.array([observable.wires[0]])
    else:
        # if PauliX(i) @ PauliX(i), then we only use one local site
        target_locs = np.unique(np.array([o.wires[0] for o in observable.obs]))
        assert len(target_locs) <= 2, "Only support up to two-point observables"
    # e.g., if observable = PauliX(i)@PauliY(j), then n_locs = 2
    n_locs = len(target_locs)
    means = []  # for median-of-means
    sample_idxes = np.random.permutation(len(measurements))
    list_sample_idxes = np.array_split(sample_idxes, k)

    @qml.qnode(device=qml.device('lightning.qubit', wires=n_qubits, ))
    def _circ(ground_state, observables):
        qml.QubitStateVector(ground_state, wires=range(n_qubits))
        return [qml.expval(o) for o in observables]

    for sample_idxes in list_sample_idxes:
        bitstrs = all_bitstrs[sample_idxes]
        local_bitstrs = bitstrs[:, target_locs]
        local_state = bitstrs2pure_state(local_bitstrs)

        # print(observable.matrix())
        expval = local_state @ (observable.matrix() @ local_state)
        means.append(expval)
    return np.mean(means)


def est_density_from_z_measurements(n_qubits, measurements, k=None,
                                    pauli_idx: str = 'Z', n_basis=True):
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits)  # group size

    def est_single_site(i):
        o = PAULI_NAME_TO_OP[pauli_idx.capitalize()](i)
        expval = est_obs_from_z_measurements(measurements, o, k=k)
        if n_basis:
            # from Z-basis to the basis of number operator
            expval = (expval + 1) / 2
        return expval

    vals = []
    for i in range(n_qubits):
        vals.append(est_single_site(i))
    return np.array(vals)


def est_corr_from_z_measurements(n_qubits, measurements, pauli_idx: str, k=None):
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits - 1)  # group size
    corr_mat = np.zeros((n_qubits, n_qubits))
    # diagonal entries have all ones
    for i in range(n_qubits):
        for j in range(i, n_qubits):
            obs = corr_function_single_pauli(i, j, pauli_idx=pauli_idx)
            expval_ij = est_obs_from_z_measurements(measurements, obs, k=k)
            corr_mat[i, j] = corr_mat[j, i] = expval_ij
    return corr_mat


def est_cov_from_z_measurements(n_qubits, measurements, pauli_idx: str, k=None):
    expval_pauli = est_density_from_z_measurements(
        n_qubits, measurements, k=k, n_basis=False, pauli_idx=pauli_idx)
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits - 1)  # group size
    corr_mat = np.zeros((n_qubits, n_qubits))
    # diagonal entries have all ones
    for i in range(n_qubits):
        for j in range(i, n_qubits):
            obs = corr_function_single_pauli(i, j, pauli_idx=pauli_idx)
            expval_ij = est_obs_from_z_measurements(measurements, obs, k=k)
            corr_mat[i, j] = corr_mat[j, i] = expval_ij - \
                                              expval_pauli[i] * expval_pauli[j]
    return corr_mat


def est_ccorr_from_z_measurements(n_qubits, measurements, pauli_idx: str, s=1, k=None):
    expval_pauli = est_density_from_z_measurements(
        n_qubits, measurements, k=k, n_basis=False, pauli_idx=pauli_idx)
    if k is None:
        k = n_groups_median_of_means(n_obs=n_qubits - 1)  # group size
    terms = []
    for i in range(n_qubits - s):
        j = i + s
        obs = corr_function_single_pauli(i, j, pauli_idx=pauli_idx)
        expval_ij = est_obs_from_z_measurements(measurements, obs, k=k)
        terms.append(expval_ij - expval_pauli[i] * expval_pauli[j])
    return np.mean(terms)


class RydbergEvaluator():
    def __init__(self, n_qubits, observable='density',
                 exact_val=None, n_threads=1, use_tqdm=False, save_best=True,
                 return_val=False, z_basis_recon=False):
        self.n_qubits = n_qubits
        self.n_threads = n_threads
        self.use_tqdm = use_tqdm
        self.save_best = save_best
        self.observable = observable
        if observable.startswith('density') or observable.startswith(
                'order_param') or observable.startswith('cov') or (
                'corr' in observable):
            self.exact_val = exact_val
        else:
            raise NotImplementedError

        self.optimal_error = np.inf
        self.best_model = None
        self.return_val = return_val
        self.z_basis_recon = z_basis_recon

    def eval_model(self, model, n_samples, **kwargs):
        shadows = measurement2readout_obs(model.generate(n_samples))
        model.eval()
        est_val, error = self.eval_shadows(shadows, **kwargs)
        if error < self.optimal_error:
            self.optimal_error = error
            if self.save_best:
                self.best_model = model.state_dict()
                for k, v in self.best_model.items():
                    self.best_model[k] = v.cpu()
        return est_val, error

    def est_obs_from_shadows(self, n_qubits, shadows, k=None):
        if self.observable.startswith('density'):
            if '_' in self.observable:
                pauli_idx = self.observable.split('_')[1]
            else:
                pauli_idx = 'Z'
            return est_density_from_shadows(n_qubits, shadows, pauli_idx=pauli_idx, k=k)
        elif self.observable == 'order_param':
            return est_order_param_from_shadows(n_qubits, shadows, k=k)
        elif self.observable.startswith('corr_'):
            pauli_idx = self.observable.split('_')[1]
            return est_corr_from_shadows(n_qubits, shadows, pauli_idx, k=k)
        elif self.observable.startswith('ccorr_'):
            pauli_idx = self.observable.split('_')[1]
            return est_ccorr_from_shadows(n_qubits, shadows, pauli_idx, k=k)
        else:
            return None

    def eval_obs_from_z_measurements(self, n_qubits, measurements, k=None):
        if self.observable.startswith('density'):
            if '_' in self.observable:
                pauli_idx = self.observable.split('_')[1]
            else:
                pauli_idx = 'Z'
            return est_density_from_z_measurements(n_qubits, measurements,
                                                   pauli_idx=pauli_idx, k=k)
        elif self.observable.startswith('corr_'):
            pauli_idx = self.observable.split('_')[1]
            return est_corr_from_z_measurements(n_qubits, measurements,
                                                pauli_idx=pauli_idx, k=k)
        elif self.observable.startswith('cov_'):
            pauli_idx = self.observable.split('_')[1]
            return est_cov_from_z_measurements(n_qubits, measurements,
                                               pauli_idx=pauli_idx, k=k)
        elif self.observable.startswith('ccorr_'):
            pauli_idx = self.observable.split('_')[1]
            return est_ccorr_from_z_measurements(n_qubits, measurements, pauli_idx, k=k)
        else:
            return None

    def eval_estimation(self, est_val, metric='mse'):
        if metric == 'mse':
            error = np.mean((est_val - self.exact_val) ** 2)
        else:
            raise ValueError("Invalid metric {}".format(metric))
        if not self.return_val:
            est_val = None
        return est_val, error

    def eval_shadows(self, shadows, metric='mse', k=None):
        if self.z_basis_recon:
            z_measurements = shadows[0][np.all(shadows[1] == 2, axis=1)]
            return self.eval_z_measurements(z_measurements, metric=metric, k=k)
        else:
            est_val = self.est_obs_from_shadows(
                n_qubits=self.n_qubits, shadows=shadows, k=k)
            return self.eval_estimation(est_val, metric=metric)

    def eval_z_measurements(self, z_measurements, metric='mse', k=None):
        est_val = self.eval_obs_from_z_measurements(
            n_qubits=self.n_qubits, measurements=z_measurements, k=k)
        return self.eval_estimation(est_val, metric=metric)
