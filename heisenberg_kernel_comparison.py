import fnmatch
import itertools as it
import json
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm

from constants import DATA_DIR
from src.data.utils import ints_to_bits_and_recipes
from src.properties.correlation import compute_correlation_matrix_from_shadow
from src.properties.entropy import compute_entropies_from_shadow

DATA_ROOT = os.path.join(DATA_DIR, '2d_heisenberg/')
RESULTS_ROOT = './results/'

_PROP_CORRELATIONS = 'correlations'
_PROP_ENTROPIES = 'entropies'

_RBF_KERNEL = 'rbf'


def main(rows, cols, prop, kernel_name):
    assert prop in [_PROP_CORRELATIONS, _PROP_ENTROPIES]

    qubits = rows * cols

    # load training data
    train_hamiltonians, train_couplings, train_shadow_y_values, train_true_y_values = load_data(
        rows, cols, 'train',
        prop)

    # load testing data
    test_hamiltonians, test_couplings, test_shadow_y_values, test_true_y_values = load_data(
        rows, cols, 'test', prop)

    if kernel_name == _RBF_KERNEL:
        train_kernel_mat = train_couplings
        test_kernel_mat = test_couplings
        kernel = 'rbf'
    else:
        raise ValueError(f'unknown kernel_name {kernel_name}')

    # containers for predictions and true correlation matrices
    test_predicted_properties = {i: np.zeros(shape=(qubits, qubits)) for i in
                                 test_hamiltonians}
    test_true_properties = {i: np.zeros(shape=(qubits, qubits)) for i in
                            test_hamiltonians}

    train_predicted_properties = {i: np.zeros(shape=(qubits, qubits)) for i in
                                  train_hamiltonians}
    train_true_properties = {i: np.zeros(shape=(qubits, qubits)) for i in
                             train_hamiltonians}

    # make predictions
    pbar = tqdm(total=((qubits + 1) * qubits) / 2,
                desc=f'running inference for {prop} with {kernel_name} kernel')
    for i in range(qubits):
        for j in range(i, qubits):
            idx = i * qubits + j

            if i == j and prop == _PROP_CORRELATIONS:
                test_predictions_instance = np.ones(shape=len(test_hamiltonians))
                train_predictions_instance = np.ones(shape=len(train_hamiltonians))
            else:
                # make predictions
                train_predictions_instance, test_predictions_instance = train_and_predict(
                    train_kernel_mat, test_kernel_mat, train_shadow_y_values[:, idx],
                    kernel=kernel
                )

            for hid, pred in enumerate(test_predictions_instance):
                test_predicted_properties[hid][i, j] = test_predicted_properties[hid][
                    j, i] = pred
                test_true_properties[hid][i, j] = test_true_properties[hid][j, i] = \
                test_true_y_values[hid, idx]

            for hid, pred in enumerate(train_predictions_instance):
                train_predicted_properties[hid][i, j] = train_predicted_properties[hid][
                    j, i] = pred
                train_true_properties[hid][i, j] = train_true_properties[hid][j, i] = \
                train_true_y_values[hid, idx]

            pbar.update()

    # dump predictions and ground truths
    res_root = os.path.join(
        RESULTS_ROOT,
        f'conditional_heisenberg_{rows}x{cols}/{kernel_name}-kernel/ns1000/results/'
    )
    props_dir_test = os.path.join(res_root, f'properties/test/model/')
    data_dir_test = os.path.join(res_root, f'data/{rows}x{cols}/test/')

    props_dir_train = os.path.join(res_root, f'properties/train/model/')
    data_dir_train = os.path.join(res_root, f'data/{rows}x{cols}/train/')

    if not os.path.exists(os.path.join(props_dir_test, f'{prop}')):
        os.makedirs(os.path.join(props_dir_test, f'{prop}'))

    if not os.path.exists(os.path.join(props_dir_train, f'{prop}')):
        os.makedirs(os.path.join(props_dir_train, f'{prop}'))

    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)

    if not os.path.exists(data_dir_train):
        os.makedirs(data_dir_train)

    # compute mse for each hamiltonian and dump to props dir
    test_mse = {
        idx: np.mean((test_predicted_properties[idx] - test_true_properties[idx]) ** 2)
        for idx in test_hamiltonians
    }

    train_mse = {
        idx: np.mean(
            (train_predicted_properties[idx] - train_true_properties[idx]) ** 2) for idx
        in train_hamiltonians
    }

    print(
        f'Test Prediction MSE for {prop} with {kernel_name} kernel: {np.mean(list(test_mse.values()))}')
    print(
        f'Train Prediction MSE for {prop} with {kernel_name} kernel: {np.mean(list(train_mse.values()))}')

    with open(os.path.join(props_dir_test, f'model_{prop}_mse.json'), 'w') as f:
        json.dump(test_mse, f)

    with open(os.path.join(props_dir_train, f'model_{prop}_mse.json'), 'w') as f:
        json.dump(train_mse, f)

    # dump individual predictions and ground truths
    for idx in test_hamiltonians:
        # predictions
        np.save(os.path.join(props_dir_test, f'{prop}/{prop}_model_id{idx}.npy'),
                test_predicted_properties[idx])

        # ground truths
        fn_tag = 'correlation_matrix' if prop == _PROP_CORRELATIONS else 'entanglement_entropies'
        np.save(os.path.join(data_dir_test, f'{fn_tag}_id{idx}.npy'),
                test_true_properties[idx])

    for idx in train_hamiltonians:
        # predictions
        np.save(os.path.join(props_dir_train, f'{prop}/{prop}_model_id{idx}.npy'),
                train_predicted_properties[idx])

        # ground truths
        fn_tag = 'correlation_matrix' if prop == _PROP_CORRELATIONS else 'entanglement_entropies'
        np.save(os.path.join(data_dir_train, f'{fn_tag}_id{idx}.npy'),
                train_true_properties[idx])


def train_and_predict(train_kernel, test_kernel, train_labels_shadow, kernel='linear'):
    # instance-wise normalization
    for i in range(len(train_kernel)):
        train_kernel[i] /= np.linalg.norm(train_kernel[i])

    for i in range(len(test_kernel)):
        test_kernel[i] /= np.linalg.norm(test_kernel[i])

    # use cross validation to find the best method + hyper-param
    best_cv_score, test_score = 999.0, 999.0
    test_predictions = np.nan
    train_predictions = np.nan
    for ML_method in [
        lambda cx: svm.SVR(kernel=kernel, C=cx),
        lambda cx: KernelRidge(kernel=kernel, alpha=1 / (2 * cx))
    ]:
        for C in [0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.0, 2.0]:
            score = -np.mean(cross_val_score(
                ML_method(C), train_kernel, train_labels_shadow, cv=5,
                scoring="neg_root_mean_squared_error"
            ))
            if best_cv_score > score:
                clf = ML_method(C)
                clf.fit(train_kernel, train_labels_shadow.ravel())
                test_predictions = clf.predict(test_kernel)
                train_predictions = clf.predict(train_kernel)
                best_cv_score = score

    return train_predictions, test_predictions


def load_data(rows, cols, split, prop):
    data_dir = os.path.join(DATA_ROOT, f'{rows}x{cols}', split)

    # get hamiltonian ids
    hamiltonians_ids = sorted([
        int(fn[(fn.find('id') + 2):fn.find('.npy')]) for fn in os.listdir(data_dir)
        if fnmatch.fnmatch(fn, 'data_id*.npy')
    ])

    true_properties = np.empty(shape=(len(hamiltonians_ids), (rows * cols) ** 2))
    shadow_properties = np.empty(shape=(len(hamiltonians_ids), (rows * cols) ** 2))
    couplings = []

    if prop == _PROP_CORRELATIONS:
        fn_tag = 'correlation_matrix'

        def prop_func(b, r):
            return compute_correlation_matrix_from_shadow(b, r, k=1)

    elif prop == _PROP_ENTROPIES:
        fn_tag = 'entanglement_entropies'

        def prop_func(b, r):
            return compute_entropies_from_shadow(b, r)

    else:
        raise ValueError(f'unknown property {prop}')

    # loop through data dir
    for i, hid in enumerate(sorted(hamiltonians_ids)):
        # load true correlation
        true_prop_hid = np.load(os.path.join(data_dir, f'{fn_tag}_id{hid}.npy'))
        true_properties[i, :] = true_prop_hid.flatten()

        # compute shadow correlations
        samples = np.load(os.path.join(data_dir, f'data_id{hid}.npy'))
        bits, recipes = ints_to_bits_and_recipes(samples)
        shadow_prop = prop_func(bits, recipes)
        shadow_properties[i, :] = shadow_prop.flatten()

        # load couplings
        coupling_factors = np.load(
            os.path.join(data_dir, f'coupling_matrix_id{hid}.npy'))
        coupling_factors = [coupling_factors[si, sj] for (si, sj) in
                            it.combinations(range(rows * cols), 2) if
                            ((sj % cols > 0) and sj - si == 1) or sj - si == cols]
        couplings.append(coupling_factors)

    couplings = np.array(couplings)

    return hamiltonians_ids, couplings, shadow_properties, true_properties


if __name__ == '__main__':
    # correlations
    main(2, 5, prop=_PROP_CORRELATIONS, kernel_name=_RBF_KERNEL)
    main(2, 6, prop=_PROP_CORRELATIONS, kernel_name=_RBF_KERNEL)
    main(2, 7, prop=_PROP_CORRELATIONS, kernel_name=_RBF_KERNEL)
    main(2, 8, prop=_PROP_CORRELATIONS, kernel_name=_RBF_KERNEL)
    main(2, 9, prop=_PROP_CORRELATIONS, kernel_name=_RBF_KERNEL)

    # entropies
    main(2, 5, prop=_PROP_ENTROPIES, kernel_name=_RBF_KERNEL)
    main(2, 6, prop=_PROP_ENTROPIES, kernel_name=_RBF_KERNEL)
    main(2, 7, prop=_PROP_ENTROPIES, kernel_name=_RBF_KERNEL)
    main(2, 8, prop=_PROP_ENTROPIES, kernel_name=_RBF_KERNEL)
    main(2, 9, prop=_PROP_ENTROPIES, kernel_name=_RBF_KERNEL)
