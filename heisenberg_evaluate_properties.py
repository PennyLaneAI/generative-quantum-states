"""
script to evaluate models with classical shadows
"""

import argparse
import json
import numpy as np
import os
from tqdm import tqdm
import warnings

from src.utils import filter_folder_for_files
from src.properties.entropy import compute_entropies_from_shadow
from src.properties.correlation import compute_correlation_matrix_from_shadow
from src.data.utils import ints_to_bits_and_recipes

parser = argparse.ArgumentParser()
parser.add_argument('--results-root', type=str)
parser.add_argument('--snapshots', type=int, default=20000,
                    help='number of generated samples for evaluation')
parser.add_argument('--k', type=int, default=1, help='k in median of means')
args = parser.parse_args()


def main(results_root, shots, k):
    with open(os.path.join(results_root, 'args.json'), 'r') as f:
        model_args = json.load(f)

    train_size = model_args['train_size']
    samples_dir = os.path.join(results_root, 'samples')
    data_dir = os.path.join(results_root, 'data', train_size)
    properties_dir = os.path.join(results_root, 'properties')

    if not os.path.exists(properties_dir):
        os.makedirs(properties_dir)

    m_samples_fp = f'model_samples_{shots}' + '_id{}.npy'
    s_samples_fp = 'data_id{}.npy'

    # ------ Model evaluation
    # compute properties for test hamiltonians with model
    test_properties_model, test_errors_model = eval_properties_from_shadow(
        samples_dir=samples_dir, data_dir=data_dir, samples_fp=m_samples_fp,
        split='test', model_name='model', k=k
    )

    # save
    save_properties(test_properties_model, properties_dir, 'test', 'model')
    save_errors(test_errors_model, properties_dir, 'test', 'model')

    # compute properties for train hamiltonians with model
    train_properties_model, train_errors_model = eval_properties_from_shadow(
        samples_dir=samples_dir, data_dir=data_dir, samples_fp=m_samples_fp,
        split='train', model_name='model', k=k
    )

    # save
    save_properties(train_properties_model, properties_dir, 'train', 'model')
    save_errors(train_errors_model, properties_dir, 'train', 'model')

    # print errors
    c_test_mse = np.mean(list(test_errors_model["correlations"].values()))
    c_train_mse = np.mean(list(train_errors_model["correlations"].values()))
    print(f'Model Correlation MSE test/train: {c_test_mse:.4f} / {c_train_mse:.4f}')

    e_test_mse = np.mean(list(test_errors_model["entropies"].values()))
    e_train_mse = np.mean(list(train_errors_model["entropies"].values()))
    print(f'Model Entropy MSE test/train: {e_test_mse:.4f} / {e_train_mse:.4f}')

    # ------ Shadow evaluation (for reference)
    # compute properties for test hamiltonians with shadow
    test_properties_shadow, test_errors_shadow = eval_properties_from_shadow(
        samples_dir=data_dir, data_dir=data_dir, samples_fp=s_samples_fp, split='test',
        model_name=None, k=k
    )

    # save
    save_properties(test_properties_shadow, properties_dir, 'test', 'shadow')
    save_errors(test_errors_shadow, properties_dir, 'test', 'shadow')

    # compute properties for train hamiltonians with shadow
    train_properties_shadow, train_errors_shadow = eval_properties_from_shadow(
        samples_dir=data_dir, data_dir=data_dir, samples_fp=s_samples_fp, split='train',
        model_name=None, k=k
    )

    # save
    save_properties(train_properties_shadow, properties_dir, 'train', 'shadow')
    save_errors(train_errors_shadow, properties_dir, 'train', 'shadow')

    # print errors
    c_test_mse = np.mean(list(test_errors_shadow["correlations"].values()))
    c_train_mse = np.mean(list(train_errors_shadow["correlations"].values()))
    print(f'Shadow Correlation MSE test/train: {c_test_mse:.4f} / {c_train_mse:.4f}')

    e_test_mse = np.mean(list(test_errors_shadow["entropies"].values()))
    e_train_mse = np.mean(list(train_errors_shadow["entropies"].values()))
    print(f'Shadow Entropy MSE test/train: {e_test_mse:.4f} / {e_train_mse:.4f}')


def save_properties(properties_dict, properties_dir, split, model_name):
    for k, d in properties_dict.items():
        save_dir = os.path.join(properties_dir, split, model_name, k)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for hid, v in d.items():
            np.save(os.path.join(save_dir, f'{k}_{model_name}_id{hid}.npy'), v)


def save_errors(errors_dict, properties_dir, split, model_name):
    with open(os.path.join(properties_dir, split, model_name,
                           f'{model_name}_correlations_mse.json'), 'w') as f:
        json.dump(errors_dict['correlations'], f)
    with open(os.path.join(properties_dir, split, model_name,
                           f'{model_name}_entropies_mse.json'), 'w') as f:
        json.dump(errors_dict['entropies'], f)


def eval_properties_from_shadow(samples_dir, data_dir, samples_fp, split, model_name,
                                k):
    samples_dir = os.path.join(samples_dir, split, model_name or '')
    files = filter_folder_for_files(samples_dir, samples_fp.format('*'))
    ids = [int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in files]

    properties = {
        'correlations': {},
        'entropies': {},
    }

    errors = {
        'correlations': {},
        'entropies': {},
    }

    no_true_entanglement_entropies = []

    for hid in tqdm(ids):
        samples = np.load(os.path.join(samples_dir, samples_fp.format(hid)))
        bits, recipes = ints_to_bits_and_recipes(samples)

        # compute correlations
        correlations = compute_correlation_matrix_from_shadow(bits, recipes, k=k)
        properties['correlations'][hid] = correlations

        # compute correlations mse
        true_correlations = np.load(
            os.path.join(data_dir, split, f'correlation_matrix_id{hid}.npy'))
        mse = np.mean((true_correlations - correlations) ** 2)
        errors['correlations'][hid] = mse

        # compute subsystem entanglement entropies
        entropies = compute_entropies_from_shadow(bits, recipes)
        properties['entropies'][hid] = entropies

        # compute entropies mse (this is only supported with <= 20 qubits)
        try:
            true_entropy = np.load(
                os.path.join(data_dir, split, f'entanglement_entropies_id{hid}.npy'))
        except FileNotFoundError:
            no_true_entanglement_entropies.append(hid)
            true_entropy = None

        if true_entropy is not None:
            mse = np.mean((true_entropy - entropies) ** 2)
            errors['entropies'][hid] = mse
        else:
            errors['entropies'][hid] = np.nan

    if len(no_true_entanglement_entropies) > 0:
        warnings.warn(
            f'did not find true entanglement entropies '
            f'for ids {no_true_entanglement_entropies}'
        )

    return properties, errors


if __name__ == '__main__':
    main(results_root=args.results_root, shots=args.snapshots, k=args.k)
