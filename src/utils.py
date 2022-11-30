import os
from collections import OrderedDict
from datetime import datetime as dt
from fnmatch import fnmatch
from typing import OrderedDict as OrderedDict_T

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from constants import NUM_MMT_OUTCOMES


def timestamp():
    return dt.now().strftime('%d-%m-%Y %H:%M:%S')


def one_hot(data, num_classes):
    """
    Convert a one-dimensional array of labels into a matrix of one-hot labels.
    :param data: numpy array of shape (num_samples,)
    :param num_classes: number of classes
    :return: numpy array of shape (num_samples, num_classes)
    """
    return np.eye(num_classes)[data]


def torch2numpy(*tensors):
    arrays = []
    for tensor in tensors:
        arrays.append(tensor.detach().cpu().numpy())
    if len(arrays) == 1:
        return arrays[0]
    else:
        return tuple(arrays)


def numpy2torch(*arrays, device=torch.device('cpu')):
    tensors = []
    for array in arrays:
        tensors.append(torch.from_numpy(array).to(device))
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tuple(tensors)


def measurement2onehot(data, n_qubits=None):
    num_data, num_qubits = data.shape
    if n_qubits is not None: assert num_qubits == n_qubits
    new_data = np.sum([data[:, i] * (NUM_MMT_OUTCOMES ** (num_qubits - i - 1)) for i in range(num_qubits)], axis=0)
    data_onehot = one_hot(new_data, num_classes=NUM_MMT_OUTCOMES ** num_qubits)
    return data_onehot


def measurement2label(data, n_qubits=None):
    # TODO: definition of the label:
    num_data, num_qubits = data.shape
    if n_qubits is not None: assert num_qubits == n_qubits
    new_data = np.sum([data[:, i] * (NUM_MMT_OUTCOMES ** (num_qubits - i - 1)) for i in range(num_qubits)], axis=0)
    return new_data


def label2measurement(labels, num_qubits=None):
    # TODO: definition of the label
    assert num_qubits == 2
    m1 = labels // NUM_MMT_OUTCOMES
    m2 = labels % NUM_MMT_OUTCOMES
    return np.array([m1, m2]).T


def measurement2multihot(data, n_qubits=None):
    num_data, num_qubits = data.shape
    if n_qubits is not None: assert num_qubits == n_qubits
    data = data.reshape(-1, )
    data_onehot = one_hot(data, num_classes=NUM_MMT_OUTCOMES)
    return np.reshape(data_onehot, (num_data, num_qubits * NUM_MMT_OUTCOMES)).astype(float)


def multihot2measurement(data_onehot, n_qubits=None):
    N = len(data_onehot)
    data_onehot = data_onehot.reshape((-1, NUM_MMT_OUTCOMES))
    measurements = np.argmax(data_onehot, axis=1)
    measurements = np.reshape(measurements, (N, -1)).astype(int)
    if n_qubits is not None: assert measurements.shape[1] == n_qubits
    return measurements


def measurement2readout_obs(measurments):
    if isinstance(measurments, torch.Tensor):
        measurments = torch2numpy(measurments)
    assert measurments.min() >= 0 and measurments.max() <= 5
    observables = measurments // 2
    readouts = 2. * (measurments % 2) - 1
    return readouts, observables


def process_weight_names(ckpt: OrderedDict_T[str, torch.Tensor]) -> OrderedDict_T[str, torch.Tensor]:
    return OrderedDict([(str(k).replace('module.', ''), v) for k, v in ckpt.items()])


def filter_folder_for_files(folder, file_pattern):
    return [os.path.join(folder, fn) for fn in os.listdir(folder) if fnmatch(fn, file_pattern)]


def plot_phase_diagram(df, figsize=(4.1, 4.5), title=None, train_idxes=None,
                       return_ax: bool = False, hue_order=None, marker='D', legend=False,
                       given_ax=None, x_label=True, y_label=True, y_ticks=True, x_ticks=True,
                       legend_config=dict(loc='upper left', fontsize=11.5, bbox_to_anchor=(-.02, 1.015)),
                       title_config={},
                       ):
    df = df.copy().sort_values(['interaction_range', 'detuning', ], )

    df['new_index'] = np.arange(len(df))

    counts = np.unique(df['interaction_range'].values, return_counts=True)[1]
    assert np.all(counts == counts[0])
    n_detuning = counts[0]

    marker_colors = ['gray', 'gold', 'tab:green', 'indigo']
    marker_colors = np.array([matplotlib.colors.to_rgb(color) for color in marker_colors])

    order_params = hue_order
    im = np.zeros((len(df), 4))

    background_color = marker_colors[0]
    background_alpha = 0.1
    channel_colors = np.array([marker_colors[C] for i, C in enumerate([1, 2, 3, ])])
    phases = df['phase'].values
    background_idxes = np.argwhere(phases == 'Disordered').flatten()
    im[background_idxes, :3] = background_color
    im[background_idxes, 3] = background_alpha  # transparency

    channel = 0
    for order_param in order_params:
        if order_param == 'Disordered': continue
        idxes = np.argwhere(phases == order_param).flatten()
        im[idxes, :3] = channel_colors[channel]
        im[idxes, 3] = 1
        channel += 1
        assert channel <= 3
    if given_ax is not None:
        fig, ax = None, given_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
    im_2d = np.flip(im.reshape(-1, n_detuning, 4), axis=0)
    ax.imshow(im_2d)
    if x_ticks:
        ax.set_xticks(np.array([0, im_2d.shape[1]]) - .5)
        ax.set_xticklabels([np.round(df['detuning'].min(), 1), np.round(df['detuning'].max(), 1)])
    else:
        ax.set_xticks([])
    if y_ticks:
        ax.set_yticks(np.array([0, im_2d.shape[0]]) - .5)
        ax.set_yticklabels([np.round(df['interaction_range'].max(), 1), np.round(df['interaction_range'].min(), 1)])
    else:
        ax.set_yticks([])

    if x_label:
        ax.set_xlabel(r'$\Delta/ \Omega$', fontdict={'fontsize': 10}, labelpad=-5)
    if y_label:
        ax.set_ylabel(r'$R_0/a$', labelpad=-5)

    handles = [mpatches.Patch(color=background_color, alpha=background_alpha, label='Disordered')]
    handles += [mpatches.Patch(color=channel_colors[i - 1], label=hue_order[i]) for i in range(1, len(hue_order))]
    if train_idxes is not None:
        mask = np.zeros(len(im))
        mask[df.loc[train_idxes]['new_index'].values] = 1
        train_coords = np.argwhere(np.flip(mask.reshape(-1, n_detuning), axis=0))
        ax.scatter(train_coords[:, 1], train_coords[:, 0], c='black', alpha=1, s=4, marker=marker)
        handles.append(Line2D([0], [0], marker=marker, color='white', label='Train',
                              markerfacecolor='black', markersize=6, ), )
    if legend == True:
        ax.legend(handles=handles, **legend_config)

    title_config['y'] = title_config.get('y', 1.02)
    ax.set_title(title, **title_config)

    if given_ax is None: fig.tight_layout()

    if return_ax:
        return fig, ax
    else:
        return fig
