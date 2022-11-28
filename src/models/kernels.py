"""
Code adapted from https://github.com/hsinyuan-huang/provable-ml-quantum/blob/main/Code.ipynb by Hsin-Yuan Huang
"""

import itertools as it

import numpy as np
from neural_tangents import stax
from sklearn.kernel_ridge import KernelRidge


def build_dirichlet_kernel(xs):
    dirichlet_kernel = np.zeros((len(xs), xs.shape[1] * 5))
    for i, x1 in enumerate(xs):
        cnt = 0
        for k in range(len(x1)):
            for k1 in range(-2, 3):
                dirichlet_kernel[i, cnt] += np.cos(np.pi * k1 * x1[k])
                cnt += 1
    return dirichlet_kernel


def build_NTK(xs, ys, layers: int, C: float, normalize=False, activation: str = 'Relu', return_kernel_fn=False):
    activation = getattr(stax, activation)
    init_fn, apply_fn, kernel_fn = stax.serial(
        *[*it.chain(*[[stax.Dense(32), activation()] for _ in range(layers)]), stax.Dense(1)]
    )
    nt_kernel = kernel_fn(xs, xs, 'ntk')
    if normalize:
        diag = np.diag(nt_kernel)
        nt_kernel = nt_kernel / (np.sqrt(np.outer(diag, diag)))

    def ntk_pred_fn(xs_test, C=C, return_kernel=False, return_pred=True):
        clf_ntk = KernelRidge(kernel='precomputed', alpha=1 / (2 * C))
        clf_ntk.fit(nt_kernel, ys)

        kernel_test_train = kernel_fn(xs_test, xs, 'ntk')
        if normalize:
            kernel_test = kernel_fn(xs_test, xs_test, 'ntk')
            diag_test, diag_train = np.diag(kernel_test), np.diag(nt_kernel)
            kernel_test_train = kernel_test_train / (np.sqrt(np.outer(diag_test, diag_train)))
        pred = clf_ntk.predict(kernel_test_train)
        if return_kernel:
            return pred, kernel_test_train
        else:
            return pred

    if return_kernel_fn:
        return nt_kernel, ntk_pred_fn, kernel_fn
    else:
        return nt_kernel, ntk_pred_fn
