import os
import shutil
import sys
import uuid

import numpy as np
import torch


class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def average(self):
        return self.avg

    def value(self):
        return self.val


def warmup_sqrt_decay_lr_scheduler(optimizer, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    def lr_decay_func(step):
        if step == 0:
            step = 1

        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: lr_decay_func(step))


def warm_up_cosine_lr_scheduler(optimizer, epochs, warm_up_epochs, eta_min):
    """ Description: Warm up cosin learning rate scheduler, first epoch lr is small

    Arguments:
        - optimizer: input optimizer for the training
        - epochs: int, total epochs for your training, default is 100.
            NOTE: you should pass correct epochs for your training
        - warm_up_epochs: int, default is 5, which mean the lr will be warm up for 5 epochs. if warm_up_epochs=0, means
        no need to warn up, will be as cosine lr scheduler
        - eta_min: float, setup ConsinAnnealingLR eta_min while warm_up_epochs = 0

    Returns:
        - scheduler
    """

    if warm_up_epochs <= 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    else:
        def lr_decay_func(epoch):
            if epoch <= warm_up_epochs:
                return eta_min + epoch / warm_up_epochs
            else:
                return eta_min + 0.5 * (1 + np.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * np.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: lr_decay_func(step))

    return scheduler


def dir_setup(results_root, dataset_id, model_id, num_train, train_id, verbose=True):
    """setup dir for training results and model checkpoints"""
    if train_id is None:
        train_id = 'id-' + str(uuid.uuid4())
        if verbose:
            print(f'==> generated random uuid {train_id}')

    model_dir = os.path.join(results_root, dataset_id, model_id)
    results_dir = os.path.join(model_dir, num_train, train_id)

    if train_id == 'debug' and os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        if verbose:
            print(f'removed dir {results_dir}')

    os.makedirs(results_dir)
    if verbose:
        print(f'created dir {results_dir}')

    # create directory to save trained model
    checkpoints_dir = os.path.join(results_dir, "checkpoints/")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    return results_dir, model_dir, train_id


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "out.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
            print(f'removed {self.log_file}')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


def split_train_val(measurements, coupling_matrices, hamiltonian_ids, rng, frac=0.0):
    """
    function splits data into train and validation splits consisting of different Hamiltonians
    Args:
        measurements (np.ndarray):  contains measurements for all hamiltonians, shape (N, n)
        coupling_matrices (np.ndarray): contains coupling matrices for all hamiltonians, shape (N, n, n)
        hamiltonian_ids (np.ndarray): contains id for each hamiltonians, shape (N, 1)
        rng: numpy random number generator
        frac (float): defines the fraction of hamiltonians to be used for validation

    Returns:
        two tuples containing measurements, coupling matrices and hamiltonian ids, one tuple for train and one for
         validation
    """
    hamiltonian_ids_unique = np.unique(hamiltonian_ids)

    if frac == 0.0:
        return (measurements, coupling_matrices, hamiltonian_ids), (None, None, None)

    n_train = int(len(hamiltonian_ids_unique) * (1.0 - frac))
    train_ids = rng.choice(hamiltonian_ids_unique, n_train, replace=False)
    val_ids = np.setdiff1d(hamiltonian_ids_unique, train_ids, assume_unique=True)

    train_measurements = measurements[np.isin(hamiltonian_ids, train_ids)]
    train_coupling_matrices = coupling_matrices[np.isin(hamiltonian_ids, train_ids)]
    train_hamiltonian_ids = hamiltonian_ids[np.isin(hamiltonian_ids, train_ids)]

    val_measurements = measurements[np.isin(hamiltonian_ids, val_ids)]
    val_coupling_matrices = coupling_matrices[np.isin(hamiltonian_ids, val_ids)]
    val_hamiltonian_ids = hamiltonian_ids[np.isin(hamiltonian_ids, val_ids)]

    return (
        (train_measurements, train_coupling_matrices, train_hamiltonian_ids),
        (val_measurements, val_coupling_matrices, val_hamiltonian_ids)
    )
