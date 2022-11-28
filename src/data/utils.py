import itertools as it
import numpy as np
import pennylane as qml

from constants import PAULI_ENSEMBLE


def build_hamiltonian_from_couplings(coupling_mat):
    coeffs, ops = [], []
    ns = coupling_mat.shape[0]

    for i, j in it.combinations(range(ns), r=2):
        coeff = coupling_mat[i, j]
        if coeff:
            for op in PAULI_ENSEMBLE:
                coeffs.append(coeff)
                ops.append(op(i) @ op(j))

    return qml.Hamiltonian(coeffs, ops)


def ints_to_bits_and_recipes(samples_int: np.ndarray):
    recipes = samples_int // 2
    bits = samples_int - 2 * recipes
    return bits, recipes
