import numpy as np
from pennylane import shadows


def compute_energy_from_shadow(bits, recipes, coupling_matrix):
    """
    compute expectation value of the Hamiltonian of the Heisenberg model specified via the coupling_matrix
    """
    wires = bits.shape[1]
    expval_sum = .0

    for i, j in zip(*np.triu_indices(coupling_matrix.shape[0], k=1)):
        coeff = coupling_matrix[i, j]
        if coeff == 0:
            continue

        for p in range(3):
            pauli_word = -1 * np.ones(shape=(wires,)).astype(int)
            pauli_word[i] = pauli_word[j] = int(p)
            expval_sum += coeff * shadows.pauli_expval(bits, recipes, word=np.expand_dims(pauli_word, 0))

    return np.mean(expval_sum)
