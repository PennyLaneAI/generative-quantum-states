import itertools as it
import numpy as np


def generate_coupling_matrix(rows, cols, rng, edge_weight_range=(0, 2), periodic_boundary=False):
    """Generate a lattice of n_qubits."""
    wires = rows * cols

    if rows == 1 or cols == 1:
        edges = _ring_lattice(wires, periodic_boundary=periodic_boundary)
    else:
        edges = _square_lattice(rows, cols, periodic_boundary=periodic_boundary)

    edge_weights = rng.uniform(edge_weight_range[0], edge_weight_range[1], size=len(edges))

    coupling_matrix = np.zeros((wires, wires))
    for (i, j), w in zip(edges, edge_weights):
        coupling_matrix[i, j] = coupling_matrix[j, i] = w

    return coupling_matrix


def _ring_lattice(n_qubits, periodic_boundary=False):
    """Generate a 1D ring of n_qubits."""
    edges = [(i, i + 1) for i in range(n_qubits - 1)]

    if periodic_boundary:
        edges.append((n_qubits - 1, 0))

    return edges


def _square_lattice(n_rows, n_cols, periodic_boundary=False):
    """Generate a 2D square lattice of n_rows and n_cols."""
    n_qubits = n_rows * n_cols
    edges = [(si, sj) for (si, sj) in it.combinations(range(n_qubits), 2)
             if ((sj % n_cols > 0) and sj - si == 1) or sj - si == n_cols]

    if periodic_boundary:
        raise NotImplementedError(
            "periodic boundary conditions not implemented yet for square lattice")

    return edges
