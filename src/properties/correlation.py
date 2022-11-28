import itertools as it
import numpy as np
import pennylane as qml


def compute_correlation_matrix_from_statevector(statevector, wires, device_name):
    # build circuit to compute expectation values for the ground state
    def _circ(observables):
        qml.QubitStateVector(statevector, wires=range(wires))
        return [qml.expval(o) for o in observables]

    device = qml.device(device_name, wires=wires, shots=None)
    exact_circuit = qml.QNode(_circ, device, interface=None, diff_method=None)

    qubit_pairs = list(it.combinations(range(wires), r=2))

    correlations = np.zeros((wires, wires))
    np.fill_diagonal(correlations, 1.0)

    for idx, (i, j) in enumerate(qubit_pairs):
        obs = [op(i) @ op(j) for op in [qml.PauliX, qml.PauliY, qml.PauliZ]]
        correlations[i, j] = correlations[j, i] = np.sum(np.array(exact_circuit(observables=obs)).T) / 3

    return correlations


def compute_correlation_matrix_from_shadow(bits, recipes, k=1):
    wires = bits.shape[1]
    shadow = qml.ClassicalShadow(bits=bits, recipes=recipes)

    qubit_pairs = list(it.combinations(range(wires), r=2))

    correlations = np.zeros((wires, wires))
    np.fill_diagonal(correlations, 1.0)

    for idx, (i, j) in enumerate(qubit_pairs):
        obs = qml.PauliX(i) @ qml.PauliX(j) + qml.PauliY(i) @ qml.PauliY(j) + qml.PauliZ(i) @ qml.PauliZ(j)
        correlations[i, j] = correlations[j, i] = shadow.expval(H=obs, k=k) / 3

    return correlations
