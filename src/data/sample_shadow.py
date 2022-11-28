import numpy as np
import pennylane as qml


def sample_shadow(statevector, wires, shots, device_name='default.qubit'):
    """ sample classical shadows for the state described by the statevector
    The resulting shadows are encoded as integer according to the logic
    0,1 -> +,- in the X Basis
    2,3 -> r,l in the Y Basis
    4,5 -> 0,1 in the Z Bassi
    """
    @qml.qnode(device=qml.device(device_name, wires=wires, shots=shots), diff_method=None, interface=None)
    def shadow_measurement():
        qml.QubitStateVector(statevector, wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    bits, recipes = shadow_measurement()

    # encode measurements and bases as integers
    data = 2 * recipes + bits
    data = np.array(data, dtype=int)

    return data
