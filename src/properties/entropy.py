from jax import jit
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from qutip import Qobj


def compute_entropies_from_statevector(statevector, wires):
    """
    compute second-order Rényi entanglement entropies for all subsystems of size at most
    for a state described by a statevector
    """
    statevector_qobj = Qobj(statevector, dims=[[2] * wires, [1] * wires])

    # compute entropies
    entropies = np.zeros(shape=(wires, wires), dtype=float)
    for i in range(wires):
        ptrace_diag = statevector_qobj.ptrace(sel=[i])
        entropies[i, i] = -np.log(np.trace(ptrace_diag * ptrace_diag).real)

        for j in range(i + 1, wires):
            ptrace = statevector_qobj.ptrace(sel=[i, j])
            e = -np.log(np.trace(ptrace * ptrace).real)
            entropies[i, j] = entropies[j, i] = e

    return entropies


@jit
def _jax_compute_size_two_entropies(x):
    return -jnp.log(jnp.einsum('tilm,siml,tjrk,sjkr->ij', x, x, x, x))


@jit
def _jax_compute_size_one_entropies(x):
    return -jnp.log(jnp.einsum('timl,silm->i', x, x))


def compute_entropies_from_shadow(bits, recipes):
    """
    compute second-order Rényi entanglement entropies for all subsystems of size at most
    two, using the classical shadow protocol
    """
    # init shadow
    shadow = qml.ClassicalShadow(bits=bits, recipes=recipes)
    local_snapshots = shadow.local_snapshots()
    shadow_size = shadow.snapshots

    # compute size two entropies
    entropies = np.array(
        _jax_compute_size_two_entropies(local_snapshots) + 2 * np.log(shadow_size)
    )

    # compute size one entropies
    entropies_size_one = np.array(
        _jax_compute_size_one_entropies(local_snapshots) + 2 * np.log(shadow_size)
    )
    np.fill_diagonal(entropies, entropies_size_one)

    return entropies.real
