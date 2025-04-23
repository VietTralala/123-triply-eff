import numpy as np


def create_ghz_density_matrix(n):
    """
    Create the density matrix of an `n`-qubit GHZ state.
    |GHZ> = (|0...0> + |1...1>) / sqrt(2)

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : ndarray
        GHZ state as a 2^n x 2^n density matrix.
    """
    dim = 2**n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    psi /= np.sqrt(2)
    return np.outer(psi, psi.conjugate())


def create_zero_density_matrix(n):
    """
    Create the density matrix for the all-zero `n`-qubit state.
    |0...0> as a density matrix for n qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    rho : ndarray
        Density matrix representing |0⟩^⊗n.
    """
    dim = 2**n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    return np.outer(psi, psi.conjugate())
