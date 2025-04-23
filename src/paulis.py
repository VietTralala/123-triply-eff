import numpy as np
import numba
from joblib import Memory


memory = Memory("/tmp/triply_cache", verbose=0)

import ipdb


@numba.jit
def calc_pauli_vec(rho, P_stack):
    """
    Compute expectation values ⟨P⟩ = Tr(P·ρ) for a all Pauli operators P.

    Parameters
    ----------
    rho : ndarray
        Density matrix of shape (2^n, 2^n).
    P_stack : ndarray
        Array of shape (4^n, 2^n, 2^n) containing all Pauli matrices.

    Returns
    -------
    uP : ndarray
        Real-valued vector of Pauli expectations, shape (4^n,).
    """

    num_paulis = P_stack.shape[0]
    # dim = rho.shape[0]
    uP = np.empty(num_paulis, dtype=np.float64)

    # Loop over Pauli matrices
    for i in range(num_paulis):
        prod = P_stack[i] @ rho
        tr = np.trace(prod)
        uP[i] = tr.real

    return uP


@memory.cache
def generate_all_Ps_stacked(n, allPs=None):
    """
    Return sorted labels and a stacked array of all Pauli operators for `n` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    allPs : dict, optional
        Precomputed dictionary of Pauli operators. If None, it is generated.

    Returns
    -------
    labels : list of str
        Sorted Pauli labels.
    P_stack : ndarray
        Array of shape (4^n, 2^n, 2^n) with all Pauli matrices.
    """
    if allPs is None:
        allPs = generate_all_Ps(n)
    assert isinstance(allPs, dict)

    labels = sorted(list(allPs.keys()))
    P_list = [allPs[l] for l in labels]
    P_stack = np.array(P_list)
    return np.array(labels), P_stack


pauli_dict = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


@memory.cache
def generate_all_Ps(n):
    """
    Generate all tensor products of Pauli matrices for `n` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    pauli_dict : dict
        Dictionary mapping Pauli labels (e.g. 'XYI') to 2^n x 2^n matrices.
    """
    return __generate_all_Ps(n)


_generate_all_pauli_cache = {}


def __generate_all_Ps(n, clear_smaller=True):
    """
    Recursive, cached generation of all `n`-qubit Pauli operators.

    Parameters
    ----------
    n : int
        Number of qubits.
    clear_smaller : bool, optional
        Whether to clear cached entries for smaller `n`.

    Returns
    -------
    pauli_dict : dict
        Dictionary mapping Pauli strings to corresponding matrices.
    """
    if n in _generate_all_pauli_cache:
        return _generate_all_pauli_cache[n]
    if n == 1:
        _generate_all_pauli_cache[1] = {l: pauli_dict[l] for l in "IXYZ"}
        return _generate_all_pauli_cache[1]
    smaller = __generate_all_Ps(n - 1, clear_smaller=False)
    paulis_n = {}
    for label1, mat1 in smaller.items():
        for l2 in "IXYZ":
            paulis_n[label1 + l2] = np.kron(mat1, pauli_dict[l2])
    _generate_all_pauli_cache[n] = paulis_n
    if clear_smaller:
        for k in list(_generate_all_pauli_cache.keys()):
            if k < n:
                del _generate_all_pauli_cache[k]
    return paulis_n


def index_to_pauli_string(label_index, n):
    """
    Convert an integer index to its corresponding Pauli string label.

    Parameters
    ----------
    label_index : int
        Index in the range [0, 4^n).
    n : int
        Number of qubits.

    Returns
    -------
    label : str
        Pauli string label (e.g., 'XIZ').
    """
    idx_to_letter = ["I", "X", "Y", "Z"]
    tmp = label_index
    letters = [None] * n
    for q in reversed(range(n)):
        letters[q] = idx_to_letter[tmp % 4]
        tmp //= 4
    return "".join(letters)


if __name__ == "__main__":
    import time

    n = 3
    allPs = generate_all_Ps(n)
    labels, P_stack = generate_all_Ps_stacked(n, allPs=allPs)
    rho = np.eye(2**n).astype(np.complex128) / (2**n)
    start = time.time()
    uP = calc_pauli_vec(rho, P_stack)
    print("Time:", time.time() - start)
    print(uP)
    ipdb.set_trace()

    test_pidx = 24
    pauli_str = index_to_pauli_string(test_pidx, n)
    print(f"Pauli string for index {test_pidx}: {pauli_str}")  # 24 -> XYI

    # manual_P = np.kron(np.kron(pauli_dict["X"], pauli_dict["Y"]), pauli_dict["I"])
    manual_P = 1.0
    for i, l in enumerate(pauli_str):
        manual_P = np.kron(manual_P, pauli_dict[l])
    print(
        f"Pauli dict construction is ok:", np.allclose(manual_P, allPs[pauli_str])
    )  # True
    print(f"Stack order is ok:", np.allclose(manual_P, P_stack[test_pidx]))  # True
