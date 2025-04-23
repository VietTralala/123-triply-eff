import numpy as np
from joblib import Memory

memory = Memory("/tmp/triply_cache", verbose=0)

from src.bell_sampling.bell_circuit import build_full_bell_generation_circuit


def bell_measurement_probs(rho, sigma=None):
    """
    Compute the probabilities of all possible Bell measurement outcomes
    on the joint state `rho ⊗ sigma`.

    Parameters
    ----------
    rho : ndarray
        Density matrix of shape (2^n, 2^n) for `n` qubits.
    sigma : ndarray, optional
        Second density matrix of shape (2^n, 2^n). If None, defaults to `rho`.

    Returns
    -------
    probs : dict
        Dictionary mapping 2n-bit binary strings (Bell outcomes) to their probabilities.
    """
    if sigma is None:
        sigma = rho
    rhorho = np.kron(rho, sigma)

    nqubits = int(np.log2(rho.shape[0]))

    bell_states = build_full_bell_generation_circuit(nqubits)
    # here we can use the bell creation circuit
    rhorho_psi = rhorho @ bell_states

    probs_array = np.real(np.sum(bell_states.conj() * rhorho_psi, axis=0))

    # Clip very small negative values to 0 for numerical stability
    # probs_array[xp.isclose(probs_array, 0.0) & (probs_array < 0)] = 0.0
    probs_array = np.clip(probs_array, 0.0, 1)

    # Renormalize the probabilities to sum to 1
    total_prob = np.sum(probs_array)
    probs_array /= total_prob
    # Convert results to a dictionary with binary keys
    probs = {
        np.binary_repr(idx, width=2 * nqubits): prob
        for idx, prob in enumerate(probs_array)
    }

    return probs


@memory.cache
def compute_pauli_labels(n):
    """
    Compute the Pauli operator type on each qubit for all `4^n` n-qubit Pauli strings.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    labels : ndarray of shape (n, 4**n)
        Each column corresponds to a Pauli string, represented as an n-tuple of integers:
        0 = I, 1 = X, 2 = Y, 3 = Z for each qubit position.
    """
    four_n = 4**n
    labels = np.zeros((n, four_n), dtype=np.uint8)
    for label_index in range(four_n):
        tmp = label_index
        for q in reversed(range(n)):
            labels[q, label_index] = tmp % 4
            tmp //= 4
    return labels


def sample_bell_measurement_binArr(rho, sigma=None, num_samples=1000, probs=None):
    """Perform Bell sampling on the given state rho and return sampled outcomes as a binary array."""
    if probs is None:
        probs = bell_measurement_probs(rho, sigma=sigma)

    # Convert binary string keys (e.g., "111011") to a NumPy array of bits
    sorted_keys = sorted(probs.keys())

    outcomes = np.array(
        [list(map(int, key)) for key in sorted_keys], dtype=np.uint8
    )  # Shape (num_unique_outcomes, 2n)
    prob_values = np.array([probs[key] for key in sorted_keys], dtype=np.float64)

    # Sample using row indices and return the corresponding binary rows
    sampled_indices = np.random.choice(len(outcomes), size=num_samples, p=prob_values)

    return outcomes[sampled_indices]  # Shape (num_samples, 2n)
