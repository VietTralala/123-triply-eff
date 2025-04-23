import numpy as np
import numba

from src.mm_state import find_violating_label, tr_prod, gibbs_state_obj


def compute_mimicking_state_v1_fast(
    n,
    epsilon,
    u_vec_thresh,
    P_stack,
    sign_oracle_vec,
    max_iterations=None,
):
    """
    Construct a mimicking quantum state using v1 algorithm (fast version).

    Parameters
    ----------
    n : int
        Number of qubits.
    epsilon : float
        Accuracy threshold for constraint satisfaction.
    u_vec_thresh : ndarray
        Target Pauli vector with thresholding applied (shape (4^n,)).
    P_stack : ndarray
        Pauli operators as an array of shape (4^n, 2^n, 2^n).
    sign_oracle_vec : ndarray
        Sign vector (+1/-1) giving constraint directions.
    max_iterations : int, optional
        Maximum number of iterations to run.

    Returns
    -------
    result : dict
        Dictionary containing the constructed state, metadata, and diagnostics.
    """
    T = int(np.ceil(64 * n / epsilon**2)) + 1
    if max_iterations:
        # T = min(T, max_iterations)
        max_iterations = min(T, max_iterations)
    else:
        max_iterations = T
    beta = np.sqrt(n / T)

    # Run fast loop
    (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        has_converged,
        M_sum,
        omega,
    ) = compute_v1_fast(
        n, epsilon, u_vec_thresh, sign_oracle_vec, P_stack, beta, max_iterations
    )

    return {
        "algo_type": "v1",
        "final_state": omega,
        "oracle_calls": oracle_calls,
        "update_calls": update_calls,
        "selected_labels": selected_labels,
        "update_factors": factor_seq,
        "T": T,
        "max_iterations": max_iterations,
        "beta": beta,
        "n": n,
        "epsilon_mm": epsilon,
        "has_converged": has_converged,
    }


@numba.njit
def compute_v1_fast(
    n,
    epsilon,
    u_vec_thresh,
    sign_oracle_vec,
    P_stack,
    beta,
    max_iters,
):
    """
    Fast Numba-accelerated loop for mimicking state construction (v1 algorithm).

    Iteratively updates a state using signed constraint violations and accumulates
    a Hamiltonian whose Gibbs state is returned.

    Parameters
    ----------
    n : int
        Number of qubits.
    epsilon : float
        Accuracy threshold.
    u_vec_thresh : ndarray
        Thresholded Pauli expectation vector.
    sign_oracle_vec : ndarray
        Signs of constraints (+1 or -1).
    P_stack : ndarray
        Pauli matrices of shape (4^n, 2^n, 2^n).
    beta : float
        Inverse temperature parameter for Gibbs update.
    max_iters : int
        Maximum number of iterations.

    Returns
    -------
    oracle_calls : int
        Number of oracle queries made.
    update_calls : int
        Number of Gibbs updates performed.
    selected_labels : list of int
        Indices of Pauli terms selected.
    factor_seq : list of int
        Signs (+1/-1) used for each selected term.
    has_converged : bool
        Whether the algorithm terminated via convergence.
    M_sum : ndarray
        Final sum of weighted Pauli matrices (Hamiltonian).
    omega : ndarray
        Final mimicking state (Gibbs state of M_sum).
    """
    D = 2**n
    omega = np.eye(D, dtype=np.complex128) / D
    M_sum = np.zeros((D, D), dtype=np.complex128)

    oracle_calls = 0
    update_calls = 0
    selected_labels = []
    factor_seq = []
    has_converged = False
    for t in range(max_iters):
        label = find_violating_label(u_vec_thresh, P_stack, omega, epsilon)
        if label == -1:
            has_converged = True
            break  # converged

        selected_labels.append(label)
        oracle_calls += 1
        update_calls += 1

        P = P_stack[label]
        r_P = sign_oracle_vec[label]
        diff = tr_prod(P, omega) - r_P * u_vec_thresh[label]
        fac = np.sign(diff)
        factor_seq.append(fac)
        M_sum += fac * P

        # call gibbs_state outside Numba
        omega = gibbs_state_obj(M_sum, beta)

    return (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        has_converged,
        M_sum,
        omega,
    )
