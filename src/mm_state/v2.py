import numpy as np
import numba
from src.mm_state import find_violating_label, tr_prod, gibbs_state_obj


@numba.njit
def update_step_numba(Ht, P, rho, r_P, uP, beta, eta_pf):
    """
    Perform a single update step with backtracking for mimicking state v2.

    Tries overshoot updates and backtracks (halving `eta_pf`) until improvement is achieved.

    Parameters
    ----------
    Ht : ndarray
        Current Hamiltonian matrix.
    P : ndarray
        Pauli operator for the violated constraint.
    rho : ndarray
        Current density matrix (Gibbs state).
    r_P : int
        Sign (+1 or -1) from sign oracle.
    uP : float
        Target Pauli expectation value.
    beta : float
        Inverse temperature.
    eta_pf : float
        Initial prefactor for the update.

    Returns
    -------
    H_new : ndarray of shape (2^n, 2^n)
        Updated Hamiltonian.
    rho_new : ndarray of shape (2^n, 2^n)
        Updated Gibbs state.
    new_eta_pf : float or None
        Adapted eta prefactor (or None if update rejected).
    update_fac : float or None
        Magnitude of update applied.
    substeps : int
        Number of backtracking steps performed.
    accepted : bool
        Whether update was accepted.
    """
    substeps = 0
    eta = tr_prod(P, rho) - r_P * uP
    update_fac = eta

    while True:
        # Attempt an overshoot update
        Htp1 = Ht + eta_pf * update_fac * P
        rhop1 = gibbs_state_obj(Htp1, beta)

        # If we got a valid improvement, accept it
        if np.abs(tr_prod(P, rhop1) - r_P * uP) < np.abs(eta):
            return Htp1, rhop1, eta_pf * 1.3, update_fac, substeps, True

        eta_pf /= 2
        substeps += 1
        if eta_pf < 1e-20:
            # since we cant do a meaningful update, we return the current state
            return Ht, rho, None, None, substeps, False


@numba.njit
def compute_v2_fast(
    n,
    epsilon,
    u_vec_thresh,
    sign_oracle_vec,
    P_stack,
    beta,
    eta_init,
    max_iters,
):
    """
    Run the fast version of mimicking state construction (v2 algorithm).

    Incorporates adaptive step size with backtracking.

    Parameters
    ----------
    n : int
        Number of qubits.
    epsilon : float
        Constraint tolerance.
    u_vec_thresh : ndarray
        Target Pauli vector after thresholding.
    sign_oracle_vec : ndarray
        Signs for constraints (+1/-1).
    P_stack : ndarray
        Stack of Pauli operators (shape: 4^n x 2^n x 2^n).
    beta : float
        Inverse temperature.
    eta_init : float
        Initial update prefactor.
    max_iters : int
        Maximum number of iterations.

    Returns
    -------
    oracle_calls : int
        Number of calls to the sign oracle.
    update_calls : int
        Number of updates applied (same as length of `selected_labels`).
    selected_labels : list of int
        Indices of selected violating Pauli terms.
    factor_seq : list of float
        Signed update factors applied to each selected term.
    overshoot_list : list of int
        Number of backtracking steps for each iteration.
    eta_prefactor_list : list of float
        Eta prefactor used at each iteration.
    total_overshoot : int
        Total number of backtracking steps across all iterations.
    has_converged : bool
        True if convergence criterion met within `max_iters`.
    accepted : bool
        Whether the final attempted update was accepted.
    omega : ndarray of shape (2^n, 2^n)
        Final mimicking quantum state (Gibbs state of accumulated Hamiltonian).
    """
    D = 2**n
    omega = np.eye(D, dtype=np.complex128) / D
    M_sum = np.zeros((D, D), dtype=np.complex128)

    oracle_calls = 0
    update_calls = 0
    selected_labels = []
    factor_seq = []
    overshoot_list = []
    eta_prefactor_list = []
    total_overshoot = 0
    current_eta_pf = eta_init
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
        u_val = u_vec_thresh[label]

        M_sum, omega, current_eta_pf, update_fac, substeps, accepted = (
            update_step_numba(M_sum, P, omega, r_P, u_val, beta, current_eta_pf)
        )

        overshoot_list.append(substeps)
        eta_prefactor_list.append(current_eta_pf)
        factor_seq.append(update_fac)
        total_overshoot += substeps

        if not accepted:
            break

    return (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        overshoot_list,
        eta_prefactor_list,
        total_overshoot,
        has_converged,
        accepted,
        omega,
    )


def compute_mimicking_state_v2_fast(
    n,
    epsilon,
    u_vec_thresh,
    P_stack,
    sign_oracle_vec,
    max_iterations=None,
    eta_prefactor=100,
):
    """
    Construct a mimicking quantum state using v2 algorithm (with overshoot updates).

    Parameters
    ----------
    n : int
        Number of qubits.
    epsilon : float
        Constraint tolerance.
    u_vec_thresh : ndarray
        Thresholded Pauli vector.
    P_stack : ndarray
        Stack of Pauli operators.
    sign_oracle_vec : ndarray
        Signs of constraints.
    max_iterations : int, optional
        Maximum iterations allowed.
    eta_prefactor : float, optional
        Initial step size scaling.

    Returns
    -------
    result : dict
        {
            "algo_type" : str,                      # Identifier: 'v2'
            "final_state" : ndarray,                # Final density matrix (2^n x 2^n)
            "oracle_calls" : int,                   # Number of oracle calls
            "update_calls" : int,                   # Number of Gibbs updates performed
            "selected_labels" : list of int,        # Indices of Pauli terms used
            "update_factors" : list of float,       # Signed update magnitudes
            "overshoot_sequence" : list of int,     # Backtracking steps per iteration
            "eta_prefactor_sequence" : list of float, # Eta values per iteration
            "total_overshoot" : int,                # Sum of all overshoot steps
            "T" : int,                               # Theoretical iteration bound
            "max_iterations" : int,                  # Actual max iterations used
            "beta" : float,                          # Inverse temperature
            "n" : int,                               # Number of qubits
            "epsilon_mm" : float,                    # Target tolerance
            "has_converged" : bool,                  # Whether stopping condition was met
            "failed_due_to_substeps" : bool          # True if failure due to line search failure
        }
    """
    T = int(np.ceil(64 * n / epsilon**2)) + 1
    if max_iterations:
        max_iterations = min(T, max_iterations)
    else:
        max_iterations = T
    beta = np.sqrt(n / T)

    results = compute_v2_fast(
        n,
        epsilon,
        u_vec_thresh,
        sign_oracle_vec,
        P_stack,
        beta,
        eta_prefactor,
        max_iterations,
    )

    (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        overshoot_list,
        eta_prefactor_list,
        total_overshoot,
        has_converged,
        accepted_step,
        omega,
    ) = results

    return {
        "algo_type": "v2",
        "final_state": omega,
        "oracle_calls": oracle_calls,
        "update_calls": update_calls,
        "selected_labels": selected_labels,
        "update_factors": factor_seq,
        "overshoot_sequence": overshoot_list,
        "eta_prefactor_sequence": eta_prefactor_list,
        "total_overshoot": total_overshoot,
        "T": T,
        "max_iterations": max_iterations,
        "beta": beta,
        "n": n,
        "epsilon_mm": epsilon,
        "has_converged": has_converged,
        "failed_due_to_substeps": not accepted_step,
    }
