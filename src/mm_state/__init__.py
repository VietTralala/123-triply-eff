import numpy as np
import numba
import scipy


@numba.njit
def numba_seed(a):
    """
    Set the random seed inside a Numba JIT-compiled function.

    Parameters
    ----------
    a : int
        Seed value for Numba-compatible RNG.
    """
    np.random.seed(a)


@numba.njit
def find_violating_label(u_vec, P_stack, omega, eps):
    """
    Search for a Pauli label whose expectation under `omega` violates the constraint.

    A label `i` is considered violating if:
        |Tr(P_i · omega) - u_vec[i]| > eps/2
    or
        |Tr(P_i · omega) + u_vec[i]| > eps/2


    Parameters
    ----------
    u_vec : ndarray of shape (4^n,)
        Target Pauli vector (thresholded).
    P_stack : ndarray of shape (4^n, 2^n, 2^n)
        Pauli operator stack.
    omega : ndarray of shape (2^n, 2^n)
        Current density matrix.
    eps : float
        Tolerance parameter.

    Returns
    -------
    label_idx : int
        Index of a violating Pauli term, or -1 if none found.
    """
    significant_u_idcs = np.where(u_vec >= 3 / 4 * eps)[0]
    for ilabel in significant_u_idcs:
        tr_P_omega = tr_prod(P_stack[ilabel], omega)
        u_val = u_vec[ilabel]
        if abs(tr_P_omega - u_val) > eps / 2 and abs(tr_P_omega + u_val) > eps / 2:
            return ilabel
    return -1


@numba.njit
def tr_prod(A, B):  # should always be real for hermitian matrices!
    """
    Compute the real-valued trace of the product of two matrices.

    Equivalent to Re(Tr(A @ B)) but optimized for Hermitian matrices.

    Parameters
    ----------
    A : ndarray
        Matrix A (typically Hermitian).
    B : ndarray
        Matrix B (typically Hermitian).

    Returns
    -------
    tr : float
        Real part of the trace of A @ B.
    """
    return np.real(np.sum(A * B.T))


@numba.njit
def gibbs_state_obj(H, beta):
    """
    Compute the normalized Gibbs state ρ = exp(-βH) / Tr[exp(-βH)].

    Uses a Numba-compatible `objmode` block to call SciPy's matrix exponential.

    Parameters
    ----------
    H : ndarray of shape (D, D)
        Hermitian matrix (Hamiltonian).
    beta : float
        Inverse temperature parameter.

    Returns
    -------
    rho : ndarray of shape (D, D)
        Gibbs state (density matrix).
    """

    # D = H.shape[0]
    with numba.objmode(expH="complex128[:,:]"):
        expH = scipy.linalg.expm(-beta * H)
    Z = np.trace(expH)
    return expH / Z
