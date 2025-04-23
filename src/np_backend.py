import numpy as np
from scipy import linalg as np_linalg
import functools as ft


try:
    import cupy as cp
    from cupyx.scipy import linalg as cupy_linalg

    # import but dont use yet
    HAS_CUPY = True
except ImportError:
    cp = np  # Fallback to NumPy if CuPy isn't installed
    HAS_CUPY = False

# Default backend is CPU (NumPy)
_backend = np
_linalg = np_linalg


def set_gpu(enable=True):
    """Switch backend to GPU (CuPy) or CPU (NumPy)."""
    global _backend
    if enable:
        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy is not installed. Install it with `pip install cupy`."
            )
        _backend = cp
        _linalg = cupy_linalg
    else:
        _backend = np


def is_using_gpu():
    """Returns True if the current backend is CuPy."""
    return _backend is not np


def get_backend():
    """Returns the currently active backend (NumPy or CuPy)."""
    return _backend


def to_numpy(x):
    """Convert CuPy array to NumPy array if necessary."""
    return cp.asnumpy(x) if _backend is cp else x


def to_gpu(x):
    """Convert NumPy array to CuPy array if necessary."""
    if _backend is np and not HAS_CUPY:
        raise RuntimeError("CuPy is not installed. Install it with `pip install cupy`.")
    return cp.asarray(x) if _backend is np and HAS_CUPY else x


def __getattr__(name):
    """
    Automatically redirect undefined function calls to the active backend.
    """
    return getattr(_backend, name)


## matrix functions
def expm(x):
    return _linalg.expm(x)


def gibbs_state(H, beta: float = 1.0):
    exp_ = expm(-beta * H)
    return exp_ / _backend.trace(exp_)


# efficient implementation of trace(A @ B)
def tr_prod(A, B):
    # return np.sum(A * B.T)
    return _backend.real_if_close(_backend.sum(A * B.T))


def convert2numpy(*args):
    res = []
    for arg in args:
        if isinstance(arg, cp.ndarray):
            res.append(cp.asnumpy(arg))
        elif isinstance(arg, (list, tuple)):
            res.append(convert2numpy(*arg))
        elif isinstance(arg, dict):
            res.append({k: convert2numpy(v) for k, v in arg.items()})
        else:
            res.append(arg)
    return res


def kron_chain(ops, permutation=None):
    """Calculate the Kronecker product of a chain of operators.

    Parameters:
    ops (list): List of operators to be Kroneckered.
    permutation (list, tuple): List of indices to permute the operators.

    Returns:
    xp.ndarray: Kronecker product of the operators in the list.
    """
    if isinstance(permutation, (list, tuple)):
        assert len(permutation) == len(
            ops
        ), "Permutation must have the same length as the list of operators"
        for ii, p in enumerate(sorted(permutation)):
            assert p == ii, "Permutation must be a permutation of the list of operators"
        ops = [ops[p] for p in permutation]

    # # Initialize the result with the first operator
    # result = ops[0]
    # # Loop over the remaining operators and Kronecker them
    # for op in ops[1:]:
    #     result = xp.kron(result, op)
    # return result
    return ft.reduce(_backend.kron, ops)


def matmul_chain(ops):
    """Calculate the matrix product of a chain of operators.

    Parameters:
    ops (list): List of operators to be multiplied.

    Returns:
    xp.ndarray: Matrix product of the operators in the list.
    """

    # # Initialize the result with the first operator
    # result = ops[0]
    # # Loop over the remaining operators and multiply them
    # for op in ops[1:]:
    #     result = xp.dot(result, op)
    # return result
    return ft.reduce(_backend.dot, ops)
