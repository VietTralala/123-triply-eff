import src.np_backend as xp
from math import sqrt
from joblib import Memory

# from src.random_k_local_unitary_helper import bubblesort, perform_col_swap

memory = Memory("/tmp/triply_cache", verbose=0)

#########################################################################
# fast version
#########################################################################


@memory.cache
def build_full_bell_generation_circuit(n):
    """
    Construct the full unitary that generates Bell basis states on 2n qubits.

    Parameters
    ----------
    n : int
        Number of qubits in one subsystem.

    Returns
    -------
    U : ndarray
        Unitary matrix (2^(2n), 2^(2n)) that transforms computational basis to Bell basis.
    """
    return build_full_bell_measurement_circuit(
        n
    ).T  # dont need to conj() since real entries only


@memory.cache
def build_full_bell_measurement_circuit(n):
    """
    Construct the full Bell measurement unitary circuit on 2n qubits.

    Applies parallel CNOT gates (control: qubit i, target: qubit i+n) followed by
    Hadamard gates on the first n qubits.

    |c0>---*----------H---measure
    |c1>---|-*--------H---measure
    |c2>---|-|-*-...--H---measure
    ...
    |t0>---X-|-|----------measure
    |t1>-----X-|----------measure
    |t2>-------X----------measure
    ...

    Parameters
    ----------
    n : int
        Number of qubits in one subsystem.

    Returns
    -------
    U : ndarray
        Bell measurement unitary matrix of shape (2^(2n), 2^(2n)).
    """
    # full circuit will be built in two steps
    # first layer contains all CNOT gates in parallel
    # first n qubits are all control qubits
    # second n qubits are all target qubits
    # second layer contains all Hadamard gates for control qubits
    # and Identity gates for target qubits

    l1 = build_cnot_circuit(n)

    H = 1 / sqrt(2) * xp.array([[1, 1], [1, -1]])
    l2 = [H] * n + [xp.eye(2)] * n  # Hadamard for n c-qubits, Identity for n t-qubits
    l2 = xp.kron_chain(l2)

    rowwise_bellstates = l2 @ l1
    ## normalize rowwise
    # rowwise_bellstates /= xp.linalg.norm(rowwise_bellstates, axis=1, keepdims=True)
    return rowwise_bellstates


@memory.cache
def build_cnot_circuit(n):
    """
    Construct a parallel CNOT circuit for n control-target qubit pairs.

    |c0>---*---------
    |c1>---|-*-------
    |c2>---|-|-*-...-
    ..
    |t0>---X-|-|-----
    |t1>-----X-|-----
    |t2>-------X-----
    ...

    Parameters
    ----------
    n : int
        Number of CNOT gate pairs.

    Returns
    -------
    U : ndarray
        Unitary matrix of shape (2^(2n), 2^(2n)) applying all CNOTs in parallel.
    """
    # option 2
    # we compute swaps so that each qubit control and target are neighbors
    # than kron all CNOT gates
    # than unswap! swaps can be done efficiently by column swapping!

    # create qubit order for swaps:
    # we want to have c0,t0,c1,t1,c2,t2,...,cn,tn
    # we start with c0, c1, c2, ..., cn, t0, t1, t2, ..., tn
    # so we enumerate control qubits with even and target qubits with odd numbers and sort:

    qubit_order = [2 * i for i in range(n)] + [2 * i + 1 for i in range(n)]
    # swaps are a list of (i,i+1) tuples, where i and i+1 the qubits to be swapped
    # starting with the biggest indices!
    # i.e. apply U to 1,2 of 3 qubit system 0,1,2 will swapt such that
    # selected qubit indices are at beginning- > perm = (1,2,0)
    # and thus to get from (0,1,2) to (1,2,0) swaps =  (0,1) then (1,2)
    # but bubble sort will return the swaps in reverse order (from the POV of U)
    # so swaps = (1,2) then (0,1)
    # 0--\/----|U|
    # 1--/\-\/-|_|
    # 2-----/\----
    # and thus we apply the swap 1,2 first onto U then 1,0
    # this is the order in which bubble sort returns the swaps!
    # these swaps transform (120) ->(012)

    # with given qubit order we will swap such that
    # |c0> ---┌───────┐---|c0>---┌───────┐
    # |c1> ---|       |---|t0>---|       |
    # |c2> ---|       |---|c1>---|       |
    # |cn> ---| SWAPs |---|t1>---| CNOTs |
    # |t0> ---|       |---|c2>---|       |
    # |t1> ---|       |---|t2>---|       |
    # |t2> ---|       |---|cn>---|       |
    # |tn> ---└───────┘---|tn>---└───────┘

    # first n qubits are all control qubits
    # second n qubits are all target qubits
    CNOT = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    ops = [CNOT] * n
    allCNOTS = xp.kron_chain(ops)

    allCNOTS = perform_permute(qubit_order, 2 * n, allCNOTS, col=True)
    allCNOTS = perform_permute(qubit_order, 2 * n, allCNOTS, col=False)

    return allCNOTS


def perform_permute(idcs_mapping, n, state, col=True):
    """
    Permute qubit indices of a quantum operator according to a given mapping.

    idcs_mapping given in two line notation where we omit the first line 1...n
    so we map:
    0-> idcs_mapping[0]
    1-> idcs_mapping[1]
    ...
    n-> idcs_mapping[n]


    Parameters
    ----------
    idcs_mapping : list of int
        Target positions for each qubit.
    n : int
        Number of qubits.
    state : ndarray
        Operator to be permuted (2^n x 2^n).
    col : bool, optional
        Whether to permute column (input) indices or row (output) indices.

    Returns
    -------
    permuted_state : ndarray
        Operator with permuted qubit indices.
    """

    mpo = state.reshape([2] * (2 * n))  # reshape to 2x2 tensor for each qubit
    # idc order is sysA, sysB, sysC, ... sysA', sysB', sysC', ...
    # we image the first n indices (unprimed) are col idcs
    # and second n idcs (primed) are row idcs

    assert len(idcs_mapping) == n, "idcs_mapping must have length n"

    if col:
        mapping = {ii: idcs_mapping[ii] for ii in range(n)}
    else:
        # if row needs to be permuted we shall only operate on primed idcs
        # thus map from ii+n -> idcs_mapping[ii]+n
        mapping = {ii + n: idcs_mapping[ii] + n for ii in range(n)}
    # print(mapping)

    mpo_swapped_idcs = [
        mapping.get(i, i) for i in range(2 * n)
    ]  # all indices are swapped or stay the same (default of get())
    # only the first n unprimed indices are swapped
    # print(mpo_swapped_idcs)

    mpo_swapped = xp.transpose(mpo, mpo_swapped_idcs)  # swap the indices
    mpo_swapped = mpo_swapped.reshape([2**n, 2**n])  # back to global idcs
    return mpo_swapped


#########################################################################
# slow version
#########################################################################


def build_full_bell_circuit_slow(n):
    """
    Construct a full entangling Bell circuit for two `n`-qubit systems.

    Applies CNOT and Hadamard gates sequentially for each Bell pair:
    entangles qubit `i` with qubit `i+n` for all `i`.

    Parameters
    ----------
    n : int
        Number of qubits in one subsystem.

    Returns
    -------
    U : ndarray
        Unitary matrix (2^(2n), 2^(2n)) implementing full Bell circuit.
    """
    ops = []
    # this way is very inefficient, since we compute alot of kron products with identity
    # but actually the CNOTs can be computed in parallel
    for i in range(n):
        ops.append(single_bell_pair_circuit(i, i + n, 2 * n))

    return xp.matmul_chain(ops)


def single_bell_pair_circuit(control, target, n):
    """
    Construct a Bell pair creation circuit between two qubits.

    Applies Hadamard to control and a CNOT from control to target.

    Parameters
    ----------
    control : int
        Index of control qubit.
    target : int
        Index of target qubit.
    n : int
        Total number of qubits in the system.

    Returns
    -------
    U : ndarray
        Unitary matrix of shape (2^n, 2^n).
    """
    assert (
        0 <= control < n and 0 <= target < n
    ), f"control {control} and target {target} must be between 0 and n-1={n}"

    H = 1 / sqrt(2) * xp.array([[1, 1], [1, -1]])
    I = xp.eye(2)

    ops1 = get_cnot_op(control, target, n)
    ops2 = [I] * n
    ops2[control] = H
    ops2 = xp.kron_chain(ops2)
    return ops2 @ ops1


def get_cnot_op(control, target, n):
    """
    Return the matrix representation of a CNOT gate on an n-qubit system.
    CNOT = |0><0|_c ⊗ I_t + |1><1|_c ⊗ X_t
    index c is control, index t is target
    in words CNOT applies X gate to t-qubit if control is |1>,
    otherwise it applies identity gate

    Parameters
    ----------
    control : int
        Index of control qubit.
    target : int
        Index of target qubit.
    n : int
        Total number of qubits.

    Returns
    -------
    U : ndarray
        Unitary matrix (2^n, 2^n) implementing CNOT on given indices.
    """
    zero = xp.array([[1], [0]])
    one = xp.array([[0], [1]])
    I = xp.eye(2)
    X = xp.array([[0, 1], [1, 0]])

    control0 = [I] * n
    control0[control] = xp.outer(zero, zero)
    # control0[target] = I # default already
    control0 = xp.kron_chain(control0)

    control1 = [I] * n
    control1[control] = xp.outer(one, one)
    control1[target] = X
    control1 = xp.kron_chain(control1)

    return control0 + control1


def build_cnot_circuit_slow(n):
    """
    Construct parallel CNOT circuit using matrix multiplication of expanded gates.
    |c0>---*---------
    |c1>---|-*-------
    |c2>---|-|-*-----
    ...
    |t0>---X-|-|-----
    |t1>-----X-|-----
    |t2>-------X-----
    ...


    Parameters
    ----------
    n : int
        Number of control-target CNOT pairs.

    Returns
    -------
    U : ndarray
        Unitary matrix of shape (2^(2n), 2^(2n)).
    """
    ops = []
    for i in range(n):
        ops.append(get_cnot_op(i, i + n, 2 * n))
    return xp.matmul_chain(ops)


def build_cnot_circuit_slow2(n):
    """
    Construct parallel CNOT circuit using qubit swaps and kron-product gates.
    |c0>---*---------
    |c1>---|-*-------
    |c2>---|-|-*-...-
    ..
    |t0>---X-|-|-----
    |t1>-----X-|-----
    |t2>-------X-----
    ...


    Optimized variant: uses one full permutation (via bubble sort) before and after CNOTs.

    Parameters
    ----------
    n : int
        Number of control-target CNOT pairs.

    Returns
    -------
    U : ndarray
        Unitary matrix (2^(2n), 2^(2n)).
    """
    # option 2
    # we compute swaps so that each qubit control and target are neighbors
    # than kron all CNOT gates
    # than unswap! swaps can be done efficiently by column swapping!

    # create qubit order for swaps:
    # we want to have c0,t0,c1,t1,c2,t2,...,cn,tn
    # we start with c0, c1, c2, ..., cn, t0, t1, t2, ..., tn
    # so we enumerate control qubits with even and target qubits with odd numbers and sort:
    # first n qubits are all control qubits
    # second n qubits are all target qubits

    qubit_order = [2 * i for i in range(n)] + [2 * i + 1 for i in range(n)]
    _, swaps = bubblesort(qubit_order)
    # swaps are a list of (i,i+1) tuples, where i and i+1 the qubits to be swapped
    # starting with the biggest indices!
    # i.e. apply U to 1,2 of 3 qubit system 0,1,2 will swapt such that
    # selected qubit indices are at beginning- > perm = (1,2,0)
    # and thus to get from (0,1,2) to (1,2,0) swaps =  (0,1) then (1,2)
    # but bubble sort will return the swaps in reverse order (from the POV of U)
    # so swaps = (1,2) then (0,1)
    # 0--\/----|U|
    # 1--/\-\/-|_|
    # 2-----/\----
    # and thus we apply the swap 1,2 first onto U then 1,0
    # this is the order in which bubble sort returns the swaps!
    # these swaps transform (120) ->(012)

    # with given qubit order we will swap such that
    # |c0> ---┌───────┐---|c0>---┌───────┐
    # |c1> ---|       |---|t0>---|       |
    # |c2> ---|       |---|c1>---|       |
    # |cn> ---| SWAPs |---|t1>---| CNOTs |
    # |t0> ---|       |---|c2>---|       |
    # |t1> ---|       |---|t2>---|       |
    # |t2> ---|       |---|cn>---|       |
    # |tn> ---└───────┘---|tn>---└───────┘

    CNOT = xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    ops = [CNOT] * n
    allCNOTS = xp.kron_chain(ops)

    ## however using apply_swap_op does part of U and U^\dag in one step
    for swap_idcs in swaps[::-1]:
        # apply swap op to allCNOTS: S allCNOTS S^\dagger
        col_swapped = perform_col_swap(swap_idcs, 2 * n, allCNOTS)
        row_swapped = perform_col_swap(swap_idcs, 2 * n, col_swapped.T).T
        allCNOTS = row_swapped

    return allCNOTS


#########################################################################
# helper functions
#########################################################################


def bubblesort(seq, reverse=False):
    """
    Sort a sequence with bubblesort and return the adjacent swap operations needed.

    The returned swaps list will transform seq -> sorted seq
    if one applies (swaps[0] (swaps[1] ... (swaps[-2] (swaps[-1] seq)))...))
    This order is compatible with matmul chain and single swap matrices
    which are applied from the left to right. If we actually execute the swaps then
    the last/right most operation = swaps[-1] needs to be performed first

    Parameters
    ----------
    seq : list
        Sequence of sortable elements.
    reverse : bool, optional
        If True, sort in descending order.

    Returns
    -------
    sorted_seq : list
        Sorted version of the input sequence.
    swaps : list of tuple
        List of (i, i+1) index swaps representing the sorting path.
    """

    # Copy the input sequence
    seq = seq.copy()

    swaps = []
    # Get the length of the sequence
    n = len(seq)

    # Loop over the sequence
    for i in range(n):
        # Loop over the sequence
        did_swap = False
        for j in range(0, n - i - 1):
            # Swap the elements if they are in the wrong order
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                did_swap = True
                swaps.append((j, j + 1))

        if not did_swap:
            break

    if reverse:
        seq = seq[::-1]
    return seq, swaps


def perform_col_swap(swap_idcs, n, state):
    """
    Perform a column-wise qubit swap operation on a matrix reshaped as an MPO.

    Parameters
    ----------
    swap_idcs : list of int
        Qubit indices to be swapped (interpreted as (a, b)).
    n : int
        Total number of qubits.
    state : ndarray
        Operator matrix of shape (2^n, 2^n).

    Returns
    -------
    swapped : ndarray
        Matrix with swapped qubit columns.
    """
    mpo = state.reshape([2] * (2 * n))  # reshape to 2x2 tensor for each qubit
    # idc order is sysA, sysB, sysC, ... sysA', sysB', sysC', ...

    mapping = {k: v for k, v in zip(swap_idcs, swap_idcs[::-1])}  # map to swap indices

    mpo_swapped_idcs = [
        mapping.get(i, i) for i in range(2 * n)
    ]  # all indices are swapped or stay the same (default of get())
    mpo_swapped = xp.transpose(mpo, mpo_swapped_idcs)  # swap the indices
    mpo_swapped = mpo_swapped.reshape([2**n, 2**n])  # back to global idcs
    return mpo_swapped


#########################################################################
# functions needed for debugging only
#########################################################################


def construct_full_swap_op_fast(swap_idcs, n):
    """
    Construct the full SWAP operator corresponding to given qubit index swap.

    Parameters
    ----------
    swap_idcs : list of int
        Indices of qubits to swap.
    n : int
        Total number of qubits.

    Returns
    -------
    swap_op : ndarray
        SWAP operator as a (2^n, 2^n) matrix.
    """
    mpo = xp.eye(2**n)
    mpo_swapped = perform_col_swap(swap_idcs, n, mpo)
    return mpo_swapped


def construct_full_swap_seq_op_fast(swap_seq, n):
    """
    This operator should be used for right-multiplying on a density matrix.
    To apply from the left, reverse the swap order.

    Parameters
    ----------
    swap_seq : list of tuple
        List of (i, i+1) swaps to apply in order.
    n : int
        Total number of qubits.

    Returns
    -------
    swap_op : ndarray
        Combined SWAP operator as a (2^n, 2^n) matrix.
    """
    mpos = [construct_full_swap_op_fast(swap_idcs, n) for swap_idcs in swap_seq]
    return xp.matmul_chain(mpos)


if __name__ == "__main__":
    from time import time

    print("test only cnot section of bell circuit")

    for n in range(1, 7):
        t_fast = []
        t_slow = []
        t_slow2 = []
        for _ in range(3):
            t0 = time()
            cnot_parallel = build_cnot_circuit(n)
            t1 = time()
            cnot_slow = build_cnot_circuit_slow(n)
            t2 = time()
            cnot_slow2 = build_cnot_circuit_slow2(n)
            t3 = time()

            t_fast.append(t1 - t0)
            t_slow.append(t2 - t1)
            t_slow2.append(t3 - t2)

        print(
            f"n={n}, time\tbuild_cnot_circuit=\t\t{xp.mean(t_fast):.3e}s +- {xp.std(t_fast):.3e}s"
        )
        print(
            f"n={n}, time\tbuild_cnot_circuit_slow=\t{xp.mean(t_slow):.3e}s +- {xp.std(t_slow):.3e}s"
        )
        print(
            f"n={n}, time\tbuild_cnot_circuit_slow2=\t{xp.mean(t_slow2):.3e}s +- {xp.std(t_slow2):.3e}s"
        )

        print(f"n={n}, is same=\t{xp.allclose(cnot_parallel, cnot_slow)=}")
        print(f"n={n}, is same=\t{xp.allclose(cnot_parallel, cnot_slow2)=}")
        print()

    # build_cnot_circuit() is very similar to build_cnot_circuit_slow2()
    # but instead of swapping multiple times as in slow2, we do one permutation
    # this avoids costly single-thread memory copies of the big matrix
    # these copies can be triggered with transpose and reshape when the resulting view
    # becomes non-contiguous
    # build_cnot_circuit_slow() is fundamentally different approach,
    # which reates the single CNOT gates in the larger hilbert space
    # and multiplies them together, this matmul uses many cores
    # and thus is even faster then single thread memory copy in slow2
