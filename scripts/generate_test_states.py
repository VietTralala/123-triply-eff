import os
import numpy as np
from tqdm import tqdm
import numba


import pickle
from joblib import Memory


import ipdb

memory = Memory("/tmp/triply_cache", verbose=0)

from src.np_backend import gibbs_state
from src.paulis import calc_pauli_vec, generate_all_Ps_stacked
from src.quantum_states import create_ghz_density_matrix, create_zero_density_matrix

"""
Generates and saves test quantum states for numerical experiments.

This script creates and stores various types of quantum states, including:
- Gibbs states with varying support sizes derived from random Hamiltonians.
- Special states like Greenberger–Horne–Zeilinger (GHZ) and all-zero states.

The generated states are saved as pickle files, each containing the density matrix (`rho`),
the Pauli terms defining the Hamiltonian (for Gibbs states) or an empty list (for special states),
and the corresponding Pauli expectation vector (`up`). These files are used as input
for subsequent analysis scripts in the project workflow.

Functions:
    generate_rho: Generates a single random Gibbs state.
    save_density_matrices: Generates and saves a collection of Gibbs states.
    save_special_states: Saves GHZ and zero states.

Usage:
    Run this script directly to generate states for default qubit numbers (n=2 to 5).
    Modify the main loop to generate states for different qubit numbers.
"""


def generate_rho(
    n,
    numterm,
    seed,
    samplePauli_replace=False,
):
    """
    Generate a random Gibbs state on `n` qubits using `numterm` Pauli terms.

    Parameters
    ----------
    n : int
        Number of qubits.
    numterm : int
        Number of Pauli terms to include in the Hamiltonian.
    seed : int
        Random seed for reproducibility.
    samplePauli_replace : bool, optional
        Whether to sample Pauli terms with replacement.

    Returns
    -------
    rho : ndarray
        Generated Gibbs state as a density matrix.
    selected_labels : list of str
        Sorted list of Pauli labels used to construct the state.
    """

    np.random.seed(seed)
    labels, allPs_stacked = generate_all_Ps_stacked(n)  # 4**n x 2**n x 2**n
    selected_paulis = np.random.choice(
        allPs_stacked.shape[0],
        size=numterm,
        replace=samplePauli_replace,
    )

    H = np.sum(allPs_stacked[selected_paulis], axis=0)
    H /= np.linalg.norm(H, ord=2)  # spectral norm, largest sing. value

    rho = gibbs_state(H)

    return rho, sorted(labels[selected_paulis])


def save_density_matrices(
    n,
    save_dir=f"./data/normalized_gibbs/",
):
    """
    Generate and save a collection of Gibbs states with increasing support sizes.
    Save format is (rho, terms, up), where `terms` is a list of Pauli labels
    and `up` is the Pauli vector.

    Parameters
    ----------
    n : int
        Number of qubits.
    save_dir : str, optional
        Directory to save the generated density matrices.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_terms_max = {
        2: 8,
        3: 32,
        4: 128,
        5: 256,
        6: 512,
        7: 1024,
    }

    sample_sizes = {
        2: 100,
        3: 100,
        4: 100,
        5: 100,
        6: 100,
        7: 100,
    }

    if n not in num_terms_max:
        raise ValueError(f"{n=} not supported")

    np.random.seed(0)

    num_terms = np.round(
        np.logspace(
            np.log10(1),
            np.log10(num_terms_max[n]),
            num=sample_sizes[n],
        )  # default base of logspace is 10
    ).astype(int)

    num_terms = np.sort(num_terms)

    _, allPs_stacked = generate_all_Ps_stacked(n)
    seen_terms = set()

    for nt_idx, nt in tqdm(enumerate(num_terms), total=len(num_terms)):
        found_unique = False
        seed_count = 0
        while not found_unique and seed_count < 10:
            seed = nt_idx * 10000 + seed_count
            rho, terms = generate_rho(
                n,
                nt,
                seed=seed,
            )
            comb_terms = "".join(sorted(terms))
            if comb_terms in seen_terms:
                seed_count += 1
                continue
            seen_terms.add(comb_terms)
            found_unique = True

        if not found_unique:
            # print(f"Skipping {nt=}, {n=}")
            continue

        up = calc_pauli_vec(rho, allPs_stacked)

        filename = f"n{n}_nt{nt}_s{seed}.pkl"
        with open(os.path.join(save_dir, filename), "wb") as f:
            pickle.dump((rho, terms, up), f)


def save_special_states(n, save_dir="./data/special_states/"):
    """
    Save predefined GHZ and zero states in the same format as Gibbs samples.
    Format is (rho, terms, up), where `terms` is a placeholder.

    Parameters
    ----------
    n : int
        Number of qubits.
    save_dir : str, optional
        Directory to save the special states.
    """
    os.makedirs(save_dir, exist_ok=True)
    _, allPs_stacked = generate_all_Ps_stacked(n)

    states = {
        "ghz": create_ghz_density_matrix,
        "zero": create_zero_density_matrix,
    }

    for label, state_fn in states.items():
        rho = state_fn(n)
        up = calc_pauli_vec(rho, allPs_stacked)
        terms_placeholder = []  # GHZ/zero don't have selected terms

        fname = f"n{n}_type{label}_s0.pkl"
        with open(os.path.join(save_dir, fname), "wb") as f:
            pickle.dump((rho, terms_placeholder, up), f)

        print(f"Saved: {fname}")


if __name__ == "__main__":

    for n in range(2, 6):
        print(f"Generating test states for {n=}")
        save_density_matrices(n, save_dir=f"./data/normalized_gibbs/")
        save_special_states(n, save_dir="./data/special_states/")
