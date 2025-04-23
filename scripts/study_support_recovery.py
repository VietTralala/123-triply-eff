import os

# testing single core
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import numpy as np
import pickle
import numba
from tqdm.auto import tqdm
import time
from pathlib import Path


from src.bell_sampling import (
    bell_measurement_probs,
    compute_pauli_labels,
    sample_bell_measurement_binArr,
)
from src.dataloader.states import read_density_matrices
import ipdb

"""
[Step 1] Studies the recovery of the Pauli vector support using Bell sampling.

This script performs numerical experiments to determine the number of Bell measurements
required to accurately estimate the support of a quantum state's Pauli vector (`target_uP`)
within a given tolerance (`eps`). Support recovery is considered successful
when the Jaccard similarity between the estimated support and the true support exceeds a
predefined threshold.

It uses an incremental sampling approach: starting with a small number of samples,
it iteratively doubles the sample size and checks for recovery until the threshold is met
or a maximum sample limit is reached.

The script processes multiple quantum states (loaded from `--datadir`) for various
epsilon values (`--eps`) and saves the results, including the number of samples needed per
trial and the estimated Pauli vectors, into separate directories under `--savedir`.

Functions:
    count_nonzero_uP: Counts non-zero elements in a Pauli vector above a threshold.
    find_smallest_N_approx_incremental: Finds the minimum samples needed for support recovery via incremental sampling.
    update_accum_sign_and_check: Processes a block of measurements and checks Jaccard similarity.
    run_job: Runs the support recovery experiment for a single state and saves results.
    job_is_done: Checks if results for a specific job already exist.
    job_is_done_from_args: Checks job completion based on arguments.
    parse_arguments: Parses command-line arguments.

Usage:
    Run with `python -m scripts.study_support_recovery --n <n_values> --eps <eps_values> [--datadir <dir>] [--specialstates] [--savedir <dir>]`
"""


def count_nonzero_uP(up, eps):
    assert isinstance(up, np.ndarray)
    return np.sum(np.abs(up) >= eps)


def find_smallest_N_approx_incremental(
    n,
    rho,
    target_uP,
    eps,
    row_idx,
    recovery_threshold=0.90,
    ntrials=10,
    limitN=15_000_000,
):
    """
    Run multiple trials to find the smallest number of Bell measurements
    needed to recover the support of `target_uP` with given accuracy.

    Parameters
    ----------
    n : int
        Number of qubits.
    rho : ndarray
        Density matrix of the quantum state.
    target_uP : ndarray
        Target Pauli vector.
    eps : float
        Threshold to determine significant components.
    row_idx : int
        Index used for seeding randomness.
    recovery_threshold : float, optional
        Jaccard similarity threshold for declaring support recovery.
    ntrials : int, optional
        Number of independent trials to run.
    limitN : int, optional
        Maximum number of samples to allow.

    Returns
    -------
    result : dict
        Trial metadata and support recovery statistics.
    measurements : list of ndarray
        Recorded measurements up to first successful support recovery (or limit).
    """

    t0 = time.time()
    meas_probs = bell_measurement_probs(rho)
    print(
        f"Calculating bell_measurement_probs for {n=} took {time.time() - t0:.2f} seconds"
    )

    card_S = count_nonzero_uP(target_uP, eps)

    num_measurements = []
    all_measurements = []
    all_preds = []
    all_Js = []
    sign_table = np.array(
        [
            [1, 1, 1, 1],  # I
            [1, 1, -1, -1],  # X
            [-1, 1, 1, -1],  # Y
            [1, -1, 1, -1],  # Z
        ],
        dtype=np.int8,
    )
    pauli_labels = compute_pauli_labels(n)

    for trial_idx in range(ntrials):
        print(f"Starting trial {trial_idx} for {n=}, {eps=}, {row_idx=}")
        t0 = time.time()
        seed_value = row_idx * 1000 + trial_idx
        np.random.seed(seed_value)

        # We'll do an incremental doubling approach:
        if trial_idx == 0 or num_measurements[-1] < 0 or np.isnan(num_measurements[-1]):
            block_size = 128
        else:
            block_size = num_measurements[-1]

        total_samples = 0
        accum_sign = np.zeros(4**n, dtype=np.float64)

        found_N = -1
        pred = None
        measurements_for_this_trial = []
        Js = []

        while found_N < 0 and total_samples < limitN:
            # Sample new block
            new_measurement_block = sample_bell_measurement_binArr(
                rho, num_samples=block_size, probs=meas_probs
            )
            measurements_for_this_trial.append(new_measurement_block)

            # Update partial sums and check
            M_candidate, pred, Js_chunk = update_accum_sign_and_check(
                new_measurement_block,
                accum_sign,
                pauli_labels=pauli_labels,
                sign_table=sign_table,
                target=target_uP,
                eps=eps,
                recovery_threshold=recovery_threshold,
                start_count=total_samples,
            )
            Js.extend(Js_chunk)

            total_samples += block_size

            # If we found an M in this block that meets threshold
            if M_candidate > 0:
                found_N = M_candidate
                break

            # If not found, double block_size for the next iteration
            block_size *= 2
            # print(f"Total samples: {total_samples}, block_size: {block_size}")

        # Store results for the trial
        if found_N < 0:
            # Means even up to limitN we didn’t meet the threshold
            num_measurements.append(np.nan)
        else:
            # found_N is the first sample count crossing threshold
            num_measurements.append(found_N)

        all_preds.append(pred)
        all_Js.append(Js)

        # Collect the actual measurements up to found_N
        # (or up to limitN if not found)
        needed = found_N if found_N > 0 else limitN
        measurements_concat = np.concatenate(measurements_for_this_trial, axis=0)
        all_measurements.append(measurements_concat[:needed].copy())

        print(
            f"Trial {trial_idx} for {n=}, {eps=}, {row_idx=} {needed} took {time.time() - t0:.2f} seconds"
        )

    res = {
        "n": n,
        "eps": eps,
        "ridx": row_idx,
        "samples_needed": num_measurements,
        # "jaccard_seq": all_Js, # takes a lot of space in pkl file
        "preds": all_preds,
        "|S|": card_S,
        "recovery_threshold": recovery_threshold,
    }
    return res, all_measurements


@numba.njit
def update_accum_sign_and_check(
    meas_chunk,  # shape (block_size, 2*n)
    accum_sign,  # shape (4^n,)
    pauli_labels,  # shape (n, 4^n)
    sign_table,  # shape (4,4)
    target,  # shape (4^n,)
    eps,
    recovery_threshold,
    start_count,
):
    """
    Process a block of Bell measurement outcomes and update Pauli sign estimates.

    For each new sample, update the accumulated sign vector `accum_sign`
    and compute the predicted Pauli vector. Check whether the Jaccard similarity
    with the target vector exceeds the recovery threshold.

    Parameters
    ----------
    meas_chunk : ndarray
        Measurement outcomes of shape (block_size, 2n), each row is a Bell outcome.
    accum_sign : ndarray
        Accumulated sign vector (updated in-place), shape (4^n,).
    pauli_labels : ndarray
        Pauli label table of shape (n, 4^n).
    sign_table : ndarray
        Lookup table of shape (4, 4) encoding sign contributions for each outcome.
    target : ndarray
        Target Pauli vector to recover.
    eps : float
        Threshold for determining non-zero components.
    recovery_threshold : float
        Required Jaccard similarity for successful recovery.
    start_count : int
        Number of samples already processed before this block.

    Returns
    -------
    found_M : int
        Sample index (absolute) at which threshold was reached, or -1 if not reached.
    final_pred : ndarray
        Final predicted Pauli vector after processing the block.
    Js : list of float
        Jaccard similarity at each incremental step within the block.
    """

    num_samples, two_n = meas_chunk.shape
    n = two_n // 2
    four_n = pauli_labels.shape[1]
    found_M = -1
    found_M_largest = -1

    # Convert measurements -> array bidxs of shape (block_size, n)
    bidx_chunk = 2 * meas_chunk[:, :n] + meas_chunk[:, n:]

    Js = []

    # Process each new sample incrementally
    for i in range(num_samples):
        # Build sign_vec for the i-th new outcome
        sign_vec = np.ones(four_n, dtype=np.int8)
        for q in range(n):
            outcome = bidx_chunk[i, q]
            for j in range(four_n):
                pauli = pauli_labels[q, j]
                sign_vec[j] *= sign_table[pauli, outcome]

        # Update the accumulator in-place
        accum_sign += sign_vec

        # The total number of processed samples so far
        M = start_count + i + 1

        # Compute the predicted magnitudes
        pred_sq = accum_sign / M
        pred = np.sqrt(np.clip(pred_sq, 0.0, 1.0))  # shape (4^n,)

        # Jaccard check
        target_above_eps = np.abs(target) >= eps
        pred_above_eps = np.abs(pred) >= eps
        intersection = np.sum(target_above_eps & pred_above_eps)
        union = np.sum(target_above_eps | pred_above_eps)
        J = intersection / union if union != 0 else 1.0
        Js.append(J)

        if J >= recovery_threshold:
            found_M = M
            return found_M, pred, Js

    return -1, pred, Js


# Run jobs in parallel
def run_job(
    job_args,
    root_save_dir="./data/results/",
    recovery_threshold=0.90,
    ntrials=10,
    limitN=15_000_000,
    job_folder_name="",
):
    """
    Run a support recovery experiment and save results if not already done.

    Parameters
    ----------
    job_args : tuple
        Tuple (n, eps, ridx, fullrec) describing the job.
    root_save_dir : str, optional
        Directory where results are saved.
    recovery_threshold : float, optional
        Jaccard similarity threshold for recovery.
    ntrials : int, optional
        Number of trials per job.
    limitN : int, optional
        Maximum number of samples to draw.
    job_folder_name : str, optional
        Explicit folder name. If empty, it is auto-generated.

    Returns
    -------
    result_dict : dict or None
        Result dictionary if job runs; None if job is skipped or errors occur.
    """
    n, eps, ridx, fullrec = job_args
    rho = fullrec["rho"]
    uP = fullrec["up"]  # pauli vector

    # Create a descriptive subfolder for this job:
    # use n, eps, seed, and nt to identify the job
    if not job_folder_name:
        if fullrec["type"] in ("ghz", "zero"):
            # For special states, use the type field
            job_folder_name = f"n{n}_eps{eps}_type={fullrec['type']}"
        else:
            # For gibbs states
            job_folder_name = f"n{n}_eps{eps}_seed{fullrec['seed']}_nt{fullrec['nt']}"

    job_dir = os.path.join(root_save_dir, job_folder_name)

    if job_is_done(job_folder_name, root_save_dir):
        return None  # Skip the job

    try:
        result_dict, measurements = find_smallest_N_approx_incremental(
            n=n,
            rho=rho,
            target_uP=uP,
            eps=eps,
            row_idx=ridx,
            recovery_threshold=recovery_threshold,
            ntrials=ntrials,
            limitN=limitN,
        )
    except Exception as e:
        print(f"Error for {job_folder_name}: {e}. Didn't save results.")
        # return none so that other jobs can be started
        return None

    date_time_str = time.strftime("%Y%m%d-%H%M%S")

    Path(job_dir).mkdir(parents=True, exist_ok=True)

    result_path = os.path.join(job_dir, f"results_{date_time_str}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(result_dict, f)

    meas_path = os.path.join(job_dir, f"measurements_{date_time_str}.pkl")
    with open(meas_path, "wb") as f:
        pickle.dump(measurements, f)

    return result_dict


def job_is_done(job_folder_name: str, root_save_dir: str):
    """
    Check if both result and measurement files already exist for a job.

    Parameters
    ----------
    job_folder_name : str
        Name of the job subfolder.
    root_save_dir : str
        Root directory for job outputs.

    Returns
    -------
    done : bool
        True if job is complete, False otherwise.
    """
    job_dir = Path(root_save_dir) / job_folder_name

    if any(job_dir.glob("measurements_*.pkl")) and any(job_dir.glob("results_*.pkl")):
        print(f"Skipping existing job: {job_folder_name}")
        return True  # Skip the job
    return False


def job_is_done_from_args(job_args, root_save_dir):
    """
    Determine if a job has already been completed, using job_args.

    Parameters
    ----------
    job_args : tuple
        Job descriptor (n, eps, ridx, fullrec).
    root_save_dir : str
        Directory containing job outputs.

    Returns
    -------
    done : bool
        True if results and measurements exist, False otherwise.
    """

    n, eps, ridx, fullrec = job_args
    if fullrec["type"] in ("ghz", "zero"):
        job_folder_name = f"n{n}_eps{eps}_type={fullrec['type']}"
    else:
        job_folder_name = f"n{n}_eps{eps}_seed{fullrec['seed']}_nt{fullrec['nt']}"
    return job_is_done(job_folder_name, root_save_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="[Step 1] support recovery")

    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[
            2,
            3,
            4,
            # 5,
            # 6,
            # 7,
        ],
        help="List of n values to test.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[
            0.05,
            # 0.07,
            0.11,
            # 0.16,
            # 0.23,
            0.34,
            # 0.5,
        ],
        help="List of epsilon values to test.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per configuration.",
    )
    parser.add_argument(
        "--recov",
        type=float,
        default=0.90,
        help="Threshold for recovery.",
    )
    parser.add_argument(
        "--limitN",
        type=int,
        default=15_000_000,
        help="Maximum number of measurements allowed.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="approx_recovery_jacc",
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="normalized_gibbs",
        help="Directory where test states are loaded from.",
    )
    parser.add_argument(
        "--specialstates",
        action="store_true",
        help="Flag to indicate special states (GHZ/zero) with different filename pattern.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from src.tqdm_joblib import tqdm_joblib

    """ find_smallest_N() find the smallest number of measurements that approximately recover the supoort of u_P
    that is it finds N such that for almost all P for which |tr(P rho)| > eps we have  predictions |u_P| > eps
    almost can be more precisely defined
    lets say T(rho, eps) = {P: |tr(P rho)| > eps}
    and S = {P: |u_P| > eps}
    then we want to find N such that |T ∩ S| / |T| > threshold (0.9)
    """

    args = parse_arguments()

    # Assign parsed arguments to variables
    n_values = args.n
    eps_values = args.eps
    n_trials = args.trials
    recovery_threshold = args.recov
    limitN = args.limitN
    save_dir = args.savedir
    data_dir = args.datadir
    print(f"Running with arguments: {args}")

    project_dir = os.getcwd()
    root_save_dir = os.path.join(project_dir, "data/results/", save_dir)

    Path(root_save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {root_save_dir}")

    all_data_dir = os.path.join(project_dir, "data", data_dir)
    print(f"Reading data from {all_data_dir}")

    date_time_str = time.strftime("%Y%m%d-%H%M%S")

    jobs = []
    for n in n_values:
        recs = read_density_matrices(
            n,
            save_dir=all_data_dir,
            read_up_as_dict=False,
            special=args.specialstates,
        )
        # print(f"Found {len(recs)} records for {n=}")

        for eps in eps_values:
            for ridx, rec in enumerate(recs):
                jobs.append((rec["n"], eps, ridx, rec))

    print(f"Found {len(jobs)} jobs")

    jobs = [j for j in jobs if not job_is_done_from_args(j, root_save_dir)]

    print(f"Remaining jobs to run: {len(jobs)}")

    print(f"starting jobs for {eps_values=}, {n_values=}, {date_time_str=}")

    with tqdm_joblib(tqdm(desc="searching N", total=len(jobs))) as pbar:
        results = Parallel(n_jobs=100)(
            delayed(run_job)(
                j,
                root_save_dir=root_save_dir,
                recovery_threshold=recovery_threshold,
                ntrials=n_trials,
                limitN=limitN,
            )
            for j in jobs
        )

    # # serial runs for debug
    # results = []
    # for j in tqdm(jobs):
    #     results.append(
    #         run_job(
    #             j,
    #             root_save_dir=root_save_dir,
    #             recovery_threshold=recovery_threshold,
    #             ntrials=n_trials,
    #             limitN=limitN,
    #         )
    #     )
