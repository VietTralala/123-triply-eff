import argparse
from glob import glob
import os
import pickle
import time

import numba
from tqdm import tqdm
import numpy as np


from scripts.precompute_bell_probs import job_has_mm_done
from src.dataloader.mm_results import load_mm_result
from src.dataloader.states import read_density_matrices
from src.paulis import generate_all_Ps_stacked

from src.bell_sampling import (
    sample_bell_measurement_binArr,
    compute_pauli_labels,
)

import ipdb

"""
[Step 3] Reconstructs Pauli vector signs using Bell sampling and mimicking states.

This script evaluates the sign reconstruction phase of the shadow tomography protocol.
It uses the original test state (`rho`), the corresponding mimicking state (`sigma` from Step 2),
and the precomputed Bell measurement probabilities (from Step 2.5).

For each trial from the mimicking state construction, it simulates Bell measurements
using the precomputed probabilities. It incrementally increases the number of samples (`N2`),
estimating the signs of the significant Pauli components identified in Step 1.

The goal is to find the minimum number of samples (`N2`) needed for the estimated signs
to match the true signs (derived from `target_up`) for a specified fraction (`--recov`)
of the significant Pauli components.

Results, including the number of samples needed (`samples_needed`), the final signed Pauli
vector estimate (`signed_preds`), and various metrics (HS norm, sign agreement ratio),
are saved to `stage3_results_*.pkl` files in the respective job directories under `--savedir`.

Functions:
    parse_arguments: Parses command-line arguments.
    job_precomp_done: Checks if Bell probability precomputation (Step 2.5) is done.
    job_has_stages12: Checks if results from Steps 1 and 2 exist.
    job_has_stage3: Checks if Stage 3 results already exist.
    run_job: Runs the Stage 3 evaluation for a single job directory.
    find_smallest_N_for_sign_incremental: Finds minimum samples for sign recovery via incremental sampling.
    generate_u_P_exact_numba: Computes the exact Pauli vector for a state.
    MSE: Calculates Mean Squared Error.
    update_accum_sign_and_check: Processes a block of measurements, estimates signs, and checks recovery.

Usage:
    Run with `python -m scripts.study_sign_reconstruction --n <n_values> --eps <eps_values> [--trials <num>] [--recov <thresh>] [--limitN <max_samples>] [--datadir <dir>] [--specialstates] [--savedir <dir>]`
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="[Step 3] sign reconstruction")

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
        default=5,
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
        default=70_000,
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


def job_precomp_done(job_dir):
    """
    Check if Bell measurement probabilities were precomputed for a given job.

    Returns True if:
    - Both v1 and v2 mimicking results are present.
    - `precomputed_bell_probs` column exists in the DataFrame.
    - At least one trial has valid Bell probs.

    Parameters
    ----------
    job_dir : str
        Path to job directory.

    Returns
    -------
    bool
        Whether precomputation is done.
    """
    # this version is slightly different from the one in precompute_bell_probs.py
    # here we consider partially filled bell_probs as done!
    # that enables to still process jobdir if there are failed trials of N1 search
    mm_files = glob(os.path.join(job_dir, "mimicking_results_*.pkl"))

    assert len(mm_files) == 2, f"Expected 2 file, got {len(mm_files)}"

    # load mm resultsfile
    records = load_mm_result(
        job_dir, algo="both", load_all=True
    )  # dataframe with one row per trial

    # check for precomputed_bell_probs col in df
    if "precomputed_bell_probs" not in records.columns:
        return False

    # if all trials have failed N1 search then we skip this jobdir
    # if there are still some trials with N1 search done, we keep the jobdir and consider
    # precomputation done so that we can continue with stage 3
    if records["precomputed_bell_probs"].isna().all():
        return False

    return True


def job_has_stages12(job_dir):
    """
    Check if Stage 1 (sampling) and Stage 2 (mimicking state construction) are complete.

    Required files:
    - 2 mimicking results files
    - 1 results file
    - 1 measurements file
    - No FAILED.flag

    Parameters
    ----------
    job_dir : str

    Returns
    -------
    bool
        True if the job is eligible for Stage 3.
    """
    has_mm = len(glob(os.path.join(job_dir, "mimicking_results_*.pkl"))) == 2
    has_res = len(glob(os.path.join(job_dir, "results_*.pkl"))) == 1
    has_meas = len(glob(os.path.join(job_dir, "measurements_*.pkl"))) == 1
    has_flags = len(glob(os.path.join(job_dir, "FAILED.flag"))) == 1
    if has_mm and has_res and has_meas and not has_flags:
        return True

    print(
        f" *** Skipping {job_dir}: Did not found all necessary files:",
        f"{has_mm=}, {has_res=}, {has_meas=}, {has_flags=}***",
    )
    return False


def job_has_stage3(job_dir):
    """
    Check if Stage 3 (sign-based recovery test) has already been completed.

    Parameters
    ----------
    job_dir : str

    Returns
    -------
    bool
        True if stage3_results_*.pkl exists in job_dir.
    """
    stage3_results = glob(os.path.join(job_dir, "stage3_results_*.pkl"))

    if len(stage3_results) == 1:
        return True

    return False


def run_job(
    jobdir,
    state_info,
    n,
    eps_sampling,
    allPs,
    ntrials=5,
    limitN=70_000,
    threshold=0.9,
):
    """
    Run Stage 3 evaluation for one job directory using a given state and precomputed Bell probs.

    Parameters
    ----------
    jobdir : str
        Job directory path.
    state_info : dict
        Contains 'rho', 'up', 'seed', 'nt', etc. for this state.
    n : int
        Number of qubits.
    eps_sampling : float
        Threshold for significance.
    allPs : ndarray
        Precomputed Pauli stack of shape (4^n, 2^n, 2^n).
    ntrials : int
        Number of sampling trials.
    limitN : int
        Max number of samples per trial.
    threshold : float
        Sign recovery agreement threshold.

    Saves
    -----
    stage3_results_*.pkl : dict
        Mapping rowid -> recovery result summary.
    """
    # gather relevant info!
    n = state_info["n"]
    rho = state_info["rho"]

    target_up = state_info["up"]
    nt = state_info["nt"]
    state_seed = state_info["seed"]
    root_save_dir = os.path.dirname(jobdir)  # where jobdir is located
    jobdir_folder_only = os.path.basename(
        jobdir
    )  # contains n, eps, seed, nt/state type

    if not job_has_stages12(jobdir):
        print(f"Skipping {jobdir} since it does not have all necessary files.")
        return None  # Skip the job

    # load mm state
    df_mm = load_mm_result(jobdir, load_all=True)
    if df_mm is None or df_mm.empty:
        print(f"Skipping {jobdir} since no mimicking state data available")
        return

    if "precomputed_bell_probs" not in df_mm.columns:
        print(f"Skipping {jobdir} since no precomputed bell probs available")
        return

    # print(df_mm.columns) =
    # ['algo_type', 'final_state', 'n', 'epsilon_mm', 'has_converged', 'trial',
    #    'N', 'pred', '|S|', 'epsilon_sampling', 'seed_mm', 'nt', 'jobdir',
    #    'total_overshoot', 'failed_due_to_substeps', 'precomputed_bell_probs']

    # Container for stage-3 results
    all_stage3_results = {}
    for row_idx, row in df_mm.iterrows():
        N1 = row["N"]
        tidx = row["trial"]  # trial index from N1
        algo_type = row["algo_type"]
        rowid = f"{jobdir_folder_only}_trial{tidx}_algo{algo_type}"  # IMPORTANT format
        if np.isnan(N1) or N1 is None or N1 == -1:
            print(f"Skipping {jobdir}, {rowid} since N1 is NaN")
            # this rowid will stay empty when merging, dont need to add dummies
            continue
        sigma = row["final_state"]

        up = row["pred"]  # is thresholded already with eps_sampling
        significant_pauli_idcs_pred = np.where(up >= eps_sampling)[0]

        bell_meas_probs = row["precomputed_bell_probs"]  # dict {basis_label: probs}

        if any(np.isnan(list(bell_meas_probs.values()))):
            print(f"Skipping {jobdir}, {rowid} since bell meas probs has NaN")
            continue

        seed_mm = row["seed_mm"]

        result_dict = find_smallest_N_for_sign_incremental(
            n,
            rho,
            sigma,
            bell_meas_probs,
            significant_pauli_idcs_pred,
            allPs,
            up,
            target_up,
            seed_mm,
            eps_sampling,
            rowid=rowid,
            threshold=threshold,
            ntrials=ntrials,
            limitN=limitN,
        )
        # result_dict.keys() = ["rowid", "samples_needed", "signed_preds",
        # "significant_pauli_idcs", "hs_norm_threshold", "hs_norm",
        # "thresholded_target", "target"]

        # Store the stage-3 results for each row (mm trial)
        all_stage3_results[rowid] = result_dict

    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    results_out_file = os.path.join(jobdir, f"stage3_results_{date_time_str}.pkl")

    with open(results_out_file, "wb") as f:
        pickle.dump(all_stage3_results, f)

    print(f"Stage 3 results saved to {results_out_file}")


def find_smallest_N_for_sign_incremental(
    n,
    rho,
    sigma,
    meas_probs,
    significant_pauli_idcs_pred,
    allPs,
    up,  # thresholded magnitude prediction from bell sampling
    target_up,  # not thresholded
    seed_mm,
    eps_sampling,
    rowid,
    threshold=0.9,
    ntrials=5,
    limitN=70_000,
):
    """
    Perform incremental Bell sampling to estimate Pauli signs and evaluate sign agreement.

    Stops early if the sign match with the true Pauli vector exceeds the threshold.

    Returns
    -------
    res : dict
        Contains:
        - rowid : str
        - samples_needed : list[int]
        - signed_preds : list[ndarray]
        - sign_agreement_ratio : list[float]
        - hs_norm : list[float]
        - hs_norm_thresholded : list[float]
        - target, thresholded_target
        - all_measurements : list[ndarray]
    """
    num_measurements = []
    all_measurements = []
    all_signed_preds = []
    all_hs_norms_thresh = []
    all_hs_norms = []
    all_sign_agreement_ratios = []
    sign_table = np.array(
        [
            [1, 1, 1, 1],  # I
            [1, 1, -1, -1],  # X
            [-1, 1, 1, -1],  # Y
            [1, -1, 1, -1],  # Z
        ],
        dtype=np.int8,
    )
    # t0 = time.time()
    trPsigma = generate_u_P_exact_numba(sigma, allPs)
    # print("Generating u_P(sigma) took", time.time() - t0) # takes about 2s for n7

    # # threshold target_up
    significant_pauli_idcs_target = np.where(np.abs(target_up) >= eps_sampling)[0]
    target_up_thresholded = np.zeros_like(target_up)
    target_up_thresholded[significant_pauli_idcs_target] = target_up[
        significant_pauli_idcs_target
    ].copy()

    print(f"Starting {n=}, {seed_mm=}, {rowid=}")
    t0 = time.time()
    for trial_idx in range(ntrials):
        seed_value = seed_mm * 100 + trial_idx
        np.random.seed(seed_value)

        # We'll do an incremental doubling approach:
        if trial_idx == 0 or num_measurements[-1] < 0 or np.isnan(num_measurements[-1]):
            block_size = 128
        else:
            block_size = num_measurements[-1]

        total_samples = 0
        accum_sign = np.zeros(4**n, dtype=np.float64)

        found_N = -1
        measurements_for_this_trial = []

        while found_N < 0 and total_samples < limitN:
            # Sample new block
            new_measurement_block = sample_bell_measurement_binArr(
                rho, num_samples=block_size, probs=meas_probs
            )
            measurements_for_this_trial.append(new_measurement_block)

            # Update partial sums and check
            (
                M_candidate,
                sign,
                signed_pred,
                normalized_hs_norm_thresholded,
                sign_agreement_ratio,
            ) = update_accum_sign_and_check(
                new_measurement_block,
                accum_sign,
                pauli_labels=compute_pauli_labels(n),
                sign_table=sign_table,
                target=target_up_thresholded,
                bell_sampl_up=up,
                trPsigma=trPsigma,
                significant_pauli_idcs_target=significant_pauli_idcs_target,
                significant_pauli_idcs_pred=significant_pauli_idcs_pred,
                threshold=threshold,
                start_count=total_samples,
            )

            total_samples += block_size

            # If we found an M in this block that meets threshold
            if M_candidate > 0:
                found_N = M_candidate
                break

            # If not found, double block_size for the next iteration
            block_size *= 2
            # print(f"Total samples: {total_samples}, block_size: {block_size}")

        normalized_hs_norm = MSE(signed_pred, target_up) * (2**n)  # wo thresh
        # Store results for the trial
        if found_N < 0:
            # Means even up to limitN we didn’t meet the threshold
            num_measurements.append(np.nan)
        else:
            # found_N is the first sample count crossing threshold
            num_measurements.append(found_N)

        # always store the last prediction even if not found N
        all_signed_preds.append(signed_pred)
        all_hs_norms_thresh.append(normalized_hs_norm_thresholded)
        all_hs_norms.append(normalized_hs_norm)
        all_sign_agreement_ratios.append(sign_agreement_ratio)

        # Collect the actual measurements up to found_N
        # (or up to limitN if not found)
        needed = found_N if found_N > 0 else limitN
        measurements_concat = np.concatenate(measurements_for_this_trial, axis=0)
        all_measurements.append(measurements_concat[:needed].copy())

    print(
        (
            f"{ntrials} Trial for {n=}, {seed_mm=}, {rowid=} "
            f"{np.median(num_measurements)=}, hs_norm={np.median(all_hs_norms_thresh)} "
            f"took {time.time() - t0:.2f} seconds"
        )
    )

    res = {
        # "n": n,
        # "eps": eps,
        # "ridx": row_idx, # will be added by rowid
        "rowid": rowid,
        "samples_needed": num_measurements,
        "signed_preds": all_signed_preds,
        "significant_pauli_idcs_target": significant_pauli_idcs_target,
        "significant_pauli_idcs_pred": significant_pauli_idcs_pred,
        "threshold": threshold,
        "sign_agreement_ratio": all_sign_agreement_ratios,
        "hs_norm": all_hs_norms,
        "hs_norm_thresholded": all_hs_norms_thresh,
        "thresholded_target": target_up_thresholded,
        "target": target_up,
        "all_measurements": all_measurements,
    }
    return res


@numba.njit
def generate_u_P_exact_numba(rho, allP_stack):
    """
    Compute the Pauli expectation vector for a density matrix in Bell basis.

    Parameters
    ----------
    rho : ndarray, shape (2^n, 2^n)
    allP_stack : ndarray, shape (4^n, 2^n, 2^n)

    Returns
    -------
    uP_vec : ndarray of shape (4^n,)
    """
    n_ops, d, _ = allP_stack.shape
    uP_vec = np.zeros(n_ops)

    for i in range(n_ops):
        prod = np.dot(allP_stack[i], rho)
        tr = np.trace(prod)
        uP_vec[i] = tr.real

    return uP_vec


@numba.njit
def MSE(pred, target):
    return np.mean((pred - target) ** 2)


@numba.njit
def update_accum_sign_and_check(
    meas_chunk,  # shape (block_size, 2*n)
    accum_sign,  # shape (4^n,)
    pauli_labels,  # shape (n, 4^n)
    sign_table,  # shape (4,4)
    target,  # shape (4^n,)
    bell_sampl_up,  # shape (4^n,) # already thresholded
    trPsigma,  # shape (4^n,) # from mm state
    significant_pauli_idcs_target,
    significant_pauli_idcs_pred,
    threshold,
    start_count,
):
    """
    Update sign accumulator for Bell measurement outcomes and check for recovery threshold.

    Parameters
    ----------
    meas_chunk : ndarray, shape (B, 2n)
        Block of Bell samples.
    accum_sign : ndarray, shape (4^n,)
        Running sum of sign estimates.
    ...
    Returns
    -------
    found_M : int
        Sample count where threshold first crossed, or -1.
    sign : ndarray
        Recovered sign vector.
    signed_pred : ndarray
        Bell estimate with sign applied.
    normalized_hs_norm : float
        2^n * MSE(signed_pred, target).
    sign_agreement : float
        Fraction of matching signs on significant indices.
    """
    num_samples, two_n = meas_chunk.shape
    n = two_n // 2
    four_n = pauli_labels.shape[1]
    twopow_n = 2**n
    found_M = -1

    # Convert measurements -> array bidxs of shape (block_size, n)
    bidx_chunk = 2 * meas_chunk[:, :n] + meas_chunk[:, n:]

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

        # Compute the estimate for tr(P rho)tr(P sigma) for all P
        tr_prho_tr_psig = accum_sign / M  # shape (4^n,)

        # calculate tr(P rho) = tr(P rho)tr(P sigma)/tr(P sigma)
        sign = np.zeros(four_n)

        # use significant_pauli_idcs_pred to avoid division by zero
        sign[significant_pauli_idcs_pred] = np.sign(
            tr_prho_tr_psig[significant_pauli_idcs_pred]
            / trPsigma[significant_pauli_idcs_pred]
        )

        signed_pred = sign * bell_sampl_up

        # measure sign agreement based on target significant paulis
        # If no entries exceed eps, optionally treat as 100% matching
        sign_agreement = 1.0
        if np.any(significant_pauli_idcs_target):
            # Compare sign on those indices only, ignore actual magnitude!
            sign_agreement = np.mean(  # avg over all significant paulis only!
                np.sign(sign[significant_pauli_idcs_target])
                == np.sign(target[significant_pauli_idcs_target])
            )

        if sign_agreement >= threshold:
            found_M = M
            normalized_hs_norm = MSE(signed_pred, target) * twopow_n
            return (found_M, sign, signed_pred, normalized_hs_norm, sign_agreement)

    normalized_hs_norm = MSE(signed_pred, target) * twopow_n

    return -1, sign, signed_pred, normalized_hs_norm, sign_agreement


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from src.tqdm_joblib import tqdm_joblib

    args = parse_arguments()

    # Assign parsed arguments to variables
    n_values = args.n
    eps_values = args.eps
    results_subdir = args.savedir
    data_dir = args.datadir
    print(f"Running with arguments: {args}")

    project_dir = os.getcwd()
    # data directory that stores the density matrices:
    root_data_dir = os.path.join(project_dir, "data")
    # Where job results subdirectories are located:
    root_results_dir = os.path.join(project_dir, "data/results/")

    save_dir = os.path.join(root_results_dir, results_subdir)
    final_data_dir = os.path.join(root_data_dir, data_dir)
    print(f"Loading density matrices from {final_data_dir}")
    print(f"Loading measurements and results from {save_dir}")

    # Load all density matrices once
    all_records = {}
    for n in n_values:
        recs = read_density_matrices(
            n,
            save_dir=final_data_dir,
            read_up_as_dict=False,
            special=args.specialstates,
        )
        all_records[n] = recs

    # Build list of job specs: (job_dir, rec, n, eps)
    jobs = []
    for n in n_values:
        recs = all_records[n]
        for eps in eps_values:
            for ridx, rec in enumerate(recs):
                if rec["type"] in ("ghz", "zero"):
                    job_folder_name = f"n{n}_eps{eps}_type={rec['type']}"
                else:
                    job_folder_name = f"n{n}_eps{eps}_seed{rec['seed']}_nt{rec['nt']}"

                job_dir = os.path.join(save_dir, job_folder_name)
                # We'll store the spec
                jobs.append((job_dir, rec, n, eps))

    print(f"Found {len(jobs)} jobs")
    print(f"Filter out jobs with unfinished mimicking state computation...")
    jobs = [
        (job_dir, rec, n, eps)
        for job_dir, rec, n, eps in jobs
        if job_has_mm_done(job_dir)
    ]

    print(f"Filter out jobs with unfinished precomputation...")
    jobs_temp = []
    for job_dir, rec, n, eps in tqdm(jobs):
        if job_precomp_done(job_dir):
            jobs_temp.append((job_dir, rec, n, eps))
    jobs = jobs_temp

    print(f"Filter out jobs with finished N2 search...")
    jobs = [j for j in jobs if not job_has_stage3(j[0])]

    print(f"Remaining {len(jobs)} jobs to run")

    allP_stacks = {}
    for n in n_values:
        _, Pn_stacked = generate_all_Ps_stacked(n)
        allP_stacks[n] = Pn_stacked

    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    print(f"starting jobs for {eps_values=}, {n_values=}, {date_time_str=}")
    with tqdm_joblib(tqdm(desc="N2 search", total=len(jobs))) as progress_bar:
        Parallel(
            n_jobs=100,
        )(
            delayed(run_job)(
                job_dir,
                rec,
                n,
                eps,
                allP_stacks[n],
                ntrials=args.trials,
                limitN=args.limitN,
                threshold=args.recov,
            )
            for (job_dir, rec, n, eps) in jobs
        )

    # # serial version
    # for (job_dir, rec, n, eps), gpuid in zip(jobs, cycle(gpus)):
    #     precompute_bell_probs(job_dir, rec, saving_to_publicwork2, gpuid, fs)
