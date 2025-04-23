import argparse
from glob import glob
import os

# testing single core
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import time
import numpy as np
from tqdm.auto import tqdm


from src.tqdm_joblib import tqdm_joblib
from src.dataloader.states import read_density_matrices
from src.dataloader.n1_results import load_results

from src.paulis import generate_all_Ps_stacked
from src.mm_state import numba_seed
from src.mm_state.v1 import compute_mimicking_state_v1_fast
from src.mm_state.v2 import compute_mimicking_state_v2_fast


import ipdb

"""
[Step 2] Constructs mimicking quantum states using iterative algorithms.

This script implements and evaluates two algorithms (v1 and v2) for constructing
a "mimicking" quantum state (`sigma`). The goal is that the mimicking state's
Pauli expectations approximate those of a target state (`rho`), specifically focusing
on the components identified in the support recovery step (Step 1).

The script loads the results from Step 1 (support estimates `pred`) and the original
test states. For each trial from Step 1, it runs the mimicking state construction
algorithm(s) (`--algo` determines v1, v2, or both).

Algorithm v1 uses a simpler iterative update based on constraint violations.
Algorithm v2 incorporates an adaptive step size with backtracking (overshoot) for 
faster convergence.

The constructed mimicking states (`final_state`) and associated metadata (e.g., number of updates,
convergence status) are saved to pickle files within the corresponding job directory under `--savedir`.

Functions:
    job_is_done: Checks if mimicking results already exist for a job.
    run_mimicking_for_job: Runs the mimicking state construction for one job.
    parse_arguments: Parses command-line arguments.

Usage:
    Run with `python -m scripts.study_mimicking_state_construction --n <n_values> --eps <eps_values> --algo <v1|v2|both> [--datadir <dir>] [--specialstates] [--savedir <dir>]`
"""


def job_is_done(
    job_dir,
    algo="both",
):
    """
    Check whether mimicking results already exist in the given job directory.

    Parameters
    ----------
    job_dir : str
        Path to the directory containing job results.
    algo : str, optional
        Which algorithm(s) to check: 'v1', 'v2', or 'both'.

    Returns
    -------
    done : bool
        True if corresponding result files already exist, False otherwise.
    """

    def has_file(pattern):
        return any(glob(os.path.join(job_dir, pattern)))

    found_v1 = has_file("mimicking_results_v1_*.pkl")
    found_v2 = has_file("mimicking_results_v2_*.pkl")

    if algo == "v1" and found_v1:
        print(f"  *** Skipping {job_dir}: Found v1 results ***")
        return True
    if algo == "v2" and found_v2:
        print(f"  *** Skipping {job_dir}: Found v2 results ***")
        return True
    if algo == "both" and found_v1 and found_v2:
        print(f"  *** Skipping {job_dir}: Found v1 and v2 results ***")
        return True

    return False


def run_mimicking_for_job(job_dir, P_stack, job_args, algo="both"):
    """
    Run mimicking state construction for a specific job, using v1 and/or v2 algorithm.

    Parameters
    ----------
    job_dir : str
        Directory for saving results.
    P_stack : ndarray
        Stack of Pauli operators (shape: (4^n, 2^n, 2^n)).
    job_args : tuple
        Tuple (n, rho, eps, state_seed, nt, target_up).
    algo : str, optional
        Which algorithm(s) to run: 'v1', 'v2', or 'both'.

    Returns
    -------
    None
        Results are written to disk in the job directory.
    """
    n, rho, eps, state_seed, nt, target_up = job_args

    if job_is_done(job_dir, algo):
        return None  # Skip the job

    # Load data
    results = load_results(job_dir)
    if results is False:
        return

    assert eps == results["eps"], f"{eps=} vs {results['eps']=}"
    assert n == results["n"], f"{n=} vs {results['n']=}"
    trials = results["samples_needed"]
    if "preds" in results:
        all_preds = results["preds"]
    else:
        print(f"Skipping since no prediction for {job_dir} found")
        return

    mimick_results_v1 = []
    mimick_results_v2 = []
    # Build sign oracle
    sign_oracle_vec = np.sign(target_up)

    try:
        for trial_idx, N_needed in enumerate(trials):
            print(f"Starting trial {trial_idx} for {n=}, {eps=}, {nt=}")
            t0 = time.time()
            seed = state_seed + trial_idx
            if np.isnan(N_needed) or N_needed is None or N_needed == -1:
                job_info = {
                    "trial": trial_idx,
                    "N": N_needed,
                    "pred": None,
                    "|S|": results["|S|"],
                    "recovery_threshold": results["recovery_threshold"],
                    "epsilon_sampling": eps,
                    "epsilon_mm": 4 / 3 * eps,
                    "seed_mm": seed,
                    "nt": nt,
                    "jobdir": os.path.basename(job_dir),
                    "algo_type": "v1",
                    "final_state": None,
                    "has_converged": False,
                }
                mimick_results_v1.append(job_info.copy())
                job_info["algo_type"] = "v2"
                mimick_results_v2.append(job_info)
                continue

            pred = all_preds[trial_idx].copy()

            # Threshold
            pred[np.abs(pred) < eps] = 0

            # seed from results is seed_value = row_idx * 1000 + trial_idx
            if algo in ["v1", "both"]:
                np.random.seed(seed)
                numba_seed(seed)
                res_v1 = compute_mimicking_state_v1_fast(
                    n, 4 / 3 * eps, pred.copy(), P_stack, sign_oracle_vec.copy()
                )
            else:
                res_v1 = None  # only None if we skip

            if algo in ["v2", "both"]:
                np.random.seed(seed)
                numba_seed(seed)
                res_v2 = compute_mimicking_state_v2_fast(
                    n,
                    4 / 3 * eps,
                    pred.copy(),
                    P_stack,
                    sign_oracle_vec.copy(),
                    eta_prefactor=1.5 / 4 * 2**n,
                )
            else:
                res_v2 = None  # only None if we skip

            job_info = {
                "trial": trial_idx,
                "N": N_needed,
                "pred": pred,
                "|S|": results["|S|"],
                "recovery_threshold": results["recovery_threshold"],
                "epsilon_sampling": eps,
                "epsilon_mm": 4 / 3 * eps,
                "seed_mm": seed,
                "nt": nt,
                "jobdir": os.path.basename(job_dir),
            }

            if res_v1 is not None:
                # res_v1 overwrites job_info if key is the same
                res_v1_flat = {**job_info, **res_v1}
                mimick_results_v1.append(res_v1_flat)

            if res_v2 is not None:
                # res_v2 overwrites job_info if key is the same
                res_v2_flat = {**job_info, **res_v2}
                mimick_results_v2.append(res_v2_flat)

            iters_v1 = res_v1["update_calls"] if res_v1 else None
            iters_v2 = (
                res_v2["update_calls"] + res_v2["total_overshoot"] if res_v2 else None
            )
            print(
                f"Trial {trial_idx} for {n=}, {eps=}, {nt=} {iters_v1=}, {iters_v2} took {time.time() - t0:.2f} seconds"
            )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error in {job_dir}, for {trial_idx=}: {e}")
        print(f"Not saving anything")
        return None

    # Save logs to a new pickle in the same job_dir
    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    out_path_v1 = os.path.join(job_dir, f"mimicking_results_v1_{date_time_str}.pkl")
    out_path_v2 = os.path.join(job_dir, f"mimicking_results_v2_{date_time_str}.pkl")

    if algo in ["v1", "both"] and mimick_results_v1:
        with open(out_path_v1, "wb") as f1:
            pickle.dump(mimick_results_v1, f1)
        print(f"Saved v1 mimicking results to {out_path_v1}")

    if algo in ["v2", "both"] and mimick_results_v2:
        with open(out_path_v2, "wb") as f2:
            pickle.dump(mimick_results_v2, f2)
        print(f"Saved v2 mimicking results to {out_path_v2}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="[Step 2] mimicking state construction."
    )

    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[
            2,
            3,
            4,
            5,
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
        "--algo",
        type=str,
        choices=["v1", "v2", "both"],
        default="both",
        help="Which mimicking algorithm(s) to run: 'v1', 'v2', or 'both'",
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

    args = parse_arguments()

    # Assign parsed arguments to variables
    n_values = args.n
    eps_values = args.eps
    results_subdir = args.savedir
    data_dir = args.datadir
    algo = args.algo
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
    jobs = [
        (job_dir, rec, n, eps)
        for job_dir, rec, n, eps in jobs
        if not job_is_done(job_dir, algo)
    ]

    print(f"Remaining jobs to run: {len(jobs)}")

    print(f"Computing all Pauli stacks for {n_values=}")
    # precompute all Pauli stacks
    allP_stacks = {}
    for n in n_values:
        _, Pn_stacked = generate_all_Ps_stacked(n)
        allP_stacks[n] = Pn_stacked

    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    print(f"starting jobs for {eps_values=}, {n_values=}, {date_time_str=}")

    # # Process each job spec sequentially
    # for job_dir, rec, n, eps in tqdm(jobs):
    #     job_args = (n, rec["rho"], eps, rec["seed"], rec["nt"], rec["up"])
    #     run_mimicking_for_job(job_dir, allP_stacks[n], job_args, algo=algo)

    with tqdm_joblib(tqdm(desc="MM state", total=len(jobs))) as progress_bar:
        Parallel(n_jobs=50)(
            delayed(run_mimicking_for_job)(
                job_dir,
                allP_stacks[n],
                (n, rec["rho"], eps, rec["seed"], rec["nt"], rec["up"]),
                algo=algo,
            )
            for job_dir, rec, n, eps in jobs
        )
