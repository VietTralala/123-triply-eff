import os
import argparse
from glob import glob
import pickle
import time
from tqdm.auto import tqdm
from itertools import cycle

import numpy as np
import pandas as pd

from src.dataloader.states import read_density_matrices
from src.dataloader.mm_results import load_mm_result
from src.bell_sampling.bell_circuit import build_full_bell_generation_circuit
from src.bell_sampling import bell_measurement_probs


import ipdb

"""
[Step 2.5] Precomputes Bell measurement outcome probabilities.

This script calculates the probabilities of all $4^n$ possible outcomes when performing
a Bell measurement on the two-copy state `rho ⊗ sigma`, where `rho` is the original
test state and `sigma` is the mimicking state constructed in Step 2.

It loads the original states (`--datadir`) and the mimicking states generated in Step 2
(from `--savedir`). The computation can be performed on CPU (`--cpu`) or GPU (`--gpu`).

The precomputed probabilities (stored as dictionaries mapping binary outcome strings
to probabilities) are added to the existing mimicking state result files
(`mimicking_results_v1_*.pkl`, `mimicking_results_v2_*.pkl`) within each job directory.
This speeds up Step 3 (Sign Reconstruction) by avoiding repeated probability calculations.

Functions:
    parse_arguments: Parses command-line arguments.
    job_has_mm_done: Checks if mimicking state construction (Step 2) is complete.
    job_precomp_done: Checks if Bell probabilities have already been computed for a job.
    bell_measurement_probs_gpu: Computes Bell probabilities on GPU.
    precompute_bell_probs: Main function to precompute and save probabilities for a job.

Usage:
    Run with `python -m scripts.precompute_bell_probs --n <n_values> --eps <eps_values> [--gpu <ids> | --cpu] [--datadir <dir>] [--specialstates] [--savedir <dir>]`
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="[Step 2.5] precompute Bell probabilities for step 3"
    )

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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Flag to indicate if the code should run on CPU.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[0, 1],
        help="List of n values to test.",
    )

    return parser.parse_args()


def job_has_mm_done(job_dir):
    """
    Check if mimicking state computation (v1 and v2) is complete for a job.

    Parameters
    ----------
    job_dir : str
        Directory path of the job.

    Returns
    -------
    bool
        True if exactly one v1 and one v2 result file exists.
    """
    mm_files_v1 = glob(os.path.join(job_dir, "mimicking_results_v1*.pkl"))
    mm_files_v2 = glob(os.path.join(job_dir, "mimicking_results_v2*.pkl"))

    if len(mm_files_v1) > 1:
        print(
            f"  *** Skipping {job_dir}: Found {len(mm_files_v1)} mimicking_results_*.pkl files. ***"
        )
        return False
    if len(mm_files_v2) > 1:
        print(
            f"  *** Skipping {job_dir}: Found {len(mm_files_v2)} mimicking_results_*.pkl files. ***"
        )
        return False

    if len(mm_files_v1) == 0:
        print(
            f"  *** Skipping {job_dir}: Found no mimicking_results_v1_*.pkl files. ***"
        )
        return False

    if len(mm_files_v2) == 0:
        print(
            f"  *** Skipping {job_dir}: Found no mimicking_results_v2_*.pkl files. ***"
        )
        return False

    return True


def job_precomp_done(job_dir):
    """
    Check if Bell probabilities have already been precomputed for a job.

    Parameters
    ----------
    job_dir : str

    Returns
    -------
    bool
        True if both mimicking result files contain 'precomputed_bell_probs'
        and all entries are non-null.
    """
    mm_files = glob(os.path.join(job_dir, "mimicking_results_*.pkl"))

    assert len(mm_files) == 2, f"Expected 2 file, got {len(mm_files)}"

    # load mm resultsfile
    records = load_mm_result(
        job_dir, algo="both", load_all=True
    )  # dataframe with one row per trial

    # check for precomputed_bell_probs col in df
    if "precomputed_bell_probs" not in records.columns:
        return False

    # if any of the precomputed_bell_probs are None we consider precomputation not done
    # so that we can rerun the precomputation
    if records["precomputed_bell_probs"].isna().any():
        return False

    return True


def bell_measurement_probs_gpu(rho_gpu, sigma_gpu, gpuid):
    """
    Compute Bell measurement outcome probabilities on GPU.

    Parameters
    ----------
    rho_gpu : cp.ndarray
        Initial quantum state on GPU.
    sigma_gpu : cp.ndarray
        Mimicking state on GPU.
    gpuid : int
        GPU device ID.

    Returns
    -------
    cp.ndarray
        Normalized Bell measurement probabilities (shape: 4^n,).
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"
    import cupy as cp

    with cp.cuda.Device(0):
        # Kronecker product on GPU
        rhorho_gpu = cp.kron(rho_gpu, sigma_gpu)

        # Compute number of qubits
        nqubits = int(np.log2(rho_gpu.shape[0]))  # cp.log2 is weird with int

        # Build or load the Bell-generation circuit on GPU
        bell_states_gpu = cp.asarray(build_full_bell_generation_circuit(nqubits))

        # Main operations on GPU
        rhorho_psi_gpu = rhorho_gpu @ bell_states_gpu
        probs_array_gpu = cp.real(
            cp.sum(bell_states_gpu.conj() * rhorho_psi_gpu, axis=0)
        )

        # Clip very small negative values to 0 for numerical stability
        mask = cp.isclose(probs_array_gpu, 0.0) & (probs_array_gpu < 0)
        probs_array_gpu[mask] = 0.0

        # Renormalize probabilities to sum to 1
        total_prob_gpu = cp.sum(probs_array_gpu)
        probs_array_gpu /= total_prob_gpu

    return probs_array_gpu


def precompute_bell_probs(
    jobdir,
    state_info,
    gpuid,
    use_cpu=False,
):
    """
    Precompute and cache Bell measurement probabilities for a given job.

    Parameters
    ----------
    jobdir : str
        Path to the job directory.
    state_info : dict
        Contains 'rho', 'n', and other metadata.
    gpuid : int
        GPU device ID.
    use_cpu : bool, optional
        If True, run computation on CPU instead of GPU.

    Side Effects
    ------------
    Modifies the mimicking_results_v1_*.pkl and mimicking_results_v2_*.pkl
    by appending a `precomputed_bell_probs` entry to each trial row.
    """
    import os

    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"
        import cupy as cp

    # gather relevant info!
    n = state_info["n"]
    rho = state_info["rho"]
    root_save_dir = os.path.dirname(jobdir)  # where jobdir is located

    if job_precomp_done(jobdir):
        print(f"Skipping {jobdir}: precomputed_bell_probs already exists.")
        return

    # load mm state
    # print(f"Loading {jobdir}")
    t0 = time.time()
    df_mm = load_mm_result(jobdir, algo="both", load_all=True)
    if df_mm is None or df_mm.empty:
        return

    # print(f"Loaded {len(df_mm)} trials in {time.time() - t0:.2f}s")

    all_meas_probs = []

    if not use_cpu:
        with cp.cuda.Device(0):
            # move to GPU
            rho_gpu = cp.asarray(rho)

            for row_idx, row in df_mm.iterrows():
                if row["final_state"] is None:
                    meas_probs = cp.zeros(4**n)  # dummy
                else:
                    sigma_gpu = cp.asarray(row["final_state"])
                    meas_probs = bell_measurement_probs_gpu(rho_gpu, sigma_gpu, gpuid)
                all_meas_probs.append(meas_probs)

            all_meas_probs = cp.asnumpy(cp.stack(all_meas_probs))
            # print(all_meas_probs.shape)
            all_probs_dict = []

            binary_keys = [np.binary_repr(i, width=2 * n) for i in range(4**n)]
            # binary_keys = [format(i, f"0{2*n}b") for i in range(4**n)]
            for probs_array in all_meas_probs:
                # Build the dictionary (keys are binary strings)
                if np.allclose(probs_array, 0):
                    probs = None
                else:
                    probs = {
                        binary_keys[idx]: prob for idx, prob in enumerate(probs_array)
                    }
                all_probs_dict.append(probs)

    else:  # use CPU
        for row_idx, row in df_mm.iterrows():
            if row["final_state"] is None:
                meas_probs = np.zeros(4**n)  # dummy
            else:
                sigma = np.asarray(row["final_state"])
                meas_probs = bell_measurement_probs(rho, sigma=sigma)  # returns dict
            all_meas_probs.append(meas_probs)

        all_probs_dict = all_meas_probs  # already a list of dicts, no need to convert

    df_mm["precomputed_bell_probs"] = all_probs_dict  # list of dicts

    # print(len(all_probs_dict))
    if not use_cpu:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # check if we have results for both algo types
    for algo_type in df_mm["algo_type"].unique():
        pattern = f"mimicking_results_{algo_type}_*.pkl"
        file_paths = sorted(glob(os.path.join(jobdir, pattern)))
        if len(file_paths) != 1:
            print(
                f"WARNING: found {len(file_paths)} {pattern} in {jobdir}, skip and not saving anything."
            )
            return

    # Write each subset of rows back to the correct pkl
    for algo_type in df_mm["algo_type"].unique():
        pattern = f"mimicking_results_{algo_type}_*.pkl"
        file_paths = sorted(glob(os.path.join(jobdir, pattern)))
        pkl_path = file_paths[0]
        subdf = df_mm[df_mm["algo_type"] == algo_type].copy()
        new_records = subdf.to_dict(orient="records")

        with open(pkl_path, "wb") as f:
            pickle.dump(new_records, f)

        print(f"Updated {pkl_path} with {len(subdf)} rows.")
    return


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from src.tqdm_joblib import tqdm_joblib

    args = parse_arguments()

    # Assign parsed arguments to variables
    n_values = args.n
    eps_values = args.eps
    results_subdir = args.savedir
    data_dir = args.datadir
    gpus = args.gpu
    print(f"Running with arguments: {args}")

    project_dir = os.getcwd()

    # data directory that stores the density matrices:
    root_data_dir = os.path.join(project_dir, "data")
    # Where job results subdirectories are located:
    root_results_dir = os.path.join(project_dir, "data/results")

    save_dir = os.path.join(root_results_dir, results_subdir)
    final_data_dir = os.path.join(root_data_dir, data_dir)
    print(f"Loading density matrices from {final_data_dir}")
    print(f"Loading measurements and results from {save_dir}")

    date_time_str = time.strftime("%Y%m%d-%H%M%S")
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
        build_full_bell_generation_circuit(n)  # just to precompute the cache once
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

    print(f"Filter out jobs with finished precomputation...")

    jobs_temp = []
    for job_dir, rec, n, eps in tqdm(jobs):
        if not job_precomp_done(job_dir):
            jobs_temp.append((job_dir, rec, n, eps))
    jobs = jobs_temp

    print(f"Remaining {len(jobs)} jobs to run")

    print(f"starting jobs for {eps_values=}, {n_values=}, {date_time_str=}")

    njobs_parallel = 10 if max(n_values) < 7 else 2  # adjust based on availabe GPU ram
    # one n7 jobs takes about 16GB VRAM
    with tqdm_joblib(
        tqdm(desc="precompute bell probs", total=len(jobs))
    ) as progress_bar:
        Parallel(n_jobs=len(gpus) * njobs_parallel, backend="multiprocessing")(
            delayed(precompute_bell_probs)(
                job_dir,
                rec,
                gpuid,
                use_cpu=args.cpu,
            )
            for (job_dir, rec, n, eps), gpuid in zip(jobs, cycle(gpus))
        )

    # # serial version
    # for (job_dir, rec, n, eps), gpuid in zip(jobs, cycle(gpus)):
    #     precompute_bell_probs(job_dir, rec, saving_to_publicwork2, gpuid, use_cpu=args.cpu)
