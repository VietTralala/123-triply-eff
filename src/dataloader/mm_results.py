from glob import glob
import os
import pickle

import pandas as pd


def load_mm_result(
    jobdir,
    mm_prop_selection=None,
    load_all=False,
    algo="both",
):
    """
    Load mimicking state construction results (v1 and/or v2) from a job directory.

    Parameters
    ----------
    jobdir : str
        Path to the directory containing result pickle files.
    mm_prop_selection : list of str, optional
        List of dictionary keys (columns) to extract. Ignored if `load_all=True`.
        If None, a default subset of columns is used.
    load_all : bool, optional
        If True, load all fields in the records and ignore `mm_prop_selection`.
    algo : str, optional
        Which algorithm's results to load: 'v1', 'v2', or 'both'.

    Returns
    -------
    df_all : pd.DataFrame or None
        DataFrame with result records. Returns None if files are missing or loading fails.
    """

    if not load_all:
        if mm_prop_selection is None:
            mm_prop_selection = [
                "algo_type",
                "final_state",
                # "oracle_calls",
                "update_calls",
                # "selected_labels",
                # "update_factors",
                # "T",
                # "max_iterations": max_iterations,
                # "beta",
                "n",
                "epsilon_mm",
                "has_converged",
                "trial",
                "N",
                "pred",
                "|S|",
                # "recovery_threshold",
                "epsilon_sampling",
                "seed_mm",
                "nt",
                "jobdir",
                # "overshoot_sequence",
                # "eta_prefactor_sequence",
                "total_overshoot",
                "failed_due_to_substeps",
                "precomputed_bell_probs",
            ]
    else:
        mm_prop_selection = None

    def get_file_paths(jobdir, pattern, fs=None):
        if fs is None:
            return sorted(glob(os.path.join(jobdir, pattern)))
        else:
            return sorted(fs.glob(os.path.join(jobdir, pattern)))

    pattern_map = {
        "v1": "mimicking_results_v1_*.pkl",
        "v2": "mimicking_results_v2_*.pkl",
    }

    if algo == "both":
        algos_to_load = ["v1", "v2"]
    else:
        algos_to_load = [algo]

    dfs = []
    for alg in algos_to_load:
        paths = get_file_paths(jobdir, pattern_map[alg])
        if len(paths) != 1:
            print(f"Expected one file for {alg} in {jobdir}, found: {paths}")
            return None
        try:
            with open(paths[0], "rb") as f:
                records = pickle.load(f)
            df = pd.DataFrame(records)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to load {paths[0]}: {e}")

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    if mm_prop_selection is not None:
        df_all = df_all[[col for col in mm_prop_selection if col in df_all.columns]]
    return df_all
