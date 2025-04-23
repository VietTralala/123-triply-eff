import os
from glob import glob
import pickle


def load_results(job_dir):
    """
    Load a single result dictionary from a job directory.

    Parameters
    ----------
    job_dir : str
        Path to the job directory.

    Returns
    -------
    result_dict : dict or bool
        Loaded result dictionary if exactly one file found,
        otherwise returns False and prints a warning.
    """
    result_files = sorted(glob(os.path.join(job_dir, "results_*.pkl")))

    if len(result_files) != 1:
        print(f"  *** Skipping {job_dir}: Found {len(result_files)} results files. ***")
        return False  # can't proceed

    results_path = result_files[0]

    with open(results_path, "rb") as f:
        result_dict = pickle.load(f)

    return result_dict
