import os
from pathlib import Path
import re
import pickle
from glob import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

from src.dataloader.states import read_density_matrices
from src.dataloader.mm_results import load_mm_result


def load_states(
    result_dir,
    n_values=(2, 3, 4, 5, 6, 7),
    eps_values=(0.05, 0.07, 0.11, 0.16, 0.23, 0.34, 0.5),
    savedirs=None,
):
    if savedirs is None:
        result_dir = Path(result_dir)
        # 0: approx_recovery_jacc, 1: results, 2: data → project_dir is before that
        project_dir = result_dir.parents[2]

        save_dirs = {
            "gibbs": os.path.join(project_dir, "data/normalized_gibbs"),
            "special": os.path.join(project_dir, "data/special_states"),
        }

    jobs = []
    for state_type, save_dir in save_dirs.items():
        for n in n_values:
            recs = read_density_matrices(
                n,
                save_dir=save_dir,
                read_up_as_dict=False,
                special=state_type == "special",
            )
            print(f"loaded {state_type=}, n={n}, recs={len(recs)}")
            for eps in eps_values:
                for ridx, rec in enumerate(recs):
                    if state_type == "gibbs":
                        jobdir = f"n{n}_eps{eps}_seed{rec['seed']}_nt{rec['nt']}"
                    else:
                        jobdir = f"n{n}_eps{eps}_type={rec['type']}"
                    jobs.append(
                        (
                            rec["n"],
                            rec["seed"],
                            rec["nt"],
                            rec["rho"],
                            rec["up"],
                            rec["terms"],
                            eps,
                            jobdir,
                            # ridx,
                        )
                    )

    df_jobs = pd.DataFrame(
        jobs,
        columns=[
            "n",
            "rho_seed",
            "nt",
            "rho",
            "target_up",
            "terms",
            "eps",
            "jobdir",
            # "ridx",
        ],
    )
    print(df_jobs.shape)
    return df_jobs


def load_step1_results(
    result_dir,
    return_long_format=False,
):
    results = []
    for path in tqdm(glob(f"{result_dir}/**/results_*.pkl", recursive=True)):
        with open(path, "rb") as f:
            result_dict = pickle.load(f)
            jobdir_folder_name = os.path.basename(os.path.dirname(path))
            result_dict["jobdir"] = jobdir_folder_name
            results.append(result_dict)

    processed_df = __process_step1_results(results)
    # preprocessing
    ntrials = len(processed_df["preds"].iloc[0])
    processed_df["trial_idx"] = [np.arange(ntrials)] * len(processed_df)
    # before exploding we calc some stats over trials
    processed_df["mean_samples"] = processed_df["samples_needed"].apply(
        lambda x: np.nanmean(x)
    )
    processed_df["std_samples"] = processed_df["samples_needed"].apply(
        lambda x: np.nanstd(x)
    )
    processed_df["med_samples"] = processed_df["samples_needed"].apply(
        lambda x: np.nanmedian(x)
    )
    processed_df["num_failed_trials"] = processed_df["samples_needed"].apply(
        lambda x: np.sum(np.isnan(x))
    )
    processed_df["num_successful_trials"] = processed_df["samples_needed"].apply(
        lambda x: np.sum(~np.isnan(x))
    )
    processed_df["success_rate"] = processed_df["num_successful_trials"] / ntrials
    processed_df["state_type"] = processed_df["jobdir"].apply(
        lambda x: x.split("=")[-1] if "type" in x else "gibbs"
    )

    print(f"loaded and processed {processed_df.shape} results")
    # currently each row is a jobdir and contains a list of trials
    if return_long_format:
        # explode so that each row is a single trial if
        df_long = processed_df.explode(
            ["samples_needed", "preds", "jaccard_seq", "trial_idx"]
        )
        return df_long

    return processed_df


def __process_step1_results(results):
    processed_data = []
    for res in results:  # list of dicts
        n = res["n"]
        eps = res["eps"]
        S_size = res["|S|"]
        samples_needed = np.array(res["samples_needed"], dtype=np.float64)
        # samples_needed_largest = np.array(
        #     res["samples_needed_largest"], dtype=np.float64
        # )
        preds = res["preds"]  # list of np.ndarray, one per trial, None if failed
        jobdir = res["jobdir"]

        processed_data.append(
            (
                n,
                eps,
                S_size,
                samples_needed,
                preds,
                jobdir,
            )
        )

    # only get raw data. preprocessing is done on df
    df = pd.DataFrame(
        processed_data,
        columns=[
            "n",
            "eps",
            "|S|",
            "samples_needed",
            "preds",
            "jobdir",
        ],
    )
    return df


def compute_jacc(row):
    j_per_trial = []
    target, preds_per_trial, eps = row["target_up"], row["preds"], row["eps"]
    target_above_eps = np.abs(target) >= eps
    for pred in preds_per_trial:
        if pred is None:
            j_per_trial.append(np.nan)
            continue
        pred_above_eps = np.abs(pred) >= eps
        intersection = np.sum(target_above_eps & pred_above_eps)
        union = np.sum(target_above_eps | pred_above_eps)
        J = intersection / union if union != 0 else 1.0
        j_per_trial.append(J)
    return j_per_trial


def load_step1_and_states(
    result_dir,
    n_values=(2, 3, 4, 5, 6, 7),
    eps_values=(0.05, 0.07, 0.11, 0.16, 0.23, 0.34, 0.5),
):
    df_states = load_states(
        result_dir=result_dir, n_values=n_values, eps_values=eps_values
    )
    df_results = load_step1_results(result_dir=result_dir)

    # merge
    df_merged = pd.merge(
        df_results,
        df_states,
        left_on="jobdir",
        right_on="jobdir",
        suffixes=(None, "_jobs"),
    )

    # assert that duplicate rows (same name with suffix _jobs) have same values and then drop
    # Identify duplicate columns (those with '_jobs' suffix)
    duplicate_cols = [col for col in df_merged.columns if col.endswith("_jobs")]

    # Check that original and '_jobs' columns are equal
    for col in duplicate_cols:
        base_col = col[:-5]  # remove the '_jobs' suffix
        assert (
            df_merged[base_col] == df_merged[col]
        ).all(), f"Mismatch in column: {base_col}"

    # Drop the duplicate columns
    df_merged.drop(columns=duplicate_cols, inplace=True)

    df_merged["num_close_eps"] = df_merged.apply(
        lambda x: np.sum(np.isclose(np.abs(x["target_up"]), x["eps"], atol=0.005)),
        axis=1,
    )
    df_merged["Js"] = df_merged.apply(compute_jacc, axis=1)
    print(f"Loaded {df_merged.shape} merged results from step1 and states")
    return df_merged


def calc_num_updates(row):
    if row["algo_type"] == "v1":
        return row["update_calls"]
    elif row["algo_type"] == "v2":
        return row["update_calls"] + row["total_overshoot"]
    else:
        return None


def find_seed_state(jobdir):
    pattern = re.compile(r"seed(\d+)_nt")
    m = pattern.search(jobdir)
    if m:
        return int(m.group(1))
    else:
        return None


def filter_jobdirs(jobdirs, n=None, eps=None):
    assert isinstance(jobdirs, list), "jobdirs should be a list"
    assert n is None or isinstance(n, int), "n should be an int or None"
    assert eps is None or isinstance(eps, float), "eps should be a float or None"

    filtered_jobdirs = []

    for jobdir in jobdirs:
        if n is not None and f"n{n}_" not in jobdir:
            continue
        if eps is not None and f"eps{eps}_" not in jobdir:
            continue
        filtered_jobdirs.append(jobdir)

    return filtered_jobdirs


def load_step2_results(
    result_dir,
    discard_failed_N1=True,
    n_values=None,
    eps_values=None,
):
    jobdirs = glob(os.path.join(result_dir, "*"))

    if (n_values is None) != (eps_values is None):
        raise ValueError(
            "n_values and eps_values should either both be None or both have values"
        )

    # filter jobdirs by n and eps
    if n_values is not None and eps_values is not None:
        filtered_jobdirs = []
        for n in n_values:
            for eps in eps_values:
                filtered_jobdirs += filter_jobdirs(jobdirs, n=n, eps=eps)
        jobdirs = filtered_jobdirs

    print(f"Found {len(jobdirs)} jobdirs")

    # Load all mimicking results from every jobdir
    # takes about 5m30s on oce for 4256 jobdirs
    all_dfs = []
    for jd in tqdm(jobdirs, desc="Loading mimicking result files"):

        df_mm = load_mm_result(jd, load_all=True)
        all_dfs.append(df_mm)

    all_mm = pd.concat(all_dfs, ignore_index=True)

    all_mm["num_updates"] = all_mm.apply(calc_num_updates, axis=1)
    all_mm["log|S|"] = np.log2(all_mm["|S|"])
    all_mm["seed_state"] = all_mm["jobdir"].apply(find_seed_state)
    all_mm["maxT_reached"] = all_mm.apply(
        lambda x: x["max_iterations"] == x["update_calls"], axis=1
    )
    all_mm["state_type"] = all_mm["jobdir"].apply(
        lambda x: x.split("=")[-1] if "type" in x else "gibbs"
    )

    all_mm["mm_rowid"] = all_mm.apply(
        lambda x: f"{x['jobdir']}_trial{x['trial']}_algo{x['algo_type']}", axis=1
    )

    if discard_failed_N1:
        failed_N1_search = all_mm.N.isna()
        print(
            f"Discard {failed_N1_search.sum()} failed N1 search rows out of {len(all_mm)} total."
        )
        return all_mm[~all_mm.N.isna()].copy()
    else:
        return all_mm


def job_has_stage3(job_dir):
    stage3_results = glob(os.path.join(job_dir, "stage3_results_*.pkl"))

    if len(stage3_results) == 1:
        return True

    return False


def load_stage3_single_jobdir(jobdir):
    stage3_file = glob(os.path.join(jobdir, "stage3_results_*.pkl"))
    assert len(stage3_file) == 1, f"Found {len(stage3_file)} stage3 files in {jobdir}"
    stage3_file = stage3_file[0]

    with open(stage3_file, "rb") as f:
        stage3_result = pickle.load(f)

    # rowid = f"{jobdir_folder_only}_trial{tidx}_algo{algo_type}"  # IMPORTANT format
    # drop index bc rowid is already in the data as a column
    return pd.DataFrame(stage3_result).T.reset_index(drop=True)


def load_step3_results(
    result_dir,
    n_values=None,
    eps_values=None,
):
    # Load all stage3 results from every jobdir
    # takes about 6m-10m on oce for 4231 jobdirs
    print(f"Loading all stage3 results from {result_dir}")
    jobdirs = glob(os.path.join(result_dir, "*"))

    if (n_values is None) != (eps_values is None):
        raise ValueError(
            "n_values and eps_values should either both be None or both have values"
        )

    # filter jobdirs by n and eps
    if n_values is not None and eps_values is not None:
        filtered_jobdirs = []
        for n in n_values:
            for eps in eps_values:
                filtered_jobdirs += filter_jobdirs(jobdirs, n=n, eps=eps)
        jobdirs = filtered_jobdirs

    print(f"Found {len(jobdirs)} jobdirs")

    jobdirs_with_stag3_results = [j for j in jobdirs if job_has_stage3(j)]
    print(f"Found {len(jobdirs_with_stag3_results)} jobdirs with stage3 results")
    all_dfs3 = []
    for jd in tqdm(jobdirs_with_stag3_results, desc="Loading stage3 result files"):

        df3 = load_stage3_single_jobdir(jd)  # df with each row a trial
        all_dfs3.append(df3)

    all_s3 = pd.concat(all_dfs3, ignore_index=True)
    all_s3["s3_trial_idx"] = [(0, 1, 2, 3, 4)] * len(all_s3)
    all_s3["mean_samples"] = all_s3["samples_needed"].apply(np.nanmean)
    all_s3["std_samples"] = all_s3["samples_needed"].apply(np.nanstd)
    all_s3["med_samples"] = all_s3["samples_needed"].apply(np.nanmedian)
    all_s3["num_failed_trials"] = all_s3["samples_needed"].apply(
        lambda x: np.sum(np.isnan(x))
    )
    all_s3["relstd_samples"] = all_s3["std_samples"] / all_s3["med_samples"]

    print(f"Loaded {all_s3.shape} stage3 results")

    return all_s3


def load_step3_and_mm_results(
    result_dir,
    all_s3=None,
    all_mm=None,
    all_s1=None,
    discard_failed_N1=True,
    return_J=True,
    n_values=None,
    eps_values=None,
):
    if all_s3 is None:
        all_s3 = load_step3_results(
            result_dir=result_dir,
            n_values=n_values,
            eps_values=eps_values,
        )

    if all_mm is None:
        all_mm = load_step2_results(
            result_dir=result_dir,
            discard_failed_N1=discard_failed_N1,
            n_values=n_values,
            eps_values=eps_values,
        )

    all_mms3 = pd.merge(
        all_s3,
        all_mm,
        left_on=["rowid"],
        right_on=["mm_rowid"],
        suffixes=(None, "_mm"),
    )

    if "hs_norm_threshold" in all_mms3.columns:
        bigs3 = all_mms3.explode(
            [
                "samples_needed",
                "signed_preds",
                "hs_norm_threshold",
                "hs_norm",
                "all_measurements",
                "s3_trial_idx",
            ]
        ).reset_index(
            drop=True
        )  # each row has a s3 trial_idx
    elif "sign_agreement_ratio" in all_mms3.columns:  # studyN2_search_V3
        bigs3 = all_mms3.explode(
            [
                "samples_needed",
                "signed_preds",
                "sign_agreement_ratio",
                "hs_norm",
                "all_measurements",
                "s3_trial_idx",
            ]
        ).reset_index(
            drop=True
        )  # each row has a s3 trial_idx

    if return_J:
        if all_s1 is None:
            all_s1 = load_step1_and_states(result_dir=result_dir)
        # print(bigs3.shape)
        # get J from df_s1
        bigs3["J"] = bigs3.progress_apply(
            lambda x: all_s1[(all_s1["jobdir"] == x["jobdir"])]["Js"].values[0][
                x["trial"]
            ],
            axis=1,
        )

    return bigs3, all_mms3, all_s1


def load_meas(row, result_dir, all_trials=False):
    if all_trials:
        full_jobdir = os.path.join(result_dir, row["jobdir"].iloc[0])
    else:
        full_jobdir = os.path.join(result_dir, row["jobdir"])
    # for this jobdir load actual measurements
    meas_files = glob(os.path.join(full_jobdir, "measurements_*.pkl"))

    print("loading", meas_files[0])
    with open(meas_files[0], "rb") as f:
        all_measurements = pickle.load(f)

    if all_trials:
        return all_measurements

    select_trial = row["trial_idx"]
    return all_measurements[select_trial]
