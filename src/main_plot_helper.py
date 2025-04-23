# ---------------------------------------------------------------------------
# imports for step 1
import numba
import numpy as np
from tqdm.auto import tqdm
from src.bell_sampling import compute_pauli_labels

# ---------------------------------------------------------------------------
# imports for step 2
from src.mm_state import (
    numba_seed,
    find_violating_label,
    gibbs_state_obj,
    tr_prod,
)
from src.mm_state.v2 import update_step_numba
from src.paulis import generate_all_Ps_stacked
from scripts.study_sign_reconstruction import generate_u_P_exact_numba

# ---------------------------------------------------------------------------
# imports for step 3
from scripts.study_sign_reconstruction import MSE


# ---------------------------------------------------------------------------
# needed for step 1
# ---------------------------------------------------------------------------


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
    collect_pred_intervall=100,
):
    """
    Process a new block of measurement outcomes (meas_chunk),
    update accum_sign in-place, and after each new sample M
    (from start_count+1 to start_count+block_size),
    check if the Jaccard condition is satisfied.

    Returns:
        found_M: The first M (start_count < M <= start_count+block_size)
                 at which the threshold is met, or -1 if never met.
        found_M_largest: M at which the largest non-trivial index is recovered,
                         or -1 if never met.
        final_pred: the final predicted values at the end of the block
                    (useful if we want to store them)
    """
    num_samples, two_n = meas_chunk.shape
    n = two_n // 2
    four_n = pauli_labels.shape[1]
    found_M = -1
    found_M_largest = -1

    # Convert measurements -> array bidxs of shape (block_size, n)
    bidx_chunk = 2 * meas_chunk[:, :n] + meas_chunk[:, n:]

    allJs = np.zeros(num_samples, dtype=np.float64)

    num_collections = num_samples // collect_pred_intervall
    # all_preds = np.zeros((num_collections, four_n), dtype=np.float64)

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

        # if i % collect_pred_intervall == 0:
        #     all_preds[i // collect_pred_intervall] = pred

        # Jaccard check
        target_above_eps = np.abs(target) >= eps
        pred_above_eps = np.abs(pred) >= eps
        intersection = np.sum(target_above_eps & pred_above_eps)
        union = np.sum(target_above_eps | pred_above_eps)
        J = intersection / union if union != 0 else 1.0
        allJs[i] = J

        if J >= recovery_threshold:
            found_M = M
            return (
                found_M,
                pred,
                allJs,
            )  # all_preds

    return (
        -1,
        pred,
        allJs,
    )  # all_preds


def recompute_s1(trial_df):
    allJs_per_trial = []
    # allpred_per_trial = []

    for ridx, row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        # print(row['state_type'])

        new_measurement_block = row["meas"]
        n = row["n"]
        eps = row["eps"]
        pauli_labels = compute_pauli_labels(n)
        sign_table = np.array(
            [
                [1, 1, 1, 1],  # I
                [1, 1, -1, -1],  # X
                [-1, 1, 1, -1],  # Y
                [1, -1, 1, -1],  # Z
            ],
            dtype=np.int8,
        )

        accum_sign = np.zeros(4**n, dtype=np.float64)

        M_candidate, pred, allJs = update_accum_sign_and_check(
            new_measurement_block,
            accum_sign,
            pauli_labels=pauli_labels,
            sign_table=sign_table,
            target=row["target_up"],
            eps=eps,
            recovery_threshold=0.9,
            start_count=0,
        )
        allJs_per_trial.append(allJs)
        # allpred_per_trial.append(allpreds)

        # print("M_candidate", M_candidate)

    # trial_df["allJs"] = allJs_per_trial
    # trial_df["allpreds"] = allpred_per_trial
    return allJs_per_trial


# ---------------------------------------------------------------------------
# needed for step 2
# ---------------------------------------------------------------------------


@numba.njit
def compute_v1_fast(
    n,
    epsilon,
    u_vec_thresh,
    sign_oracle_vec,
    P_stack,
    beta,
    max_iters,
):
    D = 2**n
    omega = np.eye(D, dtype=np.complex128) / D
    M_sum = np.zeros((D, D), dtype=np.complex128)

    oracle_calls = 0
    update_calls = 0
    selected_labels = []
    factor_seq = []
    feasibility_ratio_seq = []
    significant_up_mask = u_vec_thresh > 0
    num_significant_up = np.sum(significant_up_mask)
    has_converged = False
    for t in range(max_iters):

        trPsigma = generate_u_P_exact_numba(omega, P_stack)
        feasibility_ratio = (
            np.sum(np.abs(trPsigma[significant_up_mask]) >= epsilon / 4)
            / num_significant_up
        )
        feasibility_ratio_seq.append(feasibility_ratio)

        label = find_violating_label(u_vec_thresh, P_stack, omega, epsilon)
        if label == -1:
            has_converged = True
            break  # converged

        selected_labels.append(label)
        oracle_calls += 1
        update_calls += 1

        P = P_stack[label]
        r_P = sign_oracle_vec[label]
        diff = tr_prod(P, omega) - r_P * u_vec_thresh[label]
        fac = np.sign(diff)
        factor_seq.append(fac)
        M_sum += fac * P

        omega = gibbs_state_obj(M_sum, beta)

    return (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        has_converged,
        M_sum,
        omega,
        feasibility_ratio_seq,
    )


def compute_mimicking_state_v1_fast(
    n,
    epsilon,
    u_vec_thresh,
    P_stack,
    sign_oracle_vec,
    max_iterations=None,
):
    """
    Fast version of mimicking state creation v1 using Numba.
    Relies on external gibbs_state for stability.
    """
    T = int(np.ceil(64 * n / epsilon**2)) + 1
    if max_iterations:
        # T = min(T, max_iterations)
        max_iterations = min(T, max_iterations)
    else:
        max_iterations = T
    beta = np.sqrt(n / T)

    # Run fast loop
    (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        has_converged,
        M_sum,
        omega,
        feasibility_ratio_seq,
    ) = compute_v1_fast(
        n, epsilon, u_vec_thresh, sign_oracle_vec, P_stack, beta, max_iterations
    )

    return {
        "algo_type": "v1",
        "final_state": omega,
        "oracle_calls": oracle_calls,
        "update_calls": update_calls,
        "selected_labels": selected_labels,
        "update_factors": factor_seq,
        "T": T,
        "max_iterations": max_iterations,
        "beta": beta,
        "n": n,
        "epsilon_mm": epsilon,
        "has_converged": has_converged,
        "feasibility_ratio_seq": feasibility_ratio_seq,
    }


@numba.njit
def compute_v2_fast(
    n,
    epsilon,
    u_vec_thresh,
    sign_oracle_vec,
    P_stack,
    beta,
    eta_init,
    max_iters,
):
    D = 2**n
    omega = np.eye(D, dtype=np.complex128) / D
    M_sum = np.zeros((D, D), dtype=np.complex128)

    oracle_calls = 0
    update_calls = 0
    selected_labels = []
    factor_seq = []
    overshoot_list = []
    eta_prefactor_list = []
    total_overshoot = 0
    current_eta_pf = eta_init
    feasibility_ratio_seq = []
    significant_up_mask = u_vec_thresh > 0
    num_significant_up = np.sum(significant_up_mask)
    has_converged = False
    for t in range(max_iters):

        trPsigma = generate_u_P_exact_numba(omega, P_stack)
        feasibility_ratio = (
            np.sum(np.abs(trPsigma[significant_up_mask]) >= epsilon / 4)
            / num_significant_up
        )
        feasibility_ratio_seq.append(feasibility_ratio)

        label = find_violating_label(u_vec_thresh, P_stack, omega, epsilon)
        if label == -1:
            has_converged = True
            break  # converged

        selected_labels.append(label)
        oracle_calls += 1
        update_calls += 1

        P = P_stack[label]
        r_P = sign_oracle_vec[label]
        u_val = u_vec_thresh[label]

        M_sum, omega, current_eta_pf, update_fac, substeps, accepted = (
            update_step_numba(M_sum, P, omega, r_P, u_val, beta, current_eta_pf)
        )

        overshoot_list.append(substeps)
        eta_prefactor_list.append(current_eta_pf)
        factor_seq.append(update_fac)
        total_overshoot += substeps

        if not accepted:
            break

    return (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        overshoot_list,
        eta_prefactor_list,
        total_overshoot,
        has_converged,
        accepted,
        omega,
        feasibility_ratio_seq,
    )


def compute_mimicking_state_v2_fast(
    n,
    epsilon,
    u_vec_thresh,
    P_stack,
    sign_oracle_vec,
    max_iterations=None,
    eta_prefactor=100,
):
    T = int(np.ceil(64 * n / epsilon**2)) + 1
    if max_iterations:
        max_iterations = min(T, max_iterations)
    else:
        max_iterations = T
    beta = np.sqrt(n / T)

    results = compute_v2_fast(
        n,
        epsilon,
        u_vec_thresh,
        sign_oracle_vec,
        P_stack,
        beta,
        eta_prefactor,
        max_iterations,
    )

    (
        oracle_calls,
        update_calls,
        selected_labels,
        factor_seq,
        overshoot_list,
        eta_prefactor_list,
        total_overshoot,
        has_converged,
        accepted_step,
        omega,
        feasibility_ratio_seq,
    ) = results

    return {
        "algo_type": "v2",
        "final_state": omega,
        "oracle_calls": oracle_calls,
        "update_calls": update_calls,
        "selected_labels": selected_labels,
        "update_factors": factor_seq,
        "overshoot_sequence": overshoot_list,
        "eta_prefactor_sequence": eta_prefactor_list,
        "total_overshoot": total_overshoot,
        "T": T,
        "max_iterations": max_iterations,
        "beta": beta,
        "n": n,
        "epsilon_mm": epsilon,
        "has_converged": has_converged,
        "failed_due_to_substeps": not accepted_step,
        "feasibility_ratio_seq": feasibility_ratio_seq,
    }


def recompute_s2(case, all_state_info, new_eta_prefactor=True):
    case = case.copy()
    case_state_info = all_state_info[all_state_info.jobdir == case["jobdir"]].iloc[0]

    # extract the parameters from the row
    n = int(case["n"])
    eps = case["epsilon_sampling"]
    u_vec = np.array(case["pred"]).copy()
    seed = int(case["seed_mm"])

    # threshold this with eps from N1
    u_vec[np.abs(u_vec) < eps] = 0

    # generate the Pauli-operator stack for dimension 2^n
    _, P_stack = generate_all_Ps_stacked(n)
    sign_oracle_vec = np.sign(case_state_info["target_up"])

    if new_eta_prefactor is True:
        new_eta_prefactor = 1.5 / 4 * 2**n

    algo_type = case["algo_type"]
    if algo_type == "v1":

        np.random.seed(seed)
        numba_seed(seed)
        res_v1 = compute_mimicking_state_v1_fast(
            n, 4 / 3 * eps, u_vec.copy(), P_stack, sign_oracle_vec.copy()
        )
        print(
            f"num_updates recomputed=\t{res_v1['update_calls']} vs\trecorded {case['num_updates']}"
        )
        print(
            f"converged recomputed=\t{res_v1['has_converged']} vs\trecorded {case['has_converged']}"
        )
        return res_v1
    else:
        np.random.seed(seed)
        numba_seed(seed)
        res_v2 = compute_mimicking_state_v2_fast(
            n,
            4 / 3 * eps,
            u_vec.copy(),
            P_stack,
            sign_oracle_vec.copy(),
            eta_prefactor=new_eta_prefactor,
        )
        print(
            f"num_updates recomputed=\t{res_v2['update_calls']+ res_v2['total_overshoot']} vs\trecordedld {case['num_updates']}"
        )
        print(
            f"converged recomputed=\t{res_v2['has_converged']} vs\trecorded {case['has_converged']}"
        )
        return res_v2


# ---------------------------------------------------------------------------
# needed for step 3
# ---------------------------------------------------------------------------


@numba.njit
def update_accum_sign_and_check_s3(
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
    Process a new block of measurement outcomes (meas_chunk),
    update accum_sign in-place, and after each new sample M
    (from start_count+1 to start_count+block_size),
    check if the Jaccard condition is satisfied.

    Returns:
        found_M: The first M (start_count < M <= start_count+block_size)
                 at which the threshold is met, or -1 if never met.
        found_M_largest: M at which the largest non-trivial index is recovered,
                         or -1 if never met.
        final_pred: the final predicted values at the end of the block
                    (useful if we want to store them)
    """
    num_samples, two_n = meas_chunk.shape
    n = two_n // 2
    four_n = pauli_labels.shape[1]
    twopow_n = 2**n
    found_M = -1

    # Convert measurements -> array bidxs of shape (block_size, n)
    bidx_chunk = 2 * meas_chunk[:, :n] + meas_chunk[:, n:]

    all_sign_agreement = np.zeros(num_samples, dtype=np.float64)
    all_normalized_hs_norm = np.zeros(num_samples, dtype=np.float64)

    # Process each new sample incrementally
    for i in range(num_samples):
        # Build sign_vec for the i-th new outcome
        sign_vec = np.ones(four_n, dtype=np.int8)
        for q in range(n):
            outcome = bidx_chunk[i, q]
            for j in range(four_n):
                pauli = pauli_labels[q, j]
                sign_vec[j] *= sign_table[pauli, outcome]

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

        all_sign_agreement[i] = sign_agreement
        all_normalized_hs_norm[i] = MSE(signed_pred, target) * twopow_n

        if sign_agreement >= threshold:
            found_M = M
            normalized_hs_norm = MSE(signed_pred, target) * twopow_n
            return (
                found_M,
                sign,
                signed_pred,
                normalized_hs_norm,
                sign_agreement,
                all_sign_agreement,
                all_normalized_hs_norm,
            )

    normalized_hs_norm = MSE(signed_pred, target) * twopow_n

    return (
        -1,
        sign,
        signed_pred,
        normalized_hs_norm,
        sign_agreement,
        all_sign_agreement,
        all_normalized_hs_norm,
    )


def recompute_s3(trial_df):
    all_sign_agreement_per_trial = []
    all_normalized_hs_norm_per_trial = []

    _, allPs = generate_all_Ps_stacked(trial_df.iloc[0]["n"])
    for ridx, row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        # print(row['state_type'])

        new_measurement_block = row["all_measurements"]
        n = int(row["n"])
        sign_table = np.array(
            [
                [1, 1, 1, 1],  # I
                [1, 1, -1, -1],  # X
                [-1, 1, 1, -1],  # Y
                [1, -1, 1, -1],  # Z
            ],
            dtype=np.int8,
        )

        # threshold target_up
        eps_sampling = row["epsilon_sampling"]
        up = row["pred"]
        significant_pauli_idcs_pred = np.where(up >= eps_sampling)[0]

        target_up_thresholded = row["thresholded_target"]
        significant_pauli_idcs_target = np.where(np.abs(target_up_thresholded) > 0)[0]
        sigma = row["final_state"]
        trPsigma = generate_u_P_exact_numba(sigma, allPs)

        accum_sign = np.zeros(4**n, dtype=np.float64)
        (
            M_candidate,
            sign,
            signed_pred,
            normalized_hs_norm_thresholded,
            sign_agreement,
            all_sign_agreement,
            all_normalized_hs_norm,
        ) = update_accum_sign_and_check_s3(
            new_measurement_block,
            accum_sign,
            pauli_labels=compute_pauli_labels(n),
            sign_table=sign_table,
            target=target_up_thresholded,
            bell_sampl_up=up,
            trPsigma=trPsigma,
            significant_pauli_idcs_target=significant_pauli_idcs_target,
            significant_pauli_idcs_pred=significant_pauli_idcs_pred,
            threshold=0.9,
            start_count=0,
        )

        all_sign_agreement_per_trial.append(all_sign_agreement)
        all_normalized_hs_norm_per_trial.append(all_normalized_hs_norm)

        # print("M_candidate", M_candidate)

    return all_sign_agreement_per_trial, all_normalized_hs_norm_per_trial
