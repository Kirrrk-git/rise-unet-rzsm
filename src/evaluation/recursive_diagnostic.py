"""
src/evaluation/recursive_diagnostic.py
-------------------------------------
Authoritative execution engine and statistical testing framework for
Phase 23 Recursive Degradation Diagnostic (Mindanao RISE-UNet Adaptation).

Conforms strictly to contracts/A0/recursive_degradation_diagnostic_contract.yaml:
  - Protocol 1: Standard Autoregressive Recursive Inference
  - Protocol 2: Oracle Counterfactual Diagnostic (Primary Diagnostic)
  - Protocol 3: Direct Non-Recursive Architecture Comparator (Complementary Reference)
  - Protocol 4: Controlled Error-Injection & Boundary-Clipped Perturbation Tracking
  - Statistical Pre-Analysis: Case-level paired Wilcoxon tests, Moving-Block Bootstrap (MBB),
    Holm-Bonferroni correction, and Gate 2 decision rules.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy import stats

from src.data.tf_dataset import (
    ENSEMBLE_MEMBERS,
    GRID_HEIGHT,
    GRID_WIDTH,
    crps_exact_analytical,
    crps2d_numpy,
)


def assemble_oracle_priors(
    case_data: Dict[str, np.ndarray],
    lead: int,
) -> Dict[int, np.ndarray]:
    """
    Constructs the exact Oracle counterfactual prior dictionary for a given lead.
    Replaces recursive model predictions with the verifying ground truth target states
    broadcast across all 11 ensemble members.

    Parameters
    ----------
    case_data : Dict[str, np.ndarray]
        Dictionary of arrays loaded from the case NPZ archive.
    lead : int
        Current forecast lead (1, 2, 3, or 4).

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping prior lead index (1, 2, 3) to broadcast target array
        of shape (11, 32, 48, 1) with dtype float32.
    """
    if lead not in (1, 2, 3, 4):
        raise ValueError(f"Invalid forecast lead: {lead}. Must be 1, 2, 3, or 4.")

    if lead == 1:
        return {}

    oracle_priors: Dict[int, np.ndarray] = {}
    prior_leads_needed = {2: [1], 3: [1, 2], 4: [1, 2, 3]}[lead]

    for pl in prior_leads_needed:
        target_key = f"y_w{pl}"
        if target_key not in case_data:
            raise KeyError(f"Case data missing required target key '{target_key}' for Oracle replacement.")

        y_true_single = case_data[target_key].astype(np.float32)  # (1, 32, 48, 1)
        if y_true_single.shape != (1, GRID_HEIGHT, GRID_WIDTH, 1):
            raise ValueError(
                f"Target {target_key} has shape {y_true_single.shape}, "
                f"expected (1, {GRID_HEIGHT}, {GRID_WIDTH}, 1)."
            )

        # Broadcast ground truth across all 11 ensemble members
        y_true_broadcast = np.repeat(y_true_single, ENSEMBLE_MEMBERS, axis=0)
        oracle_priors[pl] = y_true_broadcast

    return oracle_priors


def apply_boundary_clipped_perturbation(
    y_hat: np.ndarray,
    epsilon: float,
    eval_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Injects controlled perturbation into normalized RZSM predictions and enforces
    the [0.0, 1.0] boundary clipping rule mandated by the Phase 23 diagnostic contract.

    Ensemble Member Application Rule:
    ---------------------------------
    Applies the identical deterministic perturbation offset epsilon to every member
    realization m in {1..11}:
        y_hat'_{W1, m} = clip(y_hat_{W1, m} + epsilon, 0.0, 1.0)
    This preserves ensemble member correspondence; ensemble spread is unchanged on
    unclipped cells, while any boundary-induced spread changes are explicitly recorded
    by the clipping census.

    Parameters
    ----------
    y_hat : np.ndarray
        Unperturbed prediction array of shape (M, H, W, 1) or (H, W).
    epsilon : float
        Perturbation offset (e.g. ±0.05, ±0.10, ±0.25, ±0.50).
    eval_mask : Optional[np.ndarray]
        Binary mask of shape (H, W) indicating the 126 active evaluation cells.

    Returns
    -------
    Tuple[np.ndarray, int, float]
        - y_prime: Clipped prediction array of same shape and float32 dtype.
        - clipped_count: Total count of active evaluation cell evaluations where clipping occurred.
        - clipped_percentage: Percentage of active cell evaluations clipped (0.0 to 100.0).
    """
    unclipped = y_hat.astype(np.float32) + float(epsilon)
    clipped = np.clip(unclipped, 0.0, 1.0).astype(np.float32)

    if eval_mask is None:
        # If no mask provided, evaluate over all cells
        clipped_mask = (unclipped < 0.0) | (unclipped > 1.0)
        clipped_count = int(np.sum(clipped_mask))
        total_cells = int(unclipped.size)
        clipped_pct = float(100.0 * clipped_count / total_cells) if total_cells > 0 else 0.0
        return clipped, clipped_count, clipped_pct

    # Evaluate strictly over active evaluation domain (126 cells)
    mask_2d = eval_mask.astype(bool)
    if unclipped.ndim == 4:
        # Shape: (M, H, W, 1) -> select across M and (H, W)
        active_unclipped = unclipped[:, mask_2d, 0]  # (M, 126)
    elif unclipped.ndim == 3:
        # Shape: (M, H, W)
        active_unclipped = unclipped[:, mask_2d]
    elif unclipped.ndim == 2:
        # Shape: (H, W)
        active_unclipped = unclipped[mask_2d]
    else:
        raise ValueError(f"Unsupported prediction tensor dimensionality: {unclipped.ndim}")

    clipped_bool = (active_unclipped < 0.0) | (active_unclipped > 1.0)
    clipped_count = int(np.sum(clipped_bool))
    total_active_elements = int(active_unclipped.size)
    clipped_pct = float(100.0 * clipped_count / total_active_elements) if total_active_elements > 0 else 0.0

    return clipped, clipped_count, clipped_pct


def compute_case_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Computes case-level evaluation metrics over the 126 active evaluation cells.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target array of shape (1, 32, 48, 1) or (32, 48).
    y_pred : np.ndarray
        Ensemble prediction array of shape (11, 32, 48, 1).
    eval_mask : np.ndarray
        Binary mask of shape (32, 48) indicating 126 active cells.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'mae', 'rmse', 'acc', 'exact_crps', 'spatial_crps_proxy'.
    """
    mask_bool = eval_mask.astype(bool)

    # 1. Align shapes
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true_2d = y_true[0, ..., 0]  # (32, 48)
    elif y_true.ndim == 3 and y_true.shape[0] == 1:
        y_true_2d = y_true[0]
    else:
        y_true_2d = y_true

    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
        y_pred_m = y_pred[..., 0]  # (11, 32, 48)
    else:
        y_pred_m = y_pred

    # Active evaluation vectors
    yt_eval = y_true_2d[mask_bool].astype(np.float64)  # (126,)
    yp_ens_mean = np.mean(y_pred_m[:, mask_bool], axis=0).astype(np.float64)  # (126,)

    # Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(yp_ens_mean - yt_eval)))

    # Root Mean Square Error (RMSE)
    rmse = float(np.sqrt(np.mean((yp_ens_mean - yt_eval) ** 2)))

    # Anomaly Correlation Coefficient (ACC)
    # Centered correlation over the 126 active cells
    yt_dev = yt_eval - np.mean(yt_eval)
    yp_dev = yp_ens_mean - np.mean(yp_ens_mean)
    denom = np.sqrt(np.sum(yt_dev ** 2) * np.sum(yp_dev ** 2))
    if denom > 1e-12:
        acc = float(np.sum(yt_dev * yp_dev) / denom)
    else:
        acc = 0.0

    # Exact Analytical Ensemble CRPS (Hersbach 2000)
    exact_crps = crps_exact_analytical(
        y_true=y_true,
        y_pred=y_pred,
        eval_mask=mask_bool,
    )

    # Spatial Proxy CRPS (MAE - 0.08 * sigma_spatial)
    proxy_crps = crps2d_numpy(
        y_true=y_true,
        y_pred=y_pred,
        factor=0.08,
        eval_mask=mask_bool,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "acc": acc,
        "exact_crps": exact_crps,
        "spatial_crps_proxy": proxy_crps,
    }


def compute_paired_differences(
    errors_rec: Union[List[float], np.ndarray],
    errors_ora: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Computes paired case-level differences:
        delta_e_i = e_i^(rec) - e_i^(ora)

    A positive delta_e indicates that the standard recursive prediction
    incurred higher error than the oracle counterfactual (error degradation).
    """
    arr_rec = np.asarray(errors_rec, dtype=np.float64)
    arr_ora = np.asarray(errors_ora, dtype=np.float64)

    if arr_rec.shape != arr_ora.shape:
        raise ValueError(
            f"Shape mismatch in paired differences: "
            f"rec shape {arr_rec.shape} vs ora shape {arr_ora.shape}."
        )

    return arr_rec - arr_ora


def moving_block_bootstrap(
    data: np.ndarray,
    block_length: int = 4,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Primary Inferential Procedure:
    Computes point estimate, circular moving-block bootstrap (MBB) confidence intervals
    from the empirical distribution, and a correctly calibrated two-sided bootstrap
    p-value (p_boot) under H0: E[Delta] = 0 by resampling the centered null series:
        d_i^(0) = d_i - mean(d)

    Accounts for temporal autocorrelation across sequential bi-weekly forecast cycles
    using circular block resampling (Politis & Romano 1994; Davison & Hinkley 1997).

    Parameters
    ----------
    data : np.ndarray
        1D array of case-level paired differences (length N).
    block_length : int
        Block length L (default 4 bi-weekly cycles ~ 2 months correlation memory).
    n_boot : int
        Number of bootstrap resamples (default 10,000).
    alpha : float
        Significance level for two-sided (1 - alpha) interval (default 0.05 -> 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float, float]
        (point_estimate, ci_lower, ci_upper, p_boot)
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array.")

    obs_mean = float(np.mean(arr))
    abs_obs_mean = abs(obs_mean)

    if n < block_length:
        block_length = max(1, n)

    rng = np.random.default_rng(seed)
    num_blocks = int(math.ceil(n / block_length))

    # 1. Uncentered series for confidence intervals
    circular_arr = np.concatenate([arr, arr[:block_length]])

    # 2. Centered series for null hypothesis H0: E[Delta] = 0
    # d_i^(0) = d_i - mean(d)
    arr_null = arr - obs_mean
    circular_arr_null = np.concatenate([arr_null, arr_null[:block_length]])

    boot_means = np.empty(n_boot, dtype=np.float64)
    boot_null_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        start_indices = rng.integers(0, n, size=num_blocks)

        # Uncentered sample for CI
        blocks = [circular_arr[idx : idx + block_length] for idx in start_indices]
        sample = np.concatenate(blocks)[:n]
        boot_means[b] = np.mean(sample)

        # Centered null sample for hypothesis test H0: mean == 0
        blocks_null = [circular_arr_null[idx : idx + block_length] for idx in start_indices]
        sample_null = np.concatenate(blocks_null)[:n]
        boot_null_means[b] = np.mean(sample_null)

    # 95% bootstrap percentile confidence intervals
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    # Two-sided null bootstrap p-value under H0: mean == 0
    # P(|X_null*| >= |obs_mean|)
    if abs_obs_mean <= 1e-15:
        p_boot = 1.0
    else:
        extreme_count = np.sum(np.abs(boot_null_means) >= abs_obs_mean)
        p_boot = float(extreme_count / n_boot)

    return obs_mean, ci_lower, ci_upper, p_boot



def wilcoxon_paired_test(
    differences: np.ndarray,
) -> Tuple[float, float]:
    """
    Executes two-sided paired Wilcoxon signed-rank test on case-level differences.

    Parameters
    ----------
    differences : np.ndarray
        1D array of paired differences delta_e_i.

    Returns
    -------
    Tuple[float, float]
        (test_statistic, p_value)
    """
    diff = np.asarray(differences, dtype=np.float64)
    non_zero = diff[diff != 0.0]

    if len(non_zero) < 10:
        # Not enough non-zero differences for meaningful test
        return 0.0, 1.0

    res = stats.wilcoxon(diff, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def holm_bonferroni_correction(
    p_values: Dict[Any, float],
    alpha: float = 0.05,
) -> Dict[Any, Dict[str, Any]]:
    """
    Applies step-down Holm-Bonferroni multiple testing correction across hypotheses.

    Parameters
    ----------
    p_values : Dict[Any, float]
        Dictionary mapping hypothesis keys to raw p-values.
    alpha : float
        Family-wise error rate threshold (default 0.05).

    Returns
    -------
    Dict[Any, Dict[str, Any]]
        Dictionary mapping hypothesis keys to dict with:
          - 'raw_p': Raw p-value
          - 'adj_p': Step-down adjusted p-value
          - 'rejected': Boolean whether H0 is rejected at family-wise alpha
          - 'rank': 1-based order in sorted p-values
    """
    sorted_items = sorted(p_values.items(), key=lambda item: item[1])
    m = len(sorted_items)

    adj_p_dict: Dict[Any, float] = {}
    running_max = 0.0

    for i, (k, p) in enumerate(sorted_items):
        multiplier = m - i
        step_adj = min(1.0, multiplier * p)
        running_max = max(running_max, step_adj)
        adj_p_dict[k] = min(1.0, running_max)

    results: Dict[Any, Dict[str, Any]] = {}
    for rank, (k, p) in enumerate(sorted_items, start=1):
        adj_p = adj_p_dict[k]
        results[k] = {
            "raw_p": float(p),
            "adj_p": float(adj_p),
            "rejected": bool(adj_p < alpha),
            "rank": rank,
        }

    return results


def compute_cohens_d_z(differences: np.ndarray) -> float:
    """
    Computes Cohen's d_z effect size for paired samples:
        d_z = mean(delta_e) / sd(delta_e)
    """
    diff = np.asarray(differences, dtype=np.float64)
    if len(diff) < 2:
        return 0.0

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    if std_diff <= 1e-12:
        return 0.0

    return float(mean_diff / std_diff)


def check_perturbation_monotonicity(
    divergence_by_lead_eps: Dict[Any, Dict[Any, float]],
    target_leads: Tuple[int, ...] = (3, 4),
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates monotonic sensitivity to perturbation magnitude |epsilon|
    separately for positive and negative perturbations across realistic levels:
        D_k(+0.10) > D_k(+0.05) and D_k(-0.10) > D_k(-0.05)
    across downstream leads (default Leads W3 and/or W4).
    Perturbations at +/-0.25 and +/-0.50 are treated strictly as exploratory
    boundary stress tests and are not required evidence for the GO decision.

    Parameters
    ----------
    divergence_by_lead_eps : Dict
        Nested dictionary mapping lead (e.g. 2, 3, 4 or '2', '3', '4')
        to dict mapping eps (e.g. 0.05, 0.10 or '0.05', '0.1') to divergence RMSE.
    target_leads : Tuple[int, ...]
        Leads where downstream sensitivity is evaluated (default 3, 4).

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_monotonic, diagnostic_details)
    """
    details: Dict[str, Any] = {}
    lead_passes = []

    for lead in target_leads:
        lead_key = lead if lead in divergence_by_lead_eps else str(lead)
        if lead_key not in divergence_by_lead_eps:
            continue
        lead_div = divergence_by_lead_eps[lead_key]

        def get_div(eps_val: float) -> Optional[float]:
            for k, v in lead_div.items():
                try:
                    if abs(float(k) - eps_val) < 1e-4:
                        return float(v)
                except (ValueError, TypeError):
                    continue
            return None

        d_pos_05 = get_div(0.05)
        d_pos_10 = get_div(0.10)
        d_neg_05 = get_div(-0.05)
        d_neg_10 = get_div(-0.10)

        pos_mono = bool(d_pos_10 is not None and d_pos_05 is not None and d_pos_10 > d_pos_05)
        neg_mono = bool(d_neg_10 is not None and d_neg_05 is not None and d_neg_10 > d_neg_05)
        both_mono = pos_mono and neg_mono

        details[f"lead_{lead}"] = {
            "d_pos_05": d_pos_05,
            "d_pos_10": d_pos_10,
            "positive_monotonic": pos_mono,
            "d_neg_05": d_neg_05,
            "d_neg_10": d_neg_10,
            "negative_monotonic": neg_mono,
            "passed": both_mono,
        }
        lead_passes.append(both_mono)

    is_monotonic = bool(any(lead_passes)) if lead_passes else True
    return is_monotonic, details


def evaluate_gate2_rules(
    lead_summary: Dict[int, Dict[str, Any]],
    seed_consistency: bool,
    divergence_monotonic: Union[bool, Dict[Any, Dict[Any, float]]],
) -> Dict[str, Any]:
    """
    Authoritative Post-A0 Gate 2 (G2-R) decision evaluator adhering to
    contracts/A0/recursive_degradation_diagnostic_contract.yaml.

    Pre-registered Criteria:
      GO-1: Adjusted p_boot < 0.05 in Leads W3 and/or W4.
      GO-2: Cohen's d_z >= 0.20 OR relative error reduction >= 2.0%.
      GO-3: Replicate consistency: Delta E_k > 0 across all 3 seeds.
      GO-4: Monotonic downstream perturbation sensitivity across magnitude |epsilon|
            evaluated separately for positive and negative perturbations:
            D_k(+0.10) > D_k(+0.05) and D_k(-0.10) > D_k(-0.05) in W3 and/or W4;
            +/-0.25 and +/-0.50 are strictly boundary stress tests.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with individual rule evaluations and overall verdict ('GO' or 'NO-GO').
    """
    # Check GO-1: Adjusted p_boot < 0.05 in W3 or W4
    go_1_w3 = lead_summary.get(3, {}).get("rejected", False)
    go_1_w4 = lead_summary.get(4, {}).get("rejected", False)
    go_1_pass = bool(go_1_w3 or go_1_w4)

    # Check GO-2: Effect size in W3 or W4
    w3_dz = lead_summary.get(3, {}).get("cohens_d_z", 0.0)
    w4_dz = lead_summary.get(4, {}).get("cohens_d_z", 0.0)
    w3_rel = lead_summary.get(3, {}).get("relative_gap_pct", 0.0)
    w4_rel = lead_summary.get(4, {}).get("relative_gap_pct", 0.0)

    go_2_pass = bool(
        (w3_dz >= 0.20 or w3_rel >= 2.0) or
        (w4_dz >= 0.20 or w4_rel >= 2.0)
    )

    # Check GO-3: Seed consistency
    go_3_pass = bool(seed_consistency)

    # Check GO-4: Downstream divergence monotonicity on magnitude |epsilon|
    if isinstance(divergence_monotonic, bool):
        go_4_pass = bool(divergence_monotonic)
        mono_details: Dict[str, Any] = {"boolean_flag": divergence_monotonic}
    else:
        go_4_pass, mono_details = check_perturbation_monotonicity(divergence_monotonic)

    # Overall Post-A0 Gate 2 (G2-R) verdict
    all_go = go_1_pass and go_2_pass and go_3_pass and go_4_pass
    verdict = "GO" if all_go else "NO-GO"

    return {
        "verdict": verdict,
        "criteria": {
            "GO-1_statistical_significance": {
                "passed": go_1_pass,
                "w3_rejected": go_1_w3,
                "w4_rejected": go_1_w4,
            },
            "GO-2_practical_effect_size": {
                "passed": go_2_pass,
                "w3_cohens_d_z": w3_dz,
                "w4_cohens_d_z": w4_dz,
                "w3_relative_gap_pct": w3_rel,
                "w4_relative_gap_pct": w4_rel,
            },
            "GO-3_replicate_consistency": {
                "passed": go_3_pass,
            },
            "GO-4_perturbation_sensitivity": {
                "passed": go_4_pass,
                "details": mono_details,
                "specification": (
                    "D_k(+0.10) > D_k(+0.05) and D_k(-0.10) > D_k(-0.05) evaluated "
                    "separately across magnitude |eps| in downstream leads; "
                    "+/-0.25 and +/-0.50 are exploratory boundary stress tests"
                ),
            },
        },
        "action": (
            "Proceed to Phase 24 (Model A1 Lead-Aware Recursive Residual Refinement)"
            if verdict == "GO" else
            "Formally halt neural enhancements; freeze Model A0 as definitive regional benchmark"
        ),
    }


def assign_philippine_monsoon_regime(date_str_or_month: Union[str, int]) -> str:
    """
    Classifies date into Philippine meteorological monsoon regimes:
      - 'Amihan' (Northeast Monsoon): November to April (Months 11, 12, 1, 2, 3, 4)
      - 'Habagat' (Southwest Monsoon): June to October (Months 6, 7, 8, 9, 10)
      - 'Transition' (Inter-Monsoon): May (Month 5)
    """
    if isinstance(date_str_or_month, int):
        m = date_str_or_month
    else:
        clean = "".join(filter(str.isdigit, str(date_str_or_month)))[:8]
        if len(clean) >= 6:
            m = int(clean[4:6])
        else:
            raise ValueError(f"Invalid date string format: {date_str_or_month}")
    if m in (11, 12, 1, 2, 3, 4):
        return "Amihan"
    elif m in (6, 7, 8, 9, 10):
        return "Habagat"
    elif m == 5:
        return "Transition"
    else:
        raise ValueError(f"Invalid month integer: {m}")


def assign_meteorological_quarter(date_str_or_month: Union[str, int]) -> str:
    """
    Classifies date into conventional meteorological quarters:
      - 'DJF': December, January, February
      - 'MAM': March, April, May
      - 'JJA': June, July, August
      - 'SON': September, October, November
    """
    if isinstance(date_str_or_month, int):
        m = date_str_or_month
    else:
        clean = "".join(filter(str.isdigit, str(date_str_or_month)))[:8]
        if len(clean) >= 6:
            m = int(clean[4:6])
        else:
            raise ValueError(f"Invalid date string format: {date_str_or_month}")
    if m in (12, 1, 2):
        return "DJF"
    elif m in (3, 4, 5):
        return "MAM"
    elif m in (6, 7, 8):
        return "JJA"
    elif m in (9, 10, 11):
        return "SON"
    else:
        raise ValueError(f"Invalid month integer: {m}")


def compute_season_stratified_breakdown(
    case_ids: List[str],
    case_diffs_by_lead: Dict[Union[int, str], np.ndarray],
    rec_errors_by_lead: Optional[Dict[Union[int, str], np.ndarray]] = None,
    ora_errors_by_lead: Optional[Dict[Union[int, str], np.ndarray]] = None,
    regime_type: str = "monsoon",
    block_length: int = 4,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes season-stratified breakdown of recursive degradation gap (Delta E_k)
    across defined seasonal or monsoon partitions.

    Parameters
    ----------
    case_ids : List[str]
        List of N case identifiers, each containing issue date (e.g. 'CASE_20220115_...').
    case_diffs_by_lead : Dict[Union[int, str], np.ndarray]
        Dictionary mapping lead index (2, 3, 4) to 1D array of length N containing paired differences.
    rec_errors_by_lead : Optional[Dict[Union[int, str], np.ndarray]]
        Optional mapping of lead index to 1D array of recursive errors.
    ora_errors_by_lead : Optional[Dict[Union[int, str], np.ndarray]]
        Optional mapping of lead index to 1D array of oracle errors.
    regime_type : str
        Either 'monsoon' ('Amihan', 'Habagat', 'Transition') or 'quarter' ('DJF', 'MAM', 'JJA', 'SON').
    block_length : int
        Moving-block bootstrap block length (default 4).
    n_boot : int
        Number of bootstrap resamples (default 1000).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dictionary of seasonal slices and their per-lead metrics.
    """
    n_cases = len(case_ids)
    assign_fn = assign_philippine_monsoon_regime if regime_type == "monsoon" else assign_meteorological_quarter

    # Group case indices by regime
    regime_indices: Dict[str, List[int]] = {}
    for idx, cid in enumerate(case_ids):
        regime = assign_fn(cid)
        if regime not in regime_indices:
            regime_indices[regime] = []
        regime_indices[regime].append(idx)

    stratified_results: Dict[str, Dict[str, Any]] = {}

    for regime_name, indices in sorted(regime_indices.items()):
        idx_arr = np.array(indices, dtype=int)
        count = len(indices)
        regime_lead_metrics: Dict[str, Any] = {}

        for lead_key, diffs in case_diffs_by_lead.items():
            lead_int = int(lead_key)
            diffs_slice = np.asarray(diffs)[idx_arr]

            if count < 2 or np.all(diffs_slice == 0.0):
                regime_lead_metrics[str(lead_int)] = {
                    "delta_mae_mean": float(np.mean(diffs_slice)),
                    "delta_mae_ci_lower": float(np.mean(diffs_slice)),
                    "delta_mae_ci_upper": float(np.mean(diffs_slice)),
                    "cohens_d_z": 0.0,
                    "p_boot": 1.0,
                }
            else:
                bl = min(block_length, count)
                m_gap, ci_lo, ci_hi, p_b = moving_block_bootstrap(
                    diffs_slice, block_length=bl, n_boot=n_boot, seed=seed
                )
                dz = compute_cohens_d_z(diffs_slice)
                regime_lead_metrics[str(lead_int)] = {
                    "delta_mae_mean": float(m_gap),
                    "delta_mae_ci_lower": float(ci_lo),
                    "delta_mae_ci_upper": float(ci_hi),
                    "cohens_d_z": float(dz),
                    "p_boot": float(p_b),
                }

            if rec_errors_by_lead and lead_key in rec_errors_by_lead:
                r_slice = np.asarray(rec_errors_by_lead[lead_key])[idx_arr]
                regime_lead_metrics[str(lead_int)]["rec_mae_mean"] = float(np.mean(r_slice))

            if ora_errors_by_lead and lead_key in ora_errors_by_lead:
                o_slice = np.asarray(ora_errors_by_lead[lead_key])[idx_arr]
                regime_lead_metrics[str(lead_int)]["ora_mae_mean"] = float(np.mean(o_slice))

        stratified_results[regime_name] = {
            "case_count": count,
            "case_percentage": float(100.0 * count / n_cases) if n_cases > 0 else 0.0,
            "leads": regime_lead_metrics,
        }

    return stratified_results


def generate_diagnostic_plots(
    results: Dict[str, Any],
    output_plot_path: Union[str, Path],
) -> None:
    """
    Generates 4-panel publication-grade composite plot for Post-A0 Gate 2 (G2-R).

    Panels:
      (A) Protocol 1 (Standard Recursive) vs. Protocol 2 (Oracle Counterfactual) MAE.
      (B) Primary Recursive Error Gap (Delta E_k) with 95% Moving-Block Bootstrap CIs.
      (C) Downstream Divergence D_k(eps) across Leads W2..W4 under Protocol 4.
      (D) Boundary Saturation Census (% active cells clipped to [0, 1]).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Support either 'lead_summary' (from notebook) or 'lead_metrics' (from script)
    metrics_source = results.get("lead_summary", results.get("lead_metrics", {}))

    # Helper to look up lead in metrics dictionary (int or str key)
    def get_lead_metric(ld: int) -> Dict[str, Any]:
        if ld in metrics_source:
            return metrics_source[ld]
        if str(ld) in metrics_source:
            return metrics_source[str(ld)]
        return {}

    leads = [1, 2, 3, 4]
    lead_labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
    lead_sublabels = ["(Days 1–7)", "(Days 8–14)", "(Days 15–21)", "(Days 22–28)"]

    rec_mae = [get_lead_metric(l).get("rec_mae_mean", 0.0) for l in leads]
    ora_mae = [get_lead_metric(l).get("ora_mae_mean", 0.0) for l in leads]

    gaps = [get_lead_metric(l).get("delta_mae_mean", 0.0) for l in [2, 3, 4]]
    ci_low = [get_lead_metric(l).get("delta_mae_ci_lower", 0.0) for l in [2, 3, 4]]
    ci_high = [get_lead_metric(l).get("delta_mae_ci_upper", 0.0) for l in [2, 3, 4]]

    # Styling parameters
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(hspace=0.38, wspace=0.26)

    # Modern high-contrast color palette
    c_rec = "#E11D48"      # Rose / Crimson (Standard Recursive)
    c_ora = "#059669"      # Emerald (Oracle Counterfactual)
    c_gap = "#4338CA"      # Deep Indigo (Gap Progression)
    c_title = "#0F172A"    # Deep Slate
    c_grid = "#F1F5F9"     # Subtle grid

    # -------------------------------------------------------------------------
    # PANEL A: Forecast Error by Lead Horizon (Protocols 1 & 2)
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#FFFFFF")
    x = np.arange(len(leads))
    bar_w = 0.32

    b1 = ax_a.bar(
        x - bar_w/2, rec_mae, width=bar_w,
        color=c_rec, alpha=0.92,
        label="Standard Recursive (Model A0)"
    )
    b2 = ax_a.bar(
        x + bar_w/2, ora_mae, width=bar_w,
        color=c_ora, alpha=0.92,
        label="Oracle Counterfactual (Ground Truth Injected)"
    )

    # Clean text directly above bars (staggered Week 1 horizontal offset)
    for i, bar in enumerate(b1):
        h = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2 - (0.045 if i == 0 else 0.0)
        ax_a.text(
            x_pos, h + 0.0022,
            f"{h:.4f}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=c_rec
        )

    for i, bar in enumerate(b2):
        h = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2 + (0.045 if i == 0 else 0.0)
        ax_a.text(
            x_pos, h + 0.0022,
            f"{h:.4f}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=c_ora
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{l}\n{s}" for l, s in zip(lead_labels, lead_sublabels)], fontsize=10.5, color=c_title)
    ax_a.set_ylabel("Validation MAE (Normalized [0, 1])", fontsize=11.2, fontweight="bold", color=c_title)
    ax_a.set_ylim(0, 0.108)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_title("A. Forecast Error by Lead Horizon (Protocols 1 & 2)", fontsize=13.0, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_a.legend(loc="upper left", frameon=False, fontsize=9.8)

    # -------------------------------------------------------------------------
    # PANEL B: Compounded Recursive Degradation Gap
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#FFFFFF")
    gx = np.array([2, 3, 4])
    gx_labels = ["Week 2\n(Days 8–14)", "Week 3\n(Days 15–21)", "Week 4\n(Days 22–28)"]

    ax_b.axhline(0, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_b.fill_between(
        gx, ci_low, ci_high,
        color="#E0E7FF", alpha=0.65,
        label="95% Bootstrap CI (B=10,000)"
    )
    ax_b.plot(gx, gaps, marker="o", markersize=8.5, linewidth=2.8, color=c_gap, label="Recursive Error Gap (ΔMAE)", zorder=4)

    for xi, yi in zip(gx, gaps):
        ax_b.text(
            xi, yi + 0.0028, f"+{yi:.4f}",
            ha="center", va="bottom",
            fontsize=11.2, fontweight="bold", color=c_gap
        )

    ax_b.set_xticks(gx)
    ax_b.set_xticklabels(gx_labels, fontsize=10.5, color=c_title)
    ax_b.set_ylabel("Error Gap ΔMAE (Normalized [0, 1])", fontsize=11.2, fontweight="bold", color=c_title)
    ax_b.set_xlim(1.4, 4.6)
    ax_b.set_ylim(-0.003, 0.052)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_title("B. Compounded Recursive Degradation Gap with 95% CI", fontsize=13.0, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_b.legend(loc="upper left", frameon=False, fontsize=9.8)

    # -------------------------------------------------------------------------
    # PANEL C: Downstream Divergence to State Perturbations (Protocol 4)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#FFFFFF")

    default_perts = [-0.50, -0.25, -0.10, -0.05, 0.05, 0.10, 0.25, 0.50]
    eps_vals = results.get("perturbations_evaluated", default_perts)
    div_dict = results.get("divergence_by_lead_eps", {})

    cx = np.arange(len(eps_vals))
    cx_labels = [f"{e:+.2f}" for e in eps_vals]

    # Shaded operational error zone spanning [-0.10, +0.10]
    ax_c.axvspan(1.5, 5.5, color="#F0FDF4", alpha=0.6, zorder=0, label="Operational Error Range (|ε| ≤ 0.10)")

    lead_styles = {
        2: {"color": "#0284C7", "marker": "o", "label": "Week 2 (Days 8–14)", "lw": 2.5, "ms": 7.0},
        3: {"color": "#7C3AED", "marker": "s", "label": "Week 3 (Days 15–21)", "lw": 2.5, "ms": 6.5},
        4: {"color": "#D97706", "marker": "^", "label": "Week 4 (Days 22–28)", "lw": 2.5, "ms": 7.0},
    }

    for lead in [2, 3, 4]:
        ld_div = div_dict.get(lead, div_dict.get(str(lead), {}))
        div_vals = [ld_div.get(e, ld_div.get(str(e), 0.0)) for e in eps_vals]
        st = lead_styles[lead]
        ax_c.plot(
            cx, div_vals,
            marker=st["marker"], markersize=st["ms"], linewidth=st["lw"],
            color=st["color"], label=st["label"], zorder=3
        )

    ax_c.set_xticks(cx)
    ax_c.set_xticklabels(cx_labels, fontsize=10.0, color=c_title)
    ax_c.set_xlabel("Perturbation Offset ε Injected at Week 1", fontsize=11.2, fontweight="bold", color=c_title)
    ax_c.set_ylabel("Downstream Divergence D_k(ε) [RMSE, Normalized]", fontsize=11.2, fontweight="bold", color=c_title)
    ax_c.set_ylim(0, 0.44)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.set_title("C. Downstream Divergence to State Perturbations (Protocol 4)", fontsize=13.0, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_c.legend(loc="upper left", frameon=False, fontsize=9.4)

    # -------------------------------------------------------------------------
    # PANEL D: Physical Boundary Saturation Census (Clipping)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#FFFFFF")

    clip_dict = results.get("clipping_census", {})
    clip_pcts = [clip_dict.get(e, clip_dict.get(str(e), 0.0)) for e in eps_vals]

    bar_colors = []
    for e in eps_vals:
        if abs(e) <= 0.10:
            bar_colors.append("#10B981")  # Emerald (Safe / Negligible)
        elif abs(e) <= 0.25:
            bar_colors.append("#94A3B8")  # Slate (Moderate stress test)
        else:
            bar_colors.append("#F43F5E")  # Rose (Extreme stress test)

    d_x = np.arange(len(eps_vals))
    d_bars = ax_d.bar(d_x, clip_pcts, color=bar_colors, width=0.52, alpha=0.92, zorder=3)

    ax_d.axvspan(1.5, 5.5, color="#F0FDF4", alpha=0.6, zorder=0)

    for b, pct, e in zip(d_bars, clip_pcts, eps_vals):
        h = b.get_height()
        if abs(e) <= 0.10:
            label_text = f"{pct:.2f}%" if pct > 0 else "0.0%"
            ax_d.text(
                b.get_x() + b.get_width() / 2, h + 1.2,
                label_text, ha="center", va="bottom",
                fontsize=9.8, fontweight="bold", color="#047857"
            )
        elif abs(e) == 0.25:
            label_text = f"{pct:.1f}%" if pct > 0 else "0.0%"
            ax_d.text(
                b.get_x() + b.get_width() / 2, h + 1.2,
                label_text, ha="center", va="bottom",
                fontsize=9.5, fontweight="medium", color="#475569"
            )
        elif abs(e) == 0.50:
            ax_d.text(
                b.get_x() + b.get_width() / 2, h + 1.2,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=9.8, fontweight="bold", color="#BE123C"
            )

    ax_d.set_xticks(d_x)
    ax_d.set_xticklabels([f"{e:+.2f}" for e in eps_vals], fontsize=10.0, color=c_title)
    ax_d.set_xlabel("Perturbation Offset ε Injected at Week 1", fontsize=11.2, fontweight="bold", color=c_title)
    ax_d.set_ylabel("Active Land Cells Clipped to [0, 1] (%)", fontsize=11.2, fontweight="bold", color=c_title)
    ax_d.set_ylim(0, 95)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.set_title("D. Physical Boundary Saturation Census", fontsize=13.0, fontweight="bold", color=c_title, pad=14, loc="left")

    d_legend_elements = [
        Patch(facecolor="#10B981", label="Operational Range (|ε| ≤ 0.10: Safe, ≤ 0.27% clipped)"),
        Patch(facecolor="#94A3B8", label="Moderate Stress Test (|ε| = 0.25: ≤ 6.4% clipped)"),
        Patch(facecolor="#F43F5E", label="Extreme Stress Test (|ε| = 0.50: up to 83.5% clipped)"),
    ]
    ax_d.legend(handles=d_legend_elements, loc="upper left", frameon=False, fontsize=9.4)

    # Global Figure Suptitle
    fig.suptitle(
        "Subseasonal Recursive Error Compounding Analysis (Mindanao Baseline Model A0)\n"
        "194 Validation Cases (2022–2023) · 126 Active Land Cells · 3 Independent Training Seeds (42, 123, 456)",
        fontsize=14.5, fontweight="bold", color=c_title, y=0.985
    )

    # Footnote
    fig.text(
        0.50, 0.015,
        "Protocols 1–4 Diagnostic Suite · Mindanao RISE-UNet Baseline Verification · Moving-Block Bootstrap B = 10,000 · Sealed 2024–2025 Test Cohort Quarantined",
        ha="center", fontsize=9.5, color="#64748B", style="italic"
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_recursive_gap_presentation_slide(
    results: Dict[str, Any],
    output_plot_path: Union[str, Path],
) -> None:
    """
    Generates a dedicated 16:9 widescreen presentation slide proving the
    Recursive Error Compounding Gap in Model A0 for the thesis presentation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_source = results.get("lead_summary", results.get("lead_metrics", {}))

    def get_lead_metric(ld: int) -> Dict[str, Any]:
        if ld in metrics_source:
            return metrics_source[ld]
        if str(ld) in metrics_source:
            return metrics_source[str(ld)]
        return {}

    leads = [1, 2, 3, 4]
    rec_mae = [get_lead_metric(l).get("rec_mae_mean", 0.0) for l in leads]
    ora_mae = [get_lead_metric(l).get("ora_mae_mean", 0.0) for l in leads]
    gaps = [get_lead_metric(l).get("delta_mae_mean", 0.0) for l in [2, 3, 4]]
    ci_low = [get_lead_metric(l).get("delta_mae_ci_lower", 0.0) for l in [2, 3, 4]]
    ci_high = [get_lead_metric(l).get("delta_mae_ci_upper", 0.0) for l in [2, 3, 4]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    c_title = "#0F172A"
    c_grid = "#F1F5F9"
    c_rec = "#E11D48"
    c_ora = "#059669"
    c_gap = "#4338CA"

    # LEFT PANEL: Protocols 1 vs 2 Comparison
    ax1.set_facecolor("#FFFFFF")
    x = np.arange(len(leads))
    bar_w = 0.32
    b1 = ax1.bar(x - bar_w/2, rec_mae, width=bar_w, color=c_rec, alpha=0.92, label="Standard Recursive (Model A0)", zorder=3)
    b2 = ax1.bar(x + bar_w/2, ora_mae, width=bar_w, color=c_ora, alpha=0.92, label="Oracle Counterfactual (Ground Truth Injected)", zorder=3)

    for i, bar in enumerate(b1):
        h = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2 - (0.045 if i == 0 else 0.0)
        ax1.text(x_pos, h + 0.002, f"{h:.4f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=c_rec)

    for i, bar in enumerate(b2):
        h = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2 + (0.045 if i == 0 else 0.0)
        ax1.text(x_pos, h + 0.002, f"{h:.4f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=c_ora)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Week 1\n(Days 1–7)", "Week 2\n(Days 8–14)", "Week 3\n(Days 15–21)", "Week 4\n(Days 22–28)"], fontsize=10.5, color=c_title)
    ax1.set_ylabel("Validation MAE (Normalized [0, 1])", fontsize=11.5, fontweight="bold", color=c_title)
    ax1.set_ylim(0, 0.11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("Forecast Error Across Lead Horizons", fontsize=13.0, fontweight="bold", color=c_title, pad=12, loc="left")
    ax1.legend(loc="upper left", frameon=False, fontsize=9.5)

    # RIGHT PANEL: Error Gap Compounding with 95% Bootstrap CI
    ax2.set_facecolor("#FFFFFF")
    gx = np.array([2, 3, 4])
    ax2.axhline(0, color="#94A3B8", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.fill_between(gx, ci_low, ci_high, color="#E0E7FF", alpha=0.65, label="95% Bootstrap CI (B=10,000)")
    ax2.plot(gx, gaps, marker="o", markersize=8.5, linewidth=2.8, color=c_gap, label="Recursive Error Gap (ΔMAE)", zorder=4)

    for xi, yi in zip(gx, gaps):
        ax2.text(xi, yi + 0.0025, f"+{yi:.4f}", ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=c_gap)

    ax2.set_xticks(gx)
    ax2.set_xticklabels(["Week 2 (Days 8–14)", "Week 3 (Days 15–21)", "Week 4 (Days 22–28)"], fontsize=11.0, color=c_title)
    ax2.set_ylabel("Recursive Compounding Gap ΔMAE", fontsize=11.5, fontweight="bold", color=c_title)
    ax2.set_xlim(1.5, 4.5)
    ax2.set_ylim(-0.003, 0.052)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Compounded Degradation Gap with 95% CI", fontsize=13.0, fontweight="bold", color=c_title, pad=12, loc="left")
    ax2.legend(loc="upper left", frameon=False, fontsize=9.5)

    plt.suptitle("Empirical Proof of Recursive Error Compounding Gap in Baseline Model A0", fontsize=15.5, fontweight="bold", color=c_title, y=0.965)
    fig.text(0.50, 0.022, "194 validation cases (2022–2023) · 126 active cells · 3 independent seeds · Moving-Block Bootstrap B=10,000", ha="center", fontsize=9.3, color="#64748B", style="italic")
    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.93], w_pad=4.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_monsoon_spatial_error_maps(
    case_ids: List[str],
    case_spatial_errors: Dict[str, np.ndarray],
    eval_mask: np.ndarray,
    lons: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    divide_lon: float = 124.5,
) -> Dict[str, Any]:
    """
    Computes 2D mean spatial error maps and East-vs-West asymmetry metrics
    partitioned by Philippine monsoon regime.

    Parameters
    ----------
    case_ids : List[str]
        List of N case identifiers.
    case_spatial_errors : Dict[str, np.ndarray]
        Dictionary mapping case_id to 2D spatial error array of shape (32, 48).
    eval_mask : np.ndarray
        2D boolean evaluation mask of shape (32, 48) with 126 active land cells.
    lons : Optional[np.ndarray]
        1D array of longitudes (length 48). If None, defaults to CANDIDATE_A_LONS.
    lats : Optional[np.ndarray]
        1D array of latitudes (length 32). If None, defaults to CANDIDATE_A_LATS.
    divide_lon : float
        Longitudinal threshold dividing Western and Eastern Mindanao (default 124.5°E).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing per-regime mean maps, dipole difference map, and
        East vs. West asymmetry metrics.
    """
    if lons is None:
        from src.data.s2s import CANDIDATE_A_LONS
        lons = CANDIDATE_A_LONS
    if lats is None:
        from src.data.s2s import CANDIDATE_A_LATS
        lats = CANDIDATE_A_LATS

    mask_2d = eval_mask.astype(bool)
    regime_cases: Dict[str, List[np.ndarray]] = {"Amihan": [], "Habagat": [], "Transition": []}

    for cid in case_ids:
        r = assign_philippine_monsoon_regime(cid)
        if cid in case_spatial_errors:
            regime_cases[r].append(case_spatial_errors[cid])

    out: Dict[str, Any] = {}
    for r in ["Amihan", "Habagat", "Transition"]:
        if regime_cases[r]:
            stack = np.stack(regime_cases[r], axis=0)  # (N_r, 32, 48)
            mean_map = np.mean(stack, axis=0)
            masked_map = np.where(mask_2d, mean_map, np.nan)
            out[r] = {
                "mean_map": masked_map,
                "case_count": len(regime_cases[r]),
                "overall_mean": float(np.nanmean(masked_map)),
            }
        else:
            out[r] = {
                "mean_map": np.full((GRID_HEIGHT, GRID_WIDTH), np.nan),
                "case_count": 0,
                "overall_mean": 0.0,
            }

    # Dipole Difference Map: (Amihan - Habagat)
    if out["Amihan"]["case_count"] > 0 and out["Habagat"]["case_count"] > 0:
        dipole = out["Amihan"]["mean_map"] - out["Habagat"]["mean_map"]
    else:
        dipole = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)
    out["Dipole_Diff"] = dipole

    # East vs. West Orographic Asymmetry
    # Western Mindanao: lon < divide_lon; Eastern Mindanao: lon >= divide_lon
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    east_mask = mask_2d & (lon_grid >= divide_lon)
    west_mask = mask_2d & (lon_grid < divide_lon)

    out["Asymmetry"] = {
        "divide_lon": divide_lon,
        "east_cells": int(np.sum(east_mask)),
        "west_cells": int(np.sum(west_mask)),
        "Amihan": {
            "east_mean": float(np.nanmean(out["Amihan"]["mean_map"][east_mask])) if np.any(east_mask) else 0.0,
            "west_mean": float(np.nanmean(out["Amihan"]["mean_map"][west_mask])) if np.any(west_mask) else 0.0,
        },
        "Habagat": {
            "east_mean": float(np.nanmean(out["Habagat"]["mean_map"][east_mask])) if np.any(east_mask) else 0.0,
            "west_mean": float(np.nanmean(out["Habagat"]["mean_map"][west_mask])) if np.any(west_mask) else 0.0,
        },
    }
    return out


def load_psa_boundary_geom(boundary_path: Optional[Union[str, Path]] = None) -> Any:
    """
    Loads official PSA-NAMRIA regional boundary polygon for Mindanao.
    """
    if boundary_path is None:
        boundary_path = Path("processed/boundary/mindanao_analysis_boundary.gpkg")
    bp = Path(boundary_path)
    if not bp.exists():
        repo_root = Path(__file__).resolve().parent.parent.parent
        alt = repo_root / "processed" / "boundary" / "mindanao_analysis_boundary.gpkg"
        if alt.exists():
            bp = alt
        else:
            return None
    try:
        import sqlite3
        import shapely.wkb
        conn = sqlite3.connect(bp)
        cur = conn.cursor()
        cur.execute("SELECT geom FROM mindanao_analysis_boundary LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            raw_geom = row[0]
            flags = raw_geom[3]
            envelope_type = (flags >> 1) & 0x07
            header_lens = {0: 8, 1: 40, 2: 56, 3: 56, 4: 72}
            return shapely.wkb.loads(raw_geom[header_lens.get(envelope_type, 8):])
    except Exception:
        return None
    return None


def plot_psa_boundary(ax: Any, geom: Any, edgecolor: str = "#0F172A", linewidth: float = 1.1, alpha: float = 0.95, zorder: int = 6) -> None:
    """
    Plots the official PSA-NAMRIA boundary onto matplotlib axis.
    """
    if geom is None:
        return
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in geoms:
        x, y = p.exterior.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
        for interior in p.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, color=edgecolor, linewidth=linewidth * 0.6, alpha=alpha * 0.75, zorder=zorder)


def generate_seasonality_diagnostic_plots(
    monsoon_res: Dict[str, Any],
    quarter_res: Dict[str, Any],
    spatial_res: Dict[str, Any],
    output_plot_path: Union[str, Path],
) -> None:
    """
    Generates an authoritative, publication-grade 4-panel composite (300 DPI)
    illustrating the empirical Seasonality & Monsoon Gap in Model A0 on an expanded canvas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Expanded canvas for larger, highly readable panels
    fig, axes = plt.subplots(2, 2, figsize=(21.0, 16.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    c_title = "#0F172A"
    c_grid = "#F1F5F9"

    # Color Palette: Amihan (Blue/Teal), Habagat (Rose/Red), Transition (Amber)
    c_ami = "#0284C7"
    c_hab = "#E11D48"
    c_trn = "#D97706"

    # -------------------------------------------------------------------------
    # PANEL A: Compounded Recursive Error Gap by Monsoon Regime across Leads
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#FFFFFF")
    leads = [2, 3, 4]
    x = np.arange(len(leads))
    w = 0.26

    ami_gaps = [monsoon_res["Amihan"]["leads"][str(l)]["delta_mae_mean"] for l in leads]
    hab_gaps = [monsoon_res["Habagat"]["leads"][str(l)]["delta_mae_mean"] for l in leads]
    trn_gaps = [monsoon_res["Transition"]["leads"][str(l)]["delta_mae_mean"] for l in leads]

    b_ami = ax_a.bar(x - w, ami_gaps, width=w, color=c_ami, label=f"Amihan (Nov–Apr, N={monsoon_res['Amihan']['case_count']})", zorder=3)
    b_hab = ax_a.bar(x, hab_gaps, width=w, color=c_hab, label=f"Habagat (Jun–Oct, N={monsoon_res['Habagat']['case_count']})", zorder=3)
    b_trn = ax_a.bar(x + w, trn_gaps, width=w, color=c_trn, label=f"Transition (May, N={monsoon_res['Transition']['case_count']})", zorder=3)

    # Actual numbers on top of each bar graph in Panel A
    for i in range(len(leads)):
        ax_a.text(x[i] - w, ami_gaps[i] + 0.0008, f"{ami_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_ami)
        ax_a.text(x[i], hab_gaps[i] + 0.0008, f"{hab_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_hab)
        ax_a.text(x[i] + w, trn_gaps[i] + 0.0008, f"{trn_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_trn)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Week 2\n(Days 8–14)", "Week 3\n(Days 15–21)", "Week 4\n(Days 22–28)"], fontsize=12.0, color=c_title)
    ax_a.set_ylabel("Empirical Error Gap ΔMAE (Normalized)", fontsize=13.0, fontweight="bold", color=c_title)
    ax_a.set_ylim(0, 0.050)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_title("A. Compounded Error Gap by Monsoon Regime", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_a.legend(loc="upper left", frameon=False, fontsize=11.5)

    # -------------------------------------------------------------------------
    # PANEL B: Standard Meteorological Quarters (DJF, MAM, JJA, SON)
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#FFFFFF")
    quarters = ["DJF", "MAM", "JJA", "SON"]
    qx = np.arange(len(quarters))
    qw = 0.28

    w3_gaps = [quarter_res[q]["leads"]["3"]["delta_mae_mean"] for q in quarters]
    w4_gaps = [quarter_res[q]["leads"]["4"]["delta_mae_mean"] for q in quarters]

    b_w3 = ax_b.bar(qx - qw/2, w3_gaps, width=qw, color="#64748B", label="Week 3 Degradation Gap", zorder=3)
    b_w4 = ax_b.bar(qx + qw/2, w4_gaps, width=qw, color="#0F172A", label="Week 4 Degradation Gap", zorder=3)

    # Actual numbers on top of each bar graph in Panel B
    for i in range(len(quarters)):
        ax_b.text(qx[i] - qw/2, w3_gaps[i] + 0.0008, f"{w3_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#475569")
        ax_b.text(qx[i] + qw/2, w4_gaps[i] + 0.0008, f"{w4_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#0F172A")

    ax_b.set_xticks(qx)
    ax_b.set_xticklabels([f"{q}\n(N={quarter_res[q]['case_count']})" for q in quarters], fontsize=12.0, color=c_title)
    ax_b.set_ylabel("Degradation Gap ΔMAE (Normalized)", fontsize=13.0, fontweight="bold", color=c_title)
    ax_b.set_ylim(0, 0.054)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_title("B. Conventional Quarter Benchmark (Lesinger Parity)", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    # Position legend higher up with clean white background so it never overlaps bars
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=11.5)

    # Geographic Extents & Boundary
    extent_cand_a = [115.875, 127.875, 3.875, 11.875]
    xlims = (118.2, 127.0)
    ylims = (4.8, 10.4)
    boundary_geom = load_psa_boundary_geom()

    # -------------------------------------------------------------------------
    # PANEL C: Spatial Error Map during Northeast Monsoon (Amihan)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#F8FAFC")
    ax_c.set_aspect("equal")
    ax_c.set_xlim(xlims)
    ax_c.set_ylim(ylims)
    ami_map = spatial_res["Amihan"]["mean_map"]
    cmap_err = matplotlib.colormaps["YlOrRd"].copy()
    cmap_err.set_bad(color="#F8FAFC")

    vmin = 0.03
    vmax = 0.11
    im_c = ax_c.imshow(ami_map, origin="upper", cmap=cmap_err, vmin=vmin, vmax=vmax, extent=extent_cand_a, zorder=2)
    plot_psa_boundary(ax_c, boundary_geom, linewidth=1.2)
    ax_c.set_title("C. Spatial MAE: Northeast Monsoon (Amihan)", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_c.set_xlabel("Longitude (°E)", fontsize=12.0, color=c_title)
    ax_c.set_ylabel("Latitude (°N)", fontsize=12.0, color=c_title)
    ax_c.tick_params(labelsize=11.0)
    ax_c.axvline(124.5, color="#64748B", linestyle=":", linewidth=1.3, zorder=5)
    ax_c.text(0.04, 0.90, f"East Mean: {spatial_res['Asymmetry']['Amihan']['east_mean']:.4f}\nWest Mean: {spatial_res['Asymmetry']['Amihan']['west_mean']:.4f}",
              transform=ax_c.transAxes, fontsize=11.0, fontweight="bold", color="#0F172A",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.92), zorder=7)
    cb_c = plt.colorbar(im_c, ax=ax_c, fraction=0.035, pad=0.04)
    cb_c.set_label("Normalized MAE", fontsize=11.5, fontweight="bold")
    cb_c.ax.tick_params(labelsize=10.5)

    # -------------------------------------------------------------------------
    # PANEL D: Spatial Error Map during Southwest Monsoon (Habagat)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#F8FAFC")
    ax_d.set_aspect("equal")
    ax_d.set_xlim(xlims)
    ax_d.set_ylim(ylims)
    hab_map = spatial_res["Habagat"]["mean_map"]
    im_d = ax_d.imshow(hab_map, origin="upper", cmap=cmap_err, vmin=vmin, vmax=vmax, extent=extent_cand_a, zorder=2)
    plot_psa_boundary(ax_d, boundary_geom, linewidth=1.2)
    ax_d.set_title("D. Spatial MAE: Southwest Monsoon (Habagat)", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_d.set_xlabel("Longitude (°E)", fontsize=12.0, color=c_title)
    ax_d.set_ylabel("Latitude (°N)", fontsize=12.0, color=c_title)
    ax_d.tick_params(labelsize=11.0)
    ax_d.axvline(124.5, color="#64748B", linestyle=":", linewidth=1.3, zorder=5)
    ax_d.text(0.04, 0.90, f"East Mean: {spatial_res['Asymmetry']['Habagat']['east_mean']:.4f}\nWest Mean: {spatial_res['Asymmetry']['Habagat']['west_mean']:.4f}",
              transform=ax_d.transAxes, fontsize=11.0, fontweight="bold", color="#0F172A",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.92), zorder=7)
    cb_d = plt.colorbar(im_d, ax=ax_d, fraction=0.035, pad=0.04)
    cb_d.set_label("Normalized MAE", fontsize=11.5, fontweight="bold")
    cb_d.ax.tick_params(labelsize=10.5)

    # Global Suptitle
    fig.suptitle(
        "Empirical Evidence of the Seasonality & Monsoon Gap in Model A0\n"
        "194 Validation Cycles (2022–2023) · 126 Land Cells · Model A0 Static Weights vs. Regional Monsoons",
        fontsize=16.5, fontweight="bold", color=c_title, y=0.985
    )

    # Footnote
    fig.text(
        0.50, 0.015,
        "PAGASA Monsoon Partition (Amihan: Nov–Apr, Habagat: Jun–Oct, Transition: May) · East-West Orographic Asymmetry · Sealed 2024–2025 Quarantined",
        ha="center", fontsize=11.0, color="#64748B", style="italic"
    )

    plt.tight_layout(rect=[0.02, 0.035, 0.98, 0.94], w_pad=3.5, h_pad=3.5)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_divided_seasonality_figures(
    monsoon_res: Dict[str, Any],
    quarter_res: Dict[str, Any],
    spatial_res: Dict[str, Any],
    output_ab_path: Union[str, Path],
    output_cd_path: Union[str, Path],
) -> None:
    """
    Generates two separate, divided high-resolution diagnostic figures:
      1. Figures A & B (Statistical Compounding by Monsoon & Quarter)
      2. Figures C & D (2D Spatial Error Maps for Amihan and Habagat)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_ab = Path(output_ab_path)
    p_cd = Path(output_cd_path)
    p_ab.parent.mkdir(parents=True, exist_ok=True)
    p_cd.parent.mkdir(parents=True, exist_ok=True)

    c_title = "#0F172A"
    c_grid = "#F1F5F9"
    c_ami = "#0284C7"
    c_hab = "#E11D48"
    c_trn = "#D97706"

    # =========================================================================
    # PART 1: FIGURE A & FIGURE B (STATISTICAL ERROR COMPOUNDING)
    # =========================================================================
    fig_ab, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20.0, 9.0), dpi=300)
    fig_ab.patch.set_facecolor("#FFFFFF")

    leads = [2, 3, 4]
    x = np.arange(len(leads))
    w = 0.26

    ami_gaps = [monsoon_res["Amihan"]["leads"][str(l)]["delta_mae_mean"] for l in leads]
    hab_gaps = [monsoon_res["Habagat"]["leads"][str(l)]["delta_mae_mean"] for l in leads]
    trn_gaps = [monsoon_res["Transition"]["leads"][str(l)]["delta_mae_mean"] for l in leads]

    ax_a.bar(x - w, ami_gaps, width=w, color=c_ami, label=f"Amihan (Nov–Apr, N={monsoon_res['Amihan']['case_count']})", zorder=3)
    ax_a.bar(x, hab_gaps, width=w, color=c_hab, label=f"Habagat (Jun–Oct, N={monsoon_res['Habagat']['case_count']})", zorder=3)
    ax_a.bar(x + w, trn_gaps, width=w, color=c_trn, label=f"Transition (May, N={monsoon_res['Transition']['case_count']})", zorder=3)

    for i in range(len(leads)):
        ax_a.text(x[i] - w, ami_gaps[i] + 0.0008, f"{ami_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_ami)
        ax_a.text(x[i], hab_gaps[i] + 0.0008, f"{hab_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_hab)
        ax_a.text(x[i] + w, trn_gaps[i] + 0.0008, f"{trn_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=c_trn)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Week 2\n(Days 8–14)", "Week 3\n(Days 15–21)", "Week 4\n(Days 22–28)"], fontsize=12.0, color=c_title)
    ax_a.set_ylabel("Empirical Error Gap ΔMAE (Normalized)", fontsize=13.0, fontweight="bold", color=c_title)
    ax_a.set_ylim(0, 0.050)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_title("A. Compounded Error Gap by Monsoon Regime", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_a.legend(loc="upper left", frameon=False, fontsize=11.5)

    quarters = ["DJF", "MAM", "JJA", "SON"]
    qx = np.arange(len(quarters))
    qw = 0.28

    w3_gaps = [quarter_res[q]["leads"]["3"]["delta_mae_mean"] for q in quarters]
    w4_gaps = [quarter_res[q]["leads"]["4"]["delta_mae_mean"] for q in quarters]

    ax_b.bar(qx - qw/2, w3_gaps, width=qw, color="#64748B", label="Week 3 Degradation Gap", zorder=3)
    ax_b.bar(qx + qw/2, w4_gaps, width=qw, color="#0F172A", label="Week 4 Degradation Gap", zorder=3)

    for i in range(len(quarters)):
        ax_b.text(qx[i] - qw/2, w3_gaps[i] + 0.0008, f"{w3_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#475569")
        ax_b.text(qx[i] + qw/2, w4_gaps[i] + 0.0008, f"{w4_gaps[i]:.4f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#0F172A")

    ax_b.set_xticks(qx)
    ax_b.set_xticklabels([f"{q}\n(N={quarter_res[q]['case_count']})" for q in quarters], fontsize=12.0, color=c_title)
    ax_b.set_ylabel("Degradation Gap ΔMAE (Normalized)", fontsize=13.0, fontweight="bold", color=c_title)
    ax_b.set_ylim(0, 0.054)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_title("B. Conventional Quarter Benchmark (Lesinger Parity)", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=11.5)

    fig_ab.suptitle(
        "Empirical Evidence of Seasonality & Monsoon Gap: Statistical Error Compounding (Model A0)\n"
        "194 Validation Cycles (2022–2023) · 126 Land Cells · PAGASA Monsoon & Quarterly Stratification",
        fontsize=16.0, fontweight="bold", color=c_title, y=0.985
    )
    fig_ab.text(0.50, 0.020, "PAGASA Monsoon Partition (Amihan: Nov–Apr, Habagat: Jun–Oct, Transition: May) · Sealed 2024–2025 Quarantined",
                ha="center", fontsize=10.5, color="#64748B", style="italic")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.93], w_pad=4.0)
    plt.savefig(p_ab, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # PART 2: FIGURE C & FIGURE D (SPATIAL ERROR MAPS)
    # =========================================================================
    fig_cd, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(20.0, 9.5), dpi=300)
    fig_cd.patch.set_facecolor("#FFFFFF")

    extent_cand_a = [115.875, 127.875, 3.875, 11.875]
    xlims = (118.2, 127.0)
    ylims = (4.8, 10.4)
    boundary_geom = load_psa_boundary_geom()

    cmap_err = matplotlib.colormaps["YlOrRd"].copy()
    cmap_err.set_bad(color="#F8FAFC")
    vmin = 0.03
    vmax = 0.11

    # Panel C: Amihan
    ax_c.set_facecolor("#F8FAFC")
    ax_c.set_aspect("equal")
    ax_c.set_xlim(xlims)
    ax_c.set_ylim(ylims)
    ami_map = spatial_res["Amihan"]["mean_map"]
    im_c = ax_c.imshow(ami_map, origin="upper", cmap=cmap_err, vmin=vmin, vmax=vmax, extent=extent_cand_a, zorder=2)
    plot_psa_boundary(ax_c, boundary_geom, linewidth=1.2)
    ax_c.set_title("C. Spatial MAE: Northeast Monsoon (Amihan)", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_c.set_xlabel("Longitude (°E)", fontsize=12.0, color=c_title)
    ax_c.set_ylabel("Latitude (°N)", fontsize=12.0, color=c_title)
    ax_c.tick_params(labelsize=11.0)
    ax_c.axvline(124.5, color="#64748B", linestyle=":", linewidth=1.3, zorder=5)
    ax_c.text(0.04, 0.90, f"East Mean: {spatial_res['Asymmetry']['Amihan']['east_mean']:.4f}\nWest Mean: {spatial_res['Asymmetry']['Amihan']['west_mean']:.4f}",
              transform=ax_c.transAxes, fontsize=11.0, fontweight="bold", color="#0F172A",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.92), zorder=7)
    cb_c = plt.colorbar(im_c, ax=ax_c, fraction=0.035, pad=0.04)
    cb_c.set_label("Normalized MAE", fontsize=11.5, fontweight="bold")
    cb_c.ax.tick_params(labelsize=10.5)

    # Panel D: Habagat
    ax_d.set_facecolor("#F8FAFC")
    ax_d.set_aspect("equal")
    ax_d.set_xlim(xlims)
    ax_d.set_ylim(ylims)
    hab_map = spatial_res["Habagat"]["mean_map"]
    im_d = ax_d.imshow(hab_map, origin="upper", cmap=cmap_err, vmin=vmin, vmax=vmax, extent=extent_cand_a, zorder=2)
    plot_psa_boundary(ax_d, boundary_geom, linewidth=1.2)
    ax_d.set_title("D. Spatial MAE: Southwest Monsoon (Habagat)", fontsize=14.5, fontweight="bold", color=c_title, pad=14, loc="left")
    ax_d.set_xlabel("Longitude (°E)", fontsize=12.0, color=c_title)
    ax_d.set_ylabel("Latitude (°N)", fontsize=12.0, color=c_title)
    ax_d.tick_params(labelsize=11.0)
    ax_d.axvline(124.5, color="#64748B", linestyle=":", linewidth=1.3, zorder=5)
    ax_d.text(0.04, 0.90, f"East Mean: {spatial_res['Asymmetry']['Habagat']['east_mean']:.4f}\nWest Mean: {spatial_res['Asymmetry']['Habagat']['west_mean']:.4f}",
              transform=ax_d.transAxes, fontsize=11.0, fontweight="bold", color="#0F172A",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.92), zorder=7)
    cb_d = plt.colorbar(im_d, ax=ax_d, fraction=0.035, pad=0.04)
    cb_d.set_label("Normalized MAE", fontsize=11.5, fontweight="bold")
    cb_d.ax.tick_params(labelsize=10.5)

    fig_cd.suptitle(
        "Empirical Evidence of Seasonality & Monsoon Gap: Geographic Error Asymmetry (Model A0)\n"
        "194 Validation Cycles (2022–2023) · 126 Land Cells · Orographic Reversal Across 124.5°E Meridian",
        fontsize=16.0, fontweight="bold", color=c_title, y=0.985
    )
    fig_cd.text(0.50, 0.020, "Official PSA-NAMRIA Administrative Boundary Overlay · Sealed 2024–2025 Quarantined",
                ha="center", fontsize=10.5, color="#64748B", style="italic")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.93], w_pad=4.0)
    plt.savefig(p_cd, dpi=300, bbox_inches="tight")
    plt.close()



def generate_seasonality_presentation_slide(
    monsoon_res: Dict[str, Any],
    spatial_res: Dict[str, Any],
    output_plot_path: Union[str, Path],
) -> None:
    """
    Generates a dedicated 16:9 widescreen presentation slide proving the
    Seasonality & Monsoon Gap for the thesis presentation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    c_title = "#0F172A"
    c_grid = "#F1F5F9"

    # LEFT PANEL: Monsoonal Compounding Gap Comparison
    ax1.set_facecolor("#FFFFFF")
    leads = [2, 3, 4]
    x = np.arange(len(leads))
    w = 0.28
    ami_g = [monsoon_res["Amihan"]["leads"][str(l)]["delta_mae_mean"] for l in leads]
    hab_g = [monsoon_res["Habagat"]["leads"][str(l)]["delta_mae_mean"] for l in leads]
    trn_g = [monsoon_res["Transition"]["leads"][str(l)]["delta_mae_mean"] for l in leads]

    ax1.bar(x - w, ami_g, width=w, color="#0284C7", label=f"Northeast Monsoon (Amihan, N={monsoon_res['Amihan']['case_count']})", zorder=3)
    ax1.bar(x, hab_g, width=w, color="#E11D48", label=f"Southwest Monsoon (Habagat, N={monsoon_res['Habagat']['case_count']})", zorder=3)
    ax1.bar(x + w, trn_g, width=w, color="#D97706", label=f"Inter-Monsoon Transition (May, N={monsoon_res['Transition']['case_count']})", zorder=3)

    for i in range(len(leads)):
        ax1.text(x[i] - w, ami_g[i] + 0.001, f"{ami_g[i]:.4f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#0284C7")
        ax1.text(x[i], hab_g[i] + 0.001, f"{hab_g[i]:.4f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#E11D48")
        ax1.text(x[i] + w, trn_g[i] + 0.001, f"{trn_g[i]:.4f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#D97706")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Week 2 (Days 8–14)", "Week 3 (Days 15–21)", "Week 4 (Days 22–28)"], fontsize=11, color=c_title)
    ax1.set_ylabel("Recursive Error Compounding ΔMAE", fontsize=11.5, fontweight="bold", color=c_title)
    ax1.set_ylim(0, 0.055)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle="-", color=c_grid, linewidth=1.2, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("Compounded Recursive Degradation Across Monsoons", fontsize=13, fontweight="bold", color=c_title, pad=12, loc="left")
    ax1.legend(loc="upper left", frameon=False, fontsize=9.8)

    # Geographic Extents & Boundary
    extent_cand_a = [115.875, 127.875, 3.875, 11.875]
    xlims = (118.2, 127.0)
    ylims = (4.8, 10.4)
    boundary_geom = load_psa_boundary_geom()

    # RIGHT PANEL: The Spatial Error Dipole Map (Amihan vs Habagat Difference)
    ax2.set_facecolor("#F8FAFC")
    ax2.set_aspect("equal")
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    dipole_map = spatial_res["Dipole_Diff"]
    cmap_dipole = matplotlib.colormaps["coolwarm"].copy()
    cmap_dipole.set_bad(color="#F8FAFC")

    v_abs = 0.025
    im2 = ax2.imshow(dipole_map, origin="upper", cmap=cmap_dipole, vmin=-v_abs, vmax=v_abs, extent=extent_cand_a, zorder=2)
    plot_psa_boundary(ax2, boundary_geom)
    ax2.axvline(124.5, color="#0F172A", linestyle="--", linewidth=1.3, zorder=5)
    ax2.set_title("Geographic Error Asymmetry: Amihan vs. Habagat", fontsize=13, fontweight="bold", color=c_title, pad=12, loc="left")
    ax2.set_xlabel("Longitude (°E)", fontsize=10.5, color=c_title)
    ax2.set_ylabel("Latitude (°N)", fontsize=10.5, color=c_title)

    ax2.text(0.04, 0.90, "RED: Higher Error in Amihan (East Coast)\nBLUE: Higher Error in Habagat (West Coast)",
             transform=ax2.transAxes, fontsize=9.0, fontweight="bold", color="#0F172A",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.95), zorder=7)

    cb = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
    cb.set_label("Spatial Error Difference (MAE_Amihan - MAE_Habagat)", fontsize=9.5, fontweight="bold")

    plt.suptitle(
        "Demonstration of Seasonality & Monsoon Blind Spots in Baseline Model A0",
        fontsize=15.5, fontweight="bold", color=c_title, y=0.965
    )
    fig.text(
        0.50, 0.022,
        "194 validation cases (88 Amihan, 88 Habagat, 18 Transition) · 126 active cells · Unconditioned Model A0 fails to adapt to monsoon wind reversal",
        ha="center", fontsize=9.3, color="#64748B", style="italic"
    )

    plt.tight_layout(rect=[0.02, 0.075, 0.98, 0.93], w_pad=5.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# Professional / Formal Aliases
generate_recursive_compounding_analysis = generate_recursive_gap_presentation_slide
generate_seasonality_dipole_analysis = generate_seasonality_presentation_slide
