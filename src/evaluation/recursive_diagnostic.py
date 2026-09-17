"""
src/evaluation/recursive_diagnostic.py
-------------------------------------
Authoritative execution engine and statistical testing framework for
Phase 23 Recursive Degradation Diagnostic (Mindanao RISE-UNet Adaptation).

Conforms strictly to contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml:
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
    contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml.

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

    out_path = Path(output_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    leads = [1, 2, 3, 4]
    lead_labels = ["W1 (Days 1-7)", "W2 (Days 8-14)", "W3 (Days 15-21)", "W4 (Days 22-28)"]

    # Support either 'lead_summary' (from notebook) or 'lead_metrics' (from script)
    metrics_source = results.get("lead_summary", results.get("lead_metrics", {}))

    # Helper to look up lead in metrics dictionary (int or str key)
    def get_lead_metric(ld: int) -> Dict[str, Any]:
        if ld in metrics_source:
            return metrics_source[ld]
        if str(ld) in metrics_source:
            return metrics_source[str(ld)]
        return {}

    # Panel A: Recursive vs Oracle Error
    ax_a = axes[0, 0]
    rec_mae = [get_lead_metric(l).get("rec_mae_mean", 0.0) for l in leads]
    ora_mae = [get_lead_metric(l).get("ora_mae_mean", 0.0) for l in leads]
    x_pos = np.arange(len(leads))
    width = 0.35
    ax_a.bar(x_pos - width / 2, rec_mae, width, label="Protocol 1: Standard Recursive", color="#d95f02", alpha=0.9)
    ax_a.bar(x_pos + width / 2, ora_mae, width, label="Protocol 2: Oracle Counterfactual", color="#1b9e77", alpha=0.9)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(lead_labels)
    ax_a.set_ylabel("Validation MAE (126 cells)")
    ax_a.set_title("(A) Standard Recursive vs. Oracle Counterfactual Forecast Error", fontweight="bold", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="upper left")

    # Panel B: Recursive Degradation Gap Delta E_k with MBB 95% CIs
    ax_b = axes[0, 1]
    gaps = [get_lead_metric(l).get("delta_mae_mean", 0.0) for l in [2, 3, 4]]
    ci_los = [get_lead_metric(l).get("delta_mae_ci_lower", 0.0) for l in [2, 3, 4]]
    ci_his = [get_lead_metric(l).get("delta_mae_ci_upper", 0.0) for l in [2, 3, 4]]
    yerr = [np.maximum(0.0, np.array(gaps) - np.array(ci_los)), np.maximum(0.0, np.array(ci_his) - np.array(gaps))]
    ax_b.errorbar([2, 3, 4], gaps, yerr=yerr, fmt="-o", color="#7570b3", ecolor="#7570b3", capsize=6, linewidth=2, markersize=8)
    ax_b.axhline(0, color="black", linestyle=":", alpha=0.7)
    ax_b.set_xticks([2, 3, 4])
    ax_b.set_xticklabels(["W2", "W3", "W4"])
    ax_b.set_ylabel("Recursive Degradation Gap ΔMAE")
    ax_b.set_title("(B) Primary Recursive Error Gap (ΔE_k = E_rec - E_ora) with 95% MBB CI", fontweight="bold", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Protocol 4 Perturbation Divergence D_k(eps)
    ax_c = axes[1, 0]
    default_perts = [-0.50, -0.25, -0.10, -0.05, 0.05, 0.10, 0.25, 0.50]
    eps_vals = results.get("perturbations_evaluated", default_perts)
    div_dict = results.get("divergence_by_lead_eps", {})
    for lead in [2, 3, 4]:
        ld_div = div_dict.get(lead, div_dict.get(str(lead), {}))
        div_vals = [ld_div.get(e, ld_div.get(str(e), 0.0)) for e in eps_vals]
        ax_c.plot(eps_vals, div_vals, marker="o", label=f"Lead {lead}")
    ax_c.set_xlabel("Perturbation Offset ε applied to W1")
    ax_c.set_ylabel("Downstream Divergence D_k(ε) [RMSE]")
    ax_c.set_title("(C) Downstream Error-Injection Divergence (Protocol 4)", fontweight="bold", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend()

    # Panel D: Perturbation Boundary Clipping Percentage
    ax_d = axes[1, 1]
    clip_dict = results.get("clipping_census", {})
    clip_pcts = [clip_dict.get(e, clip_dict.get(str(e), 0.0)) for e in eps_vals]
    ax_d.bar([f"{e:+0.2f}" for e in eps_vals], clip_pcts, color="#e7298a", alpha=0.85)
    ax_d.set_xlabel("Perturbation Offset ε")
    ax_d.set_ylabel("Active Cells Clipped to [0, 1] (%)")
    ax_d.set_title("(D) Boundary Saturation Census (Active Land Cells Clipped)", fontweight="bold", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Phase 23 Diagnostic: Subseasonal Recursive Error Compounding Analysis (Mindanao A0)", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

