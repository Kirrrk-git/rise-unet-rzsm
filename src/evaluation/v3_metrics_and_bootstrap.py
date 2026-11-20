"""
src/evaluation/v3_metrics_and_bootstrap.py
------------------------------------------
Thesis v3.0 evaluation, dry-event Brier/reliability, RiskNet diagnostic, and
calendar-block paired Moving-Block Bootstrap engine (pure NumPy / SciPy / Pandas).

Implements Thesis Sections 2.5 - 2.6:
  - Per-case predictive mean (over M = 11 x D = 110 draws) MAE, RMSE, spatial ACC
  - Full 110-draw empirical CRPS (Eq. 15) + population spread + 80%/95% PI coverage
  - Dry-event probability p_hat = N_dry / M using STRICT inequality (< tau)
  - 10-bin reliability diagram ([0.0, 0.1), ..., [0.9, 1.0] inclusive) & binary Brier Score
  - 28-calendar-day (and 56-day sensitivity) circular calendar-block paired MBB
    with 10,000 replicates, percentile 95% CI, mean-centered null p, and Holm-Bonferroni
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.recursive_diagnostic import holm_bonferroni_correction


def compute_empirical_crps_draws(
    draws: np.ndarray,
    y_true: np.ndarray,
    eval_mask: np.ndarray,
) -> float:
    """
    Computes domain-averaged exact empirical CRPS (Thesis Eq. 15) over active cells:
        CRPS = (1/M) sum_i |x_i - y| - (1 / (2 M^2)) sum_i sum_j |x_i - x_j|

    Parameters
    ----------
    draws : np.ndarray
        Ensemble draws shaped (M, H, W) or (M, H, W, 1).
    y_true : np.ndarray
        Ground truth shaped (H, W) or (1, H, W, 1).
    eval_mask : np.ndarray
        Boolean or binary (H, W) mask with 126 active land cells.
    """
    m_bool = np.asarray(eval_mask) == 1
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    y = np.asarray(y_true, dtype=np.float64).squeeze()

    x_act = x[:, m_bool]  # (M, 126)
    y_act = y[m_bool]     # (126,)
    m_count = x_act.shape[0]

    term1 = np.mean(np.abs(x_act - y_act[np.newaxis, :]), axis=0)
    # Exact algebraic O(M log M) equivalence to (1 / (2 M^2)) sum_{i,j} |x_i - x_j|:
    x_sorted = np.sort(x_act, axis=0)
    weights = (2.0 * np.arange(m_count, dtype=np.float64) - m_count + 1.0)[:, np.newaxis]
    term2 = np.sum(weights * x_sorted, axis=0) / float(m_count * m_count)
    return float(np.mean(term1 - term2))


def compute_predictive_mean_deterministic_metrics(
    draws: np.ndarray,
    y_true: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    """Computes MAE, RMSE, and spatial Pearson ACC from the predictive mean over draws."""
    m_bool = np.asarray(eval_mask) == 1
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    y = np.asarray(y_true, dtype=np.float64).squeeze()

    pred_mean = np.mean(x, axis=0)[m_bool]
    obs = y[m_bool]
    diff = pred_mean - obs
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    p_c = pred_mean - np.mean(pred_mean)
    o_c = obs - np.mean(obs)
    denom = float(np.sqrt(np.sum(p_c ** 2) * np.sum(o_c ** 2)))
    if denom <= 1e-12:
        return {"mae": mae, "rmse": rmse, "acc": np.nan, "zero_variance_acc": True}
    acc = float(np.sum(p_c * o_c) / denom)
    return {"mae": mae, "rmse": rmse, "acc": acc, "zero_variance_acc": False}


def compute_spread_and_pi_coverage(
    draws: np.ndarray,
    y_true: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, float]:
    """Computes population spread (ddof=0) and 80% / 95% empirical interval coverage & width."""
    m_bool = np.asarray(eval_mask) == 1
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    y = np.asarray(y_true, dtype=np.float64).squeeze()

    x_act = x[:, m_bool]
    y_act = y[m_bool]
    spread = float(np.mean(np.std(x_act, axis=0, ddof=0)))

    q025, q10, q90, q975 = np.percentile(x_act, [2.5, 10.0, 90.0, 97.5], axis=0)
    cov80 = float(np.mean((y_act >= q10) & (y_act <= q90)))
    cov95 = float(np.mean((y_act >= q025) & (y_act <= q975)))
    width80 = float(np.mean(q90 - q10))
    width95 = float(np.mean(q975 - q025))
    return {
        "spread": spread,
        "coverage_80": cov80,
        "coverage_95": cov95,
        "width_80": width80,
        "width_95": width95,
    }


def compute_dry_event_brier_and_reliability(
    draws: np.ndarray,
    y_true: np.ndarray,
    tau_season: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates binary drought event prediction using STRICT inequality (< tau(s, k)):
      - Forecast probability p_hat = N_dry / M over M=110 draws
      - Observed indicator o = 1{y < tau(s, k)}
      - Active-cell binary Brier score and 10 reliability bin counts
    """
    m_bool = np.asarray(eval_mask) == 1
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    y = np.asarray(y_true, dtype=np.float64).squeeze()
    tau = np.asarray(tau_season, dtype=np.float64).squeeze()

    x_act = x[:, m_bool]
    y_act = y[m_bool]
    tau_act = tau[m_bool]

    p_hat = np.mean(x_act < tau_act[np.newaxis, :], axis=0)
    obs_dry = (y_act < tau_act).astype(np.float64)
    brier = float(np.mean((p_hat - obs_dry) ** 2))

    # 10 reliability bins of width 0.1 with [0.9, 1.0] inclusive at upper boundary
    bin_idx = np.minimum((p_hat * 10.0).astype(int), 9)
    bin_counts = np.bincount(bin_idx, minlength=10)
    bin_sum_p = np.bincount(bin_idx, weights=p_hat, minlength=10)
    bin_sum_o = np.bincount(bin_idx, weights=obs_dry, minlength=10)

    return {
        "brier_score": brier,
        "event_rate": float(np.mean(obs_dry)),
        "mean_forecast_prob": float(np.mean(p_hat)),
        "bin_counts": bin_counts.tolist(),
        "bin_sum_prob": bin_sum_p.tolist(),
        "bin_sum_obs": bin_sum_o.tolist(),
    }


def calendar_block_paired_bootstrap(
    issue_dates: Sequence[str],
    diff_series: Sequence[float],
    block_days: int = 28,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Calendar-day circular Moving-Block Bootstrap for irregularly spaced forecast issues.
    Preserves real calendar gaps by indexing issues onto the calendar day timeline and
    sampling contiguous `block_days` windows (mean-centered null hypothesis test + 95% CI).
    """
    if len(issue_dates) != len(diff_series):
        raise ValueError("issue_dates and diff_series must have equal length.")
    if len(diff_series) == 0:
        raise ValueError("diff_series must not be empty.")

    dates = pd.to_datetime(issue_dates)
    diffs = np.asarray(diff_series, dtype=np.float64)
    order = np.argsort(dates.values)
    dates = dates[order]
    diffs = diffs[order]

    t0_ref = dates[0]
    day_offsets = (dates - t0_ref).days.to_numpy()
    span_days = int(day_offsets[-1]) + 1
    n_obs = len(diffs)
    observed_mean = float(np.mean(diffs))
    null_diffs = diffs - observed_mean

    # Map each day offset mod span_days -> indices of observations on that day
    day_to_obs: Dict[int, np.ndarray] = {}
    for d in np.unique(day_offsets):
        day_to_obs[int(d)] = np.where(day_offsets == d)[0]

    # Pre-collect indices falling inside every circular block starting on day s in [0, span_days)
    block_members = []
    for start_d in range(span_days):
        idxs = []
        for off in range(block_days):
            d_mod = (start_d + off) % span_days
            if d_mod in day_to_obs:
                idxs.extend(day_to_obs[d_mod].tolist())
        block_members.append(np.array(idxs, dtype=np.int64))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(span_days / float(block_days)))
    boot_means = np.empty(n_boot, dtype=np.float64)
    boot_null_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        starts = rng.integers(0, span_days, size=n_blocks)
        pooled = np.concatenate([block_members[s] for s in starts])
        while pooled.size < n_obs:
            extra_s = int(rng.integers(0, span_days))
            if block_members[extra_s].size > 0:
                pooled = np.concatenate([pooled, block_members[extra_s]])
        sample_idx = pooled[:n_obs]
        boot_means[b] = float(np.mean(diffs[sample_idx]))
        boot_null_means[b] = float(np.mean(null_diffs[sample_idx]))

    ci_low, ci_high = np.percentile(boot_means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    # Standard discrete two-sided null p-value: (1 + count(|null| >= |obs|)) / (B + 1)
    p_val = float((1 + np.sum(np.abs(boot_null_means) >= abs(observed_mean))) / (n_boot + 1))

    return {
        "observed_mean_diff": observed_mean,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "p_boot_two_sided": p_val,
        "block_days": int(block_days),
        "n_boot": int(n_boot),
        "n_cases": int(n_obs),
        "calendar_span_days": int(span_days),
    }


def evaluate_primary_w3_w4_family(
    issue_dates: Sequence[str],
    combined_minus_baseline_crps: Dict[int, Sequence[float]],
    block_days: int = 28,
    n_boot: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs primary W3 & W4 CRPS comparison with Holm-Bonferroni family correction."""
    if 3 not in combined_minus_baseline_crps or 4 not in combined_minus_baseline_crps:
        raise KeyError("Primary family requires lead keys 3 (W3) and 4 (W4).")
    res_w3 = calendar_block_paired_bootstrap(issue_dates, combined_minus_baseline_crps[3], block_days, n_boot, seed)
    res_w4 = calendar_block_paired_bootstrap(issue_dates, combined_minus_baseline_crps[4], block_days, n_boot, seed + 1)
    hb = holm_bonferroni_correction({3: res_w3["p_boot_two_sided"], 4: res_w4["p_boot_two_sided"]})
    return {"W3": {**res_w3, **hb[3]}, "W4": {**res_w4, **hb[4]}}


def compute_risk_decile_validity(
    q_hat_active: np.ndarray,
    raw_abs_err_active: np.ndarray,
    refined_abs_err_active: np.ndarray,
    val_decile_edges: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Computes Spearman rank correlation between predicted RiskNet q_hat and realized error,
    plus bin-wise realized raw/refined errors using fixed validation-derived bin edges.
    """
    q = np.asarray(q_hat_active, dtype=np.float64).ravel()
    raw = np.asarray(raw_abs_err_active, dtype=np.float64).ravel()
    ref = np.asarray(refined_abs_err_active, dtype=np.float64).ravel()

    rho_res = spearmanr(q, raw)
    if val_decile_edges is None:
        edges = np.quantile(q, np.linspace(0.0, 1.0, 11))
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        edges = np.asarray(val_decile_edges, dtype=np.float64)

    bin_ids = np.clip(np.digitize(q, edges[1:-1], right=False), 0, 9)
    bins = []
    for b in range(10):
        mask = bin_ids == b
        bins.append({
            "decile": b + 1,
            "count": int(np.sum(mask)),
            "mean_q_hat": float(np.mean(q[mask])) if np.any(mask) else np.nan,
            "mean_raw_err": float(np.mean(raw[mask])) if np.any(mask) else np.nan,
            "mean_delta_err": float(np.mean(ref[mask] - raw[mask])) if np.any(mask) else np.nan,
        })

    return {
        "spearman_rho": float(rho_res.statistic),
        "spearman_p": float(rho_res.pvalue),
        "bin_edges": edges.tolist(),
        "deciles": bins,
    }


def compute_perturbation_m_slope(
    metric_by_m: Dict[float, float],
    fit_m_ge_1_only: bool = True,
) -> float:
    """Computes linear slope of downstream CRPS/MAE across sensitivity multipliers m."""
    ms = sorted(float(m) for m in metric_by_m if (not fit_m_ge_1_only or float(m) >= 1.0))
    if len(ms) < 2:
        raise ValueError("Need at least two multiplier points to compute slope.")
    vals = np.array([metric_by_m[m] for m in ms], dtype=np.float64)
    slope, _ = np.polyfit(np.array(ms, dtype=np.float64), vals, deg=1)
    return float(slope)
