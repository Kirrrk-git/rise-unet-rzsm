"""
src/evaluation/__init__.py
--------------------------
Evaluation package for Mindanao RISE-UNet.
Provides authoritative diagnostic engines and statistical testing procedures.
"""

from src.evaluation.recursive_diagnostic import (
    assemble_oracle_priors,
    apply_boundary_clipped_perturbation,
    compute_case_level_metrics,
    compute_paired_differences,
    moving_block_bootstrap,
    wilcoxon_paired_test,
    holm_bonferroni_correction,
    compute_cohens_d_z,
    evaluate_gate2_rules,
)
from src.evaluation.v3_metrics_and_bootstrap import (
    compute_empirical_crps_draws,
    compute_predictive_mean_deterministic_metrics,
    compute_spread_and_pi_coverage,
    compute_dry_event_brier_and_reliability,
    calendar_block_paired_bootstrap,
    evaluate_primary_w3_w4_family,
    compute_risk_decile_validity,
    compute_perturbation_m_slope,
)

__all__ = [
    "assemble_oracle_priors",
    "apply_boundary_clipped_perturbation",
    "compute_case_level_metrics",
    "compute_paired_differences",
    "moving_block_bootstrap",
    "wilcoxon_paired_test",
    "holm_bonferroni_correction",
    "compute_cohens_d_z",
    "evaluate_gate2_rules",
    "compute_empirical_crps_draws",
    "compute_predictive_mean_deterministic_metrics",
    "compute_spread_and_pi_coverage",
    "compute_dry_event_brier_and_reliability",
    "calendar_block_paired_bootstrap",
    "evaluate_primary_w3_w4_family",
    "compute_risk_decile_validity",
    "compute_perturbation_m_slope",
]
