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
]
