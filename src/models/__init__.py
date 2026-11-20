"""
src/models/__init__.py
----------------------
Model factory and architectural specifications for the Mindanao RISE-UNet project.
"""

from .a0_unet import (
    build_a0_unet,
    LEAD_CHANNELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    TOTAL_A0_PARAMETERS,
)
from .scpw_mrgr import (
    V3_LEAD_CHANNELS,
    V3_NONREC_CHANNELS,
    EXPECTED_EXTRACONV_BACKBONE_PARAMS,
    EXPECTED_SCPW_PARAMS,
    MRGR_DIRECTED_HANDOFFS,
    EXPECTED_MRGR_BRANCH_PARAMS,
    compute_week_midpoint_date,
    compute_scpw_calendar_descriptor,
    compute_v3_parameter_budget,
    compute_mrgr_scale_and_target,
    compute_huber_risk_loss,
    apply_mrgr_gate_and_refinement,
    build_extraconv_unet_v3,
    build_mrgr_risknet,
    build_mrgr_correctionnet,
)

__all__ = [
    "build_a0_unet",
    "LEAD_CHANNELS",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "TOTAL_A0_PARAMETERS",
    "V3_LEAD_CHANNELS",
    "V3_NONREC_CHANNELS",
    "EXPECTED_EXTRACONV_BACKBONE_PARAMS",
    "EXPECTED_SCPW_PARAMS",
    "MRGR_DIRECTED_HANDOFFS",
    "EXPECTED_MRGR_BRANCH_PARAMS",
    "compute_week_midpoint_date",
    "compute_scpw_calendar_descriptor",
    "compute_v3_parameter_budget",
    "compute_mrgr_scale_and_target",
    "compute_huber_risk_loss",
    "apply_mrgr_gate_and_refinement",
    "build_extraconv_unet_v3",
    "build_mrgr_risknet",
    "build_mrgr_correctionnet",
]
