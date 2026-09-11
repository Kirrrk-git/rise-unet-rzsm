"""
Data processing modules for RISE-UNet Mindanao adaptation.
"""
from src.data.rzsm import (
    compute_depth_weighted_rzsm,
    compute_backward_rolling_mean,
    extract_antecedent_lags,
    apply_land_mask,
    compute_min_max_scale,
    remap_era5_land_to_candidate_a,
    LAYER_WEIGHTS,
)
from src.data.temporal import (
    TemporalConfig,
    compute_trailing_rolling_mean,
    compute_training_climatology,
    compute_seasonal_anomalies,
    extract_antecedent_and_target_windows,
    fit_min_max_bounds,
    standardize_with_training_bounds,
)
from src.data.compile_cube import (
    ProductionCubeConfig,
    compile_production_rzsm_pipeline,
    verify_production_cube_census,
    FastLandAwareRemapper,
    process_era5_land_monthly_pair,
    process_era5_land_antecedent_file,
    compile_full_11yr_rzsm_cube,
)

__all__ = [
    "compute_depth_weighted_rzsm",
    "compute_backward_rolling_mean",
    "extract_antecedent_lags",
    "apply_land_mask",
    "compute_min_max_scale",
    "remap_era5_land_to_candidate_a",
    "LAYER_WEIGHTS",
    "TemporalConfig",
    "compute_trailing_rolling_mean",
    "compute_training_climatology",
    "compute_seasonal_anomalies",
    "extract_antecedent_and_target_windows",
    "fit_min_max_bounds",
    "standardize_with_training_bounds",
    "ProductionCubeConfig",
    "compile_production_rzsm_pipeline",
    "verify_production_cube_census",
    "FastLandAwareRemapper",
    "process_era5_land_monthly_pair",
    "process_era5_land_antecedent_file",
    "compile_full_11yr_rzsm_cube",
]

