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
from src.data.s2s import (
    unpack_grib2_section7_simple,
    parse_ecmwf_s2s_grib_messages,
    remap_s2s_grid_to_candidate_a,
    harmonize_s2s_cycle,
    CANDIDATE_A_LATS,
    CANDIDATE_A_LONS,
    TRIPLET_PARAMS,
)
from src.data.case_builder import (
    CaseTensorHierarchy,
    assemble_single_a0_case,
    simulate_recursive_cascade_step,
)
from src.data.tf_dataset import (
    A0CaseBatchGenerator,
    create_a0_tf_dataset,
    crps2d_numpy,
    crps_exact_analytical,
    save_a0_checkpoint,
    restore_a0_checkpoint,
    validate_batch_size,
    prepare_case_lead_tensors,
    load_case_npz,
    ENSEMBLE_MEMBERS,
    LEAD_CHANNELS,
    OUTPUT_HEADS,
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
    "unpack_grib2_section7_simple",
    "parse_ecmwf_s2s_grib_messages",
    "remap_s2s_grid_to_candidate_a",
    "harmonize_s2s_cycle",
    "CANDIDATE_A_LATS",
    "CANDIDATE_A_LONS",
    "TRIPLET_PARAMS",
    "CaseTensorHierarchy",
    "assemble_single_a0_case",
    "simulate_recursive_cascade_step",
    "A0CaseBatchGenerator",
    "create_a0_tf_dataset",
    "crps2d_numpy",
    "crps_exact_analytical",
    "save_a0_checkpoint",
    "restore_a0_checkpoint",
    "validate_batch_size",
    "prepare_case_lead_tensors",
    "load_case_npz",
    "ENSEMBLE_MEMBERS",
    "LEAD_CHANNELS",
    "OUTPUT_HEADS",
]




