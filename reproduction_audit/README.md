<!-- markdownlint-disable -->
# Scientific Reproduction Audits & Verification Dossiers

This directory contains the formal, three-tier scientific verification dossiers for the **Mindanao Tropical RISE-UNet Adaptation (Track B)** and its parent paper replication (**Track A**, Lesinger & Tian 2025, *Nature Communications*).

Every milestone, numerical check, spatial contract, and physical pipeline implementation in this project is backed by an independent, reproducible audit dossier in this folder.

---

## 1. The Three-Tier Scientific Certification Standard

To prevent conflating software execution with published methodological fidelity, all assertions throughout these dossiers are classified under an explicit three-tier standard:

1. **`[PASS]` (Software & Numerical Integrity)**:
   Automated unit tests, execution assertions, and numerical scripts ran cleanly with zero failures, zero errors, finite outputs, and machine-precision arithmetic compliance. Proves the code works as implemented.
2. **`[VERIFIED]` (Parent Source-Code Reconciliation)**:
   Direct empirical reconciliation against Kyle Lesinger's authoritative parent EX29 codebase. Proves that an algorithmic or mathematical specification reproduces the exact logic of the peer-reviewed parent baseline.
3. **`[ACCEPTED]` (Methodological Regional Adaptation)**:
   Deliberate, documented scientific and engineering adaptations adopted specifically for the Mindanao regional setting (tropical archipelago climatology, ERA5-Land RZSM ground-truth definition, ECMWF CY48R1 operational calendar) where the regional implementation intentionally diverges from or extends the CONUS baseline.

---

## 2. Dossier Directory & Thematic Index

For clarity and ease of navigation for external researchers and reviewers, the audit dossiers are organized into 6 thematic domains:

### Theme 1: Authoritative Parent Baseline Replication (Track A)
- **[`01_parent_baseline_replication_audit.md`](01_parent_baseline_replication_audit.md)**: Independent replication and execution trace of Kyle Lesinger's original CONUS UNet baseline.
- **[`01a_parent_baseline_execution_log.md`](01a_parent_baseline_execution_log.md)**: Raw execution logs, environment dependencies, and convergence curves for the parent model.
- **[`02_parent_experiment_trace_audit.md`](02_parent_experiment_trace_audit.md)**: Bit-for-bit verification of Experiment 29 (EX29) configuration, loss functions, and weights.
- **[`02a_parent_experiment_trace_log.md`](02a_parent_experiment_trace_log.md)**: Trace logs of recursive cascade execution across Weeks 1 to 4 leads.
- **[`03_parent_variable_contract_audit.md`](03_parent_variable_contract_audit.md)**: Detailed audit of the 11 input predictor variables and ground-truth target specifications.
- **[`04_parent_temporal_target_parity_audit.md`](04_parent_temporal_target_parity_audit.md)**: Verification of the rolling 7-day target window and antecedent lag temporal parity.

---

### Theme 2: Mindanao Spatial Foundation & Domain Geometry (Track B)
- **[`05_mindanao_geometry_envelopes_audit.md`](05_mindanao_geometry_envelopes_audit.md)**: Comparative evaluation of Candidate bounding boxes (Candidate A $32 \times 48$ selected).
- **[`06_mindanao_boundary_and_gadm_audit.md`](06_mindanao_boundary_and_gadm_audit.md)**: Geospatial audit of the official GADM Level 0/1 administrative boundaries for Mindanao.
- **[`07_mindanao_candidate_a_grid_freeze_audit.md`](07_mindanao_candidate_a_grid_freeze_audit.md)**: Mathematical specification and coordinate freeze of the Candidate A regular $0.25^\circ$ grid.
- **[`08_mindanao_evaluation_mask_audit.md`](08_mindanao_evaluation_mask_audit.md)**: Census and topology audit of the 126 active land evaluation cells and ocean buffer cells.
- **[`09_mindanao_spatial_foundation_dossier.md`](09_mindanao_spatial_foundation_dossier.md)**: Unified spatial contract certification reconciling GDAL, rasterio, and PyGMT rasterization.

---

### Theme 3: ERA5-Land Soil Moisture Target Datacube
- **[`10_era5_land_archive_completeness_audit.md`](10_era5_land_archive_completeness_audit.md)**: Census audit of 265 monthly ERA5-Land NetCDFs covering 11 years (2015–2025).
- **[`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md)**: Numerical verification of depth-weighted integration of soil moisture Layers 1, 2, 3 ($0\text{--}100\,\text{cm}$).
- **[`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md)**: Scientific justification for adopting ERA5-Land RZSM as verified ground truth in tropical cloud-dense domains.
- **[`13_production_cube_preflight_audit.md`](13_production_cube_preflight_audit.md)**: Preflight census certifying 96,912 hourly timesteps across 4,038 continuous days with zero missing values.
- **[`14_production_cube_compilation_and_census_audit.md`](14_production_cube_compilation_and_census_audit.md)**: Final compilation audit of the continuous 11-year NetCDF target cube (`processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc`).

---

### Theme 4: Atmospheric & S2S Dynamic Predictor Pipelines
- **[`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md)**: Verification of surface atmospheric predictor derivation ($T_{2m}, T_{max}, D_{2m}, P, Z_{200}$).
- **[`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md)**: Pure-Python GRIB2 Section 7 decoding and bilinear remapping of ECMWF CY48R1 subseasonal reforecasts.

---

### Theme 5: Operational Forecast Calendar, Splits, and Normalization
- **[`17_pilot_case_ladder_and_manifest_audit.md`](17_pilot_case_ladder_and_manifest_audit.md)**: Verification of pilot case ladder and temporal antecedent lag availability.
- **[`18_production_case_calendar_1154_cycles_audit.md`](18_production_case_calendar_1154_cycles_audit.md)**: Authoritative reconciliation of the 1,154-cycle ECMWF CY48R1 forecast-origin calendar (2015–2025).
- **[`19_dataset_splits_and_normalization_contract_audit.md`](19_dataset_splits_and_normalization_contract_audit.md)**: Rigorous audit of Train (735), Validation (210), and Sealed Test (209) splits, target-horizon boundary extension semantics, and active-domain scalar min/max normalization parameters.

---

### Theme 6: Training Pipeline, Hardware Benchmarks, and Checkpoint Persistence
- **[`20_tf_dataset_pipeline_and_checkpoint_audit.md`](20_tf_dataset_pipeline_and_checkpoint_audit.md)**: Verification of ensemble batch constraints ($B \pmod{11} == 0$), target broadcasting, and bit-for-bit checkpoint restore.
- **[`21_vram_and_hardware_profiling_audit.md`](21_vram_and_hardware_profiling_audit.md)**: Comprehensive GPU VRAM profiling and throughput telemetry across candidate batch sizes on genuine Model A0.
- **[`22_a0_tiny_overfit_gradient_audit.md`](22_a0_tiny_overfit_gradient_audit.md)**: Optimization convergence proof confirming multi-head CRPS gradient propagation to zero loss on single batch.
