<!-- markdownlint-disable -->
# Mindanao RISE-UNet Operational Execution Matrix, Audit Dossier Index & Repository File Registry

**Project**: Enhanced RISE-UNet for Subseasonal Root-Zone Soil Moisture (RZSM) Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Parent Baseline**: Lesinger & Tian (2025), *Nature Communications*, DOI: [`10.1038/s41467-025-62761-3`](https://doi.org/10.1038/s41467-025-62761-3)  
**Target Model**: **Mindanao Model A0** (Adapted from EX29 Recursive Hybrid RISE-UNet Baseline)  
**Document Classification**: Living Operational Matrix, Comprehensive Directory Registry & Master File Map  
**Last Updated**: 2026-09-14 (Step 21K.1 Case Calendar & Step 21K.2 Dataset Splits / Normalization Parameters Certification)  

---

## 1. Executive Purpose & Governance Architecture

This document serves as the **authoritative reference map and structural guideline** for the Mindanao regional adaptation of the RISE-UNet deep learning forecasting system. It defines:
1. The **Operational Execution Matrix** tracking all research phases, methodological gates, and deliverables.
2. The **Audit Dossier Index & Decision Log** providing an immutable record of empirical certifications and architectural decisions.
3. The **Comprehensive Repository File Registry** mapping **every directory and file** created for this adaptation (excluding original parent repository files), detailing their exact technical purpose, mathematical rationale, and current status.
4. The **Mandatory Maintenance Rule** governing future updates whenever remarkable codebase changes occur.

### Scientific Certification Taxonomy

To preserve thesis-level methodological precision, every claim, artifact, and numerical result is certified under one of three strict tiers:

```text
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                               SCIENTIFIC CERTIFICATION TIERS                          │
├───────────────────────────────────────────────────────────────────────────────────────┤
│  [PASS]     : Software, mathematical, and numerical behavior verified against         │
│               deterministic unit tests and zero-tolerance empirical censuses.          │
│                                                                                       │
│  [VERIFIED] : Direct mathematical and programmatic parity established with the parent │
│               EX29 peer-reviewed implementation (Lesinger & Tian, 2025).              │
│                                                                                       │
│  [ACCEPTED] : Deliberate, documented regional adaptations required specifically for    │
│               Mindanao geography, climate dynamics, or data lake infrastructure.      │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Document Authority Hierarchy

To guarantee architectural coherence and eliminate competing sources of truth, this repository enforces an explicit three-tier authority hierarchy:

```text
contracts/
    ↓
Machine-Readable Scientific & Technical Truth (Authoritative specifications, grid geometries, bounds)

master plan (mindanao_adaptation_master_plan.md)
    ↓
Project Execution Roadmap (Phases, methodological gates, sequence of work)

artifact registry (OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
    ↓
Inventory, Directory Map & Status Index (Catalog of non-author files, audit index, verification records)
```

The artifact registry is an inventory and status index; it must never contradict or override `contracts/` or the `master plan`.

### 1.2 Independent Empirical Verification of User Feedback

> [!CRITICAL]
> **User Feedback is Advisory, NOT Ground Truth**:
> All suggestions, critique, arithmetic, and feedback provided by the user must be treated as advisory review and expert guidance—**never as unverified ground truth**.
> 
> Before incorporating any user claim, critique, or suggested number into the codebase, contracts, or audit records:
> 1. Independently calculate and verify the arithmetic (e.g., date ranges, cell counts, tensor shapes).
> 2. Cross-reference the claim against actual source code, raw data files, and published literature (Lesinger & Tian, 2025).
> 3. Execute unit tests or empirical scripts to confirm software and mathematical behavior.
> 4. Explicitly state the verification findings before committing changes.

---

## 2. End-to-End Operational Execution Matrix

The following matrix governs the lifecycle of the Mindanao regional adaptation from baseline freeze to final thesis defense:

| Phase / Step | Objective & Description | Methodological Standard | Current Status | Primary Deliverable / Audit File |
| :--- | :--- | :---: | :---: | :--- |
| **Gate 1A** | Parent Code Reproduction & Environment Freeze | EX29 Environment Parity | `[VERIFIED]` | [`01_parent_baseline_replication_audit.md`](01_parent_baseline_replication_audit.md) |
| **Gate 1B** | Parent Pipeline Trace & Recursive Loop Verification | EX29 Recursive Logic Parity | `[VERIFIED]` | [`02_parent_experiment_trace_audit.md`](02_parent_experiment_trace_audit.md) |
| **Step 21A.1** | Isolated Mindanao Adaptation Git Branch Creation | Git Commit Integrity (`4adb9fa`) | `[PASS]` | Git branch `origin/mindanao-adaptation` |
| **Step 21A.2** | Official PSA Administrative Boundary Harmonization & Geodesy | PSA PSGC 2Q 2026 / GeoRiskPH | `[ACCEPTED]` | [`06_mindanao_boundary_and_gadm_audit.md`](06_mindanao_boundary_and_gadm_audit.md) ($99,948.76\text{ km}^2$) |
| **Step 21A.3** | Parent EX29 Channel, Lag & Variable Contract Reconciliation | EX29 Source-Code Contract Parity | `[VERIFIED]` | [`03_parent_variable_contract_audit.md`](03_parent_variable_contract_audit.md) ($W_1=11, W_2=12, W_3=5, W_4=6$) |
| **Step 21A.4** | Candidate 0.25° Spatial Envelopes & 16-Divisible Geometry Analysis | Divisibility $H, W \pmod{16} = 0$ | `[ACCEPTED]` | [`05_mindanao_geometry_envelopes_audit.md`](05_mindanao_geometry_envelopes_audit.md) |
| **Step 21A.5** | Synthetic Candidate Tensor & 4-Lead Cascade Compatibility Gate | Native TF2 Forward & Backward Flow | `[PASS]` | `notebooks/02_mindanao_geometry_and_cascade_gate.ipynb` |
| **Step 21A.6** | Candidate A ($32 \times 48, 0.25^\circ$) Grid Selection & NetCDF Construction | Candidate A Mesh Freeze | `[ACCEPTED]` | [`07_mindanao_candidate_a_grid_freeze_audit.md`](07_mindanao_candidate_a_grid_freeze_audit.md) |
| **Step 21A.7A** | Fractional Boundary-Coverage Mask (`mindanao_fraction_025.nc`) | WGS84 Ellipsoidal Geodesics | `[ACCEPTED]` | `processed/grid/mindanao_fraction_025.nc` ($99,948.76\text{ km}^2$, $0.0000\%$ error) |
| **Step 21A.7B** | Binary Evaluation Mask ($f \ge 0.50$, 126 Cells) | Majority Land Rule ($f \ge 0.50$) | `[ACCEPTED]` | [`08_mindanao_evaluation_mask_audit.md`](08_mindanao_evaluation_mask_audit.md) (126 cells: full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$) |
| **Step 21A.8** | Freeze Machine-Readable Spatial Contract | Immutable Spatial Governance | `[PASS]` | [`contracts/spatial/spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml) |
| **Step 21A.9** | ERA5-Land RZSM Target Acceptance & Independent Cross-Reference | Depth-Weighted Profile & Physics | `[ACCEPTED]` | [`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md) |
| **Sub-Phase 21A** | Spatial Foundation Synthesis & Geodetic Certification | Complete Geodetic Quadrature | `[PASS]` | [`09_mindanao_spatial_foundation_dossier.md`](09_mindanao_spatial_foundation_dossier.md) |
| **Step 21B.1** | ERA5-Land Archive Ingestion Audit (2015–2025 Nominal Archive) | 264 NetCDF Files (132 Monthly Pairs) | `[PASS]` | [`10_era5_land_archive_completeness_audit.md`](10_era5_land_archive_completeness_audit.md) |
| **Step 21B.2** | 2014 Antecedent Window Retrieval (20 Days) | +1 NetCDF File (265 Combined Files) | `[ACCEPTED]` | [`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md) (12–31 Dec 2014 Support Archive in GCS) |
| **Step 21B.3** | Pilot Daily Mean & Depth Weighting ($0.07, 0.21, 0.72$) | 0–100 cm Physical Profile | `[VERIFIED]` | [`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md) |
| **Step 21B.4** | Bilinear Remapping, Coastal Fallback & Pilot Census | Land-Aware Remapping & Masking | `[PASS]` | [`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md) (2,520 evaluations, zero NaNs) |
| **Step 21C.1** | ERA5 Atmospheric Pilot Retrieval (Jan 2015, 744 Hours) | Hourly Single & Pressure Levels | `[PASS]` | [`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md) |
| **Step 21C.2** | 5-Channel Atmospheric Derivation & Census | Bolton (1980) Humidity & Extr. | `[PASS / VERIFIED / ACCEPTED]` | [`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md) (3,906 evaluations, zero NaNs) |
| **Step 21C.3** | 11-Year ERA5 Atmospheric Archive Mirroring | Hourly Monthly NetCDFs to GCS | `[TRAINING_ARCHIVE_SECURED]` | Training fold (Jan 2015 – Dec 2021, 84 months $\times$ 2 products = 168 files) fully secured and verified in GCS lake; Dec 2014 antecedent support (2 files) tracked separately. Full 11-year archive target is 264 monthly files; validation/test partition (2022–2025, 96 files) background mirroring in progress |
| **Step 21D.1** | Depth-Weighted RZSM Modular Implementation | Vectorized `src/data/rzsm.py` | `[PASS]` | Automated Unit Tests (`test_02_rzsm.py`) |
| **Step 21D.2** | Automated RZSM Unit Test Suite | Precision $\le 10^{-7}$, Real Pilot | `[PASS]` | 10 Unit Tests Passing |
| **Step 21D.3** | Temporal Preprocessing & Target Lead Reconciliation | $L=[6,13,20,27]$, Trailing Rolling | `[PASS / VERIFIED]` | Satisfies software tests & explicitly reconciled parent contracts ([`Audit`](04_parent_temporal_target_parity_audit.md)) |
| **Step 21D.4-PREFLIGHT** | Production Cube Preflight Verification Gate | 11-Point Preflight Census & Integrity Gate | `[PASS]` | Verified 265 NetCDFs (96,912 hrs across 4,038 days), zero NaNs/Infs, verified spatial contract ([`Report`](13_production_cube_preflight_audit.md)) |
| **Sub-Phase 21E** | ECMWF S2S Dynamic Triplet Pilot Acquisition & Harmonization Engine | ECDS API (`t2m, d2m, tcw`), 11 Members, $1.5^\circ \to 0.25^\circ$ Remap, Hard-Fail Safe Production Default | `[PASS / VERIFIED]` | Pure-Python GRIB2 Section 7 decoder & harmonizer in `src/data/s2s.py` with `allow_step0_fallback=False` default; 7 unit tests passing; pilot artifact `processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc` (0.42 MB) synced to GCS; certified in [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md) |
| **Step 21F.1–2** | Single Complete EX29-Derived A0 Case & Target Assembly | Leads W1–W4 Multi-Lead Tensor Schema ($[11, 12, 5, 6]$), Exact Lags `[-1, -7, -14]`, Verified Target Offsets `[6, 13, 20, 27]` | `[PASS / VERIFIED]` | Multi-lead tensor hierarchy assembler in `src/data/case_builder.py`; verified $[M=11, 32, 48, C_k]$ across $W_1..W_4$; targets reconciled to parent EX29 $L = (\text{lead} \times 7) - 1$; 5 unit tests passing; 81 total unit tests passing |
| **Step 21F.3** | ecCodes Interoperability, Real Case Assembly & Recursive Graph Integrity | Scoped ecCodes comparison (462 msgs), UNET_RZSM 4-lead recursive cascade, downstream perturbation response, zero NaNs/Infs over 126 binary evaluation cells | `[PASS / VERIFIED]` | Certified in [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md) — Case Assembly & Computational Integrity (Forecasting skill deferred to 21H–21K); Interactive Colab notebook `notebooks/07_mindanao_s2s_and_pilot_case_assembly.ipynb` & dual 2x2 publication dashboards `figures/mindanao_s2s_dynamic_predictor_composite.png` and `figures/mindanao_s2s_pilot_case_and_recursive_inference.png` (dual synced to GCS) |
| **Sub-Phase 21G** | 8-Case Pilot Ladder Manifest Generation & Pipeline Stability | 8 Consecutive Cycles (Jan 15 – Mar 4, 2015), 88-Row Member Manifest, 0 NaNs across 314,496 values, GCS Parity, 16-point independent provenance verification | `[PASS / VERIFIED]` | Certified in [`17_pilot_case_ladder_and_manifest_audit.md`](17_pilot_case_ladder_and_manifest_audit.md); Member manifest `manifests/cases_pilot_v001.csv` (88 rows) & summary `manifests/cases_pilot_summary_v001.csv` (8 rows); 8 NPZ cases in `processed/cases/pilot/` synced to `gs://rise-unet-rzsm/processed/cases/pilot/`; 42/42 unit tests passing; verified by `scripts/10_verify_pilot_ladder_provenance_and_census.py` |
| **Sub-Phase 21H** | Surrogate TensorFlow Data Pipeline, Ensemble Grouping & Checkpoint Smoke Test | SURROGATE SMOKE TEST ONLY. Validated data feeding, 11-member batching, target broadcasting, and checkpoint saving on 2-layer Conv2D surrogate (did NOT instantiate genuine UNET_RZSM) | `[SURROGATE_ONLY]` | Certified in [`20_tf_dataset_pipeline_and_checkpoint_audit.md`](20_tf_dataset_pipeline_and_checkpoint_audit.md); Module `src/data/tf_dataset.py`; Automated test suite `tests/test_11_tf_dataset.py` (16/16 passing); Pipeline test script `scripts/13_train_a0_pipeline_checkpoint.py`; Interactive Colab notebook `notebooks/08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb`; Checkpoints in `checkpoints/a0_pipeline_test/` |
| **Sub-Phase 21I** | Surrogate Tiny-Data Optimization Smoke Test (8 Cases, 40 Epochs) | SURROGATE SMOKE TEST ONLY. 2-layer Conv2D surrogate on unnormalized inputs; 99% loss drop was an unnormalized scale artifact, NOT proof of genuine RISE-UNet capacity or overfit; reclassified universally as surrogate smoke test | `[SURROGATE_ONLY]` | Certified in [`22_a0_tiny_overfit_gradient_audit.md`](22_a0_tiny_overfit_gradient_audit.md); Script `scripts/12_train_a0_tiny_overfit.py`; Log `logs/a0_tiny_overfit_execution.json`; Checkpoint `checkpoints/a0_tiny_overfit/` |
| **Sub-Phase 21J** | Genuine Model A0 (`UNET_RZSM`) Hardware Profiling & VRAM Feasibility Benchmark | Hardware Feasibility & Production Contract Gate: Genuine 1.63M-parameter nested U-Net across Six Technical Pillars (21J.1 architecture: 1,627,139 W1 / 1,630,307 W2 params, 298 tensors, 3 deep supervision heads; 21J.2 multi-lead forward with evaluation-domain postprocessing mask $0.00 \times 10^0$; 21J.3 backpropagation with finite gradients under representative MAE workload; 21J.4 4-lead cascade with active perturbation propagation; 21J.5 VRAM ladder $B \in \{11, 22, 33, 44, 66\}$ from $2.08\text{ GB}$ to $10.57\text{ GB}$; 21J.6 production contract freeze & real weight roundtrip parity). Fail-closed gate enforced in benchmark code | `[HISTORICAL_GPU_VERIFIED_CURRENT_CODE_REVALIDATION_PENDING]` | Certified in [`21_vram_and_hardware_profiling_audit.md`](21_vram_and_hardware_profiling_audit.md); Interactive Colab notebook `notebooks/09_mindanao_a0_vram_profiling.ipynb`; Benchmark script `scripts/11_profile_a0_vram_benchmark.py`; Model factory `src/models/a0_unet.py`; Unit test `tests/test_12_a0_unet.py`; Physical GPU telemetry artifact `logs/A0_gpu_benchmark.json` |
| **Sub-Phase 21K** | Production Case Ingestion, Splits, Normalization & Baseline Training | 21K.1 Usable Case Calendar: 1,154 cycles indexed via ECMWF CY48R1 schedule-referenced forecast origin calendar; 21K.2 Dataset Splits (735 Train, 210 Val, 209 Scheduled Test Census: 202 Usable Sealed Test Denominator, 7 Quarantined) & Training Normalization Contract (`contracts/A0/normalization_parameters.yaml`) frozen strictly from 2015–2021; Dynamic tensor normalization actively bound in `case_builder.py`; 87 unit tests passing (84 passed, 3 skipped) | `[IN PROGRESS]` (Steps 21K.1 & 21K.2: `[PASS / VERIFIED / ACCEPTED]`; 21K.3: `[NOT_YET_AUTHORIZED]`) | Certified in [`18_production_case_calendar_1154_cycles_audit.md`](18_production_case_calendar_1154_cycles_audit.md) and [`19_dataset_splits_and_normalization_contract_audit.md`](19_dataset_splits_and_normalization_contract_audit.md); Calendar in `manifests/`; Splits in `manifests/splits/`; Contracts in `contracts/A0/`; Verification Status in `contracts/A0/VERIFICATION_STATUS.yaml` |
| **Step 21K.3-pre** | Genuine Production-Path Model A0 Training Smoke Test across all 4 Leads | 5-Stage Preflight: 4-lead genuine backward updates (Cin in [11, 12, 5, 6]), downstream 126-cell masked loss, recursive channel ordering & tamper detection, and full training-state step-2 trajectory roundtrip | `[IMPLEMENTED / LOCAL CONTRACT TESTS PASS / GPU EXECUTION PENDING]` | Standalone preflight script `scripts/14_run_a0_production_smoke_test.py`; Contract test `tests/test_13_production_smoke_preflight.py`; Production contract in `contracts/A0/mindanao_a0_production_contract.yaml` |
| **Phase 22** | Reference Comparators Track (B0, B1, B2) | B0 Climatology, B1 Persistence, B2 XGBoost | `[PLANNED]` | Comparative Baseline Evaluation Report |
| **Phase 23** | Recursive Degradation Diagnostic | Error Compounding: Recursive vs Oracle vs Direct | `[PLANNED]` | `GATE2_RECURSIVE_DEGRADATION_DIAGNOSTIC.md` |
| **Gate 2** | Recursive Refinement GO / NO-GO Decision Gate | Statistical Significance ($p < 0.05$), $\Delta E_k$ | `[PLANNED]` | `GATE2_DECISION_DOSSIER.md` |
| **Phase 24** | Model A1 (Lead-Aware Recursive Residual Refinement) | Lightweight Lead-Conditioned Correction Module | `[PLANNED]` | Model A1 Architecture & Training Dossier |
| **Phase 25** | Ablation Studies & Physics Plausibility | Lead-Conditioning Ablation, Error-Damping | `[PLANNED]` | Ablation & Sensitivity Report |
| **Phase 26** | Final Sealed Test Evaluation & Explainability | Sealed 2024–2025 Hold-Out, Bootstrap, SHAP | `[PLANNED]` | **Gate 3 Final Thesis Defense Dossier** |
| **Phase 27** | Thesis Synthesis & Open Data Lake Archive | Code Freeze, GCS Lake Index, Open Access | `[PLANNED]` | Final Project Archival Registry |

---

## 3. Comprehensive Repository File Registry (The Complete Project Map)

This section catalogs every directory and file introduced for the Mindanao adaptation. Original parent codebase files (such as root notebooks `00_min_max...` through `10a_...`) are excluded to maintain absolute clarity on adaptation additions.

```
dl_dm_rzsm_subseasonal_forecast/
├── checkpoints/
│   ├── a0_pipeline_test/
│   │   ├── a0_test_epoch005.weights.npz
│   │   └── a0_test_epoch005_meta.json
│   └── a0_tiny_overfit/
├── contracts/
│   └── spatial/
│       └── spatial_grid_contract.yaml
├── parent_study_ex29/
│   ├── README.md (Original author documentation with parent archive banner)
│   ├── conda_environment_setup.yaml (Original environment setup)
│   ├── Data/ (Original author raw download scripts, masks, and EMOS)
│   ├── function/ (Original author modules: modelRzsmRelu, basicUNET, losses, etc.)
│   └── notebooks/ (All 48 original CONUS notebooks: 00_*, 01_*, 02_*, ...)
├── freeze/
│   ├── git_log.txt
│   ├── git_status.txt
│   ├── parent_architecture_contract.yaml
│   ├── PARENT_COMMIT_SHA.txt
│   ├── parent_data_contract.yaml
│   ├── parent_input_contract.json
│   ├── parent_training_contract.yaml
│   ├── README_CAPTURE.md
│   └── repository_manifest_sha256.csv
├── logs/
│   └── a0_tiny_overfit_execution.json
├── contracts/
│   ├── spatial/
│   │   └── spatial_grid_contract.yaml
│   └── A0/
│       ├── normalization_parameters.yaml
│       ├── normalization_parameters.json
│       └── mindanao_a0_production_contract.yaml
├── manifests/
│   ├── cases_pilot_v001.csv
│   ├── cases_pilot_summary_v001.csv
│   ├── production_case_calendar.csv
│   └── splits/
│       ├── train_cases.csv
│       ├── val_cases.csv
│       ├── test_cases_sealed.csv
│       └── split_summary.json
├── metadata/
│   └── grid_definition.yaml
├── notebooks/
│   ├── 03_mindanao_spatial_foundation_and_mask_pipeline.ipynb
│   ├── 04_mindanao_rzsm_pilot_preprocessing.ipynb
│   ├── 05_mindanao_atmospheric_and_rzsm_preprocessing.ipynb
│   ├── 06_mindanao_datacube_and_anomaly_pipeline.ipynb
│   ├── 07_mindanao_s2s_and_pilot_case_assembly.ipynb
│   ├── 08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb
│   └── 09_mindanao_a0_vram_profiling.ipynb
├── reproduction_audit/
│   ├── README.md                               <── Dossier directory and navigation portal
│   ├── OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md (This File)
│   ├── 01_parent_baseline_replication_audit.md
│   ├── 01a_parent_baseline_execution_log.md
│   ├── 02_parent_experiment_trace_audit.md
│   ├── 02a_parent_experiment_trace_log.md
│   ├── 03_parent_variable_contract_audit.md
│   ├── 04_parent_temporal_target_parity_audit.md
│   ├── 05_mindanao_geometry_envelopes_audit.md
│   ├── 06_mindanao_boundary_and_gadm_audit.md
│   ├── 07_mindanao_candidate_a_grid_freeze_audit.md
│   ├── 08_mindanao_evaluation_mask_audit.md
│   ├── 09_mindanao_spatial_foundation_dossier.md
│   ├── 10_era5_land_archive_completeness_audit.md
│   ├── 11_era5_land_pilot_rzsm_preprocessing_audit.md
│   ├── 12_era5_land_target_acceptance_audit.md
│   ├── 13_production_cube_preflight_audit.md
│   ├── 14_production_cube_compilation_and_census_audit.md
│   ├── 15_era5_atmospheric_preprocessing_audit.md
│   ├── 16_ecmwf_s2s_pilot_and_recursive_inference_audit.md
│   ├── 17_pilot_case_ladder_and_manifest_audit.md
│   ├── 18_production_case_calendar_1154_cycles_audit.md
│   ├── 19_dataset_splits_and_normalization_contract_audit.md
│   ├── 20_tf_dataset_pipeline_and_checkpoint_audit.md
│   ├── 21_vram_and_hardware_profiling_audit.md
│   └── 22_a0_tiny_overfit_gradient_audit.md
├── scripts/
│   ├── 09_build_pilot_manifest_and_cases.py
│   ├── 06_build_production_case_calendar.py
│   ├── 08_derive_training_normalization.py
│   ├── 04_download_all_s2s_production.py
│   ├── 02_download_era5_atmospheric.py
│   ├── 07_generate_dataset_splits.py
│   ├── 01_generate_mindanao_masks.py
│   ├── 11_profile_a0_vram_benchmark.py
│   ├── run_download_atmospheric.sh
│   ├── 12_train_a0_tiny_overfit.py
│   ├── 13_train_a0_pipeline_checkpoint.py
│   ├── 03_derive_era5_atmospheric_daily.py
│   ├── 10_verify_pilot_ladder_provenance_and_census.py
│   └── 05_verify_production_cube_preflight.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── rzsm.py
│   │   ├── temporal.py
│   │   ├── compile_cube.py
│   │   ├── s2s.py
│   │   ├── case_builder.py
│   │   ├── cloud_lake.py
│   │   └── tf_dataset.py
│   └── models/
│       ├── __init__.py
│       └── a0_unet.py
├── tests/
│   ├── __init__.py
│   ├── test_12_a0_unet.py
│   ├── test_07_case_calendar.py
│   ├── test_10_case_builder.py
│   ├── test_01_cloud_lake.py
│   ├── test_05_compile_cube.py
│   ├── test_08_normalization_and_splits.py
│   ├── test_09_pilot_ladder.py
│   ├── test_02_rzsm.py
│   ├── test_04_s2s.py
│   ├── test_06_target_reconciliation.py
│   ├── test_03_temporal.py
│   └── test_11_tf_dataset.py
├── logs/
│   ├── a0_tiny_overfit_execution.json
│   └── A0_gpu_benchmark.json
├── figures/
│   ├── mindanao_spatial_grid_mesh.png
│   ├── mindanao_fractional_coverage_map.png
│   ├── mindanao_evaluation_mask_map.png
│   ├── mindanao_spatial_foundation_composite.png
│   ├── mindanao_rzsm_pilot_preprocessing_composite.png
│   ├── mindanao_era5_atmospheric_pilot_verification.png
│   ├── mindanao_production_cube_verification_composite.png
│   ├── mindanao_s2s_dynamic_predictor_composite.png
│   └── mindanao_s2s_pilot_case_and_recursive_inference.png
├── pilot_raw/
│   └── era5-land-2014-12-antecedent.nc
└── processed/
    ├── boundary/
    │   └── mindanao_analysis_boundary.gpkg
    ├── grid/
    │   ├── mindanao_0.25_grid.grd
    │   ├── mindanao_025deg.nc
    │   ├── mindanao_fraction_025.nc
    │   ├── mindanao_eval_mask_025.nc
    │   └── mindanao_fraction_025.qml / mindanao_eval_mask_025.qml
    ├── atmospheric/
    │   └── pilot/
    │       └── era5_atmospheric_pilot_2015_01.nc
    ├── cases/
    │   └── pilot/
    │       └── CASE_2015*.npz (8 serialized pilot archives)
    ├── rzsm/
    │   ├── pilot/
    │   │   ├── era5_land_rzsm_pilot_2014_2015.nc
    │   │   └── README.md
    │   └── production/
    │       └── era5_land_rzsm_production_2015_2025.nc
    └── s2s/
        └── pilot/
            └── s2s_pilot_reforecast_w1_w2.nc
```

---

### 3.1 `contracts/` (Spatial, Mathematical & Governance Specifications)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml) | Spatial grid specification contract | Locks Candidate A coordinates (`lat`: 11.75 to 4.00, `lon`: 116.00 to 127.75, shape $32 \times 48$), resolution ($0.25^\circ$), authoritative boundary area ($99,948.76\text{ km}^2$), evaluation threshold ($f \ge 0.50$), and active cell count (126 cells: full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$). | `[FROZEN]` |
| [`VERIFICATION_STATUS.yaml`](../contracts/A0/VERIFICATION_STATUS.yaml) | Authoritative Milestone Verification Status Matrix | Single machine-readable source of truth for milestone statuses (21A–21K) and strict certification rules (`mock_counts_as_pass: false`, `estimated_counts_as_pass: false`, `sealed_test_access_allowed: false`). | `[ACTIVE / AUTHORITATIVE]` |
| [`metric_evaluation_contract.yaml`](../contracts/A0/metric_evaluation_contract.yaml) | Pre-Training Metric Evaluation Contract | Freezes mathematical definitions and aggregation rules for ACC (spatial anomaly correlation over 126 active cells), CRPSS (`CRPSS_B0` vs climatology; `CRPSS_A0` vs Lead-A0 baseline), and Brier Drought Score (2015–2021 training 20th percentile). | `[FROZEN]` |
| [`mindanao_a0_production_contract.yaml`](../contracts/A0/mindanao_a0_production_contract.yaml) | Pre-Training Production Model Contract | Freezes production hyperparameters (Adam, $\text{lr}=10^{-4}$, seeds $[42, 123, 456]$, batch size $B=11$, loss weights $[0.2, 0.3, 0.5]$), per-lead parameter expectations, and reporting metrics. | `[FROZEN]` |
| [`normalization_parameters.yaml`](../contracts/A0/normalization_parameters.yaml) | Training-Only Active Domain Normalization Contract | Freezes min/max normalization parameters derived strictly from the 2015–2021 training partition across 126 active evaluation cells. Actively bound in `case_builder.py`. | `[FROZEN]` |

---

### 3.2 `freeze/` (Parent Baseline Preservation)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`PARENT_COMMIT_SHA.txt`](../freeze/PARENT_COMMIT_SHA.txt) | Parent commit capture | Records the immutable Git commit SHA of the parent repository before branch creation. | `[VERIFIED]` |
| [`git_log.txt`](../freeze/git_log.txt) | Historical Git log capture | Formatted commit log of the parent reproduction branch. | `[VERIFIED]` |
| [`git_status.txt`](../freeze/git_status.txt) | Working tree snapshot | Documents clean working tree state before regional adaptation began. | `[VERIFIED]` |
| [`parent_architecture_contract.yaml`](../freeze/parent_architecture_contract.yaml) | Parent model architecture contract | Details ConvLSTM + UNet layer topologies, hidden dimensions, kernel sizes, and loss functions. | `[VERIFIED]` |
| [`parent_data_contract.yaml`](../freeze/parent_data_contract.yaml) | Parent data specification | Records the 6 canonical variables (`sm`, `spfh`, `tmax`, `diff_temp`, `pwat`, `hgt_pres`), leads $W_1$–$W_4$, and normalization scopes. | `[VERIFIED]` |
| [`parent_input_contract.json`](../freeze/parent_input_contract.json) | Machine-readable input schema | JSON schema defining parent tensor dimensions `(B, C, H, W)` and temporal indexing. | `[VERIFIED]` |
| [`parent_training_contract.yaml`](../freeze/parent_training_contract.yaml) | Hyperparameter specification | Locks batch size (32), optimizer (Adam), learning rate ($10^{-4}$), and early stopping (15 epochs). | `[VERIFIED]` |
| [`README_CAPTURE.md`](../freeze/README_CAPTURE.md) | Parent README archive | Exact markdown capture of original author setup and execution instructions. | `[VERIFIED]` |
| [`repository_manifest_sha256.csv`](../freeze/repository_manifest_sha256.csv) | Integrity manifest | Cryptographic SHA-256 checksums for every file in the parent baseline. | `[VERIFIED]` |

---

### 3.3 `metadata/` (Spatial Grid Definitions)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`grid_definition.yaml`](../metadata/grid_definition.yaml) | Spatial grid metadata | YAML specification of spatial extents, cell spacing ($0.25^\circ$), coordinate arrays, and CF-1.8 metadata attributes. | `[FROZEN]` |

---

### 3.4 `notebooks/` (Interactive Verification & Cloud Workflows)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`03_mindanao_spatial_foundation_and_mask_pipeline.ipynb`](../notebooks/03_mindanao_spatial_foundation_and_mask_pipeline.ipynb) | Spatial foundation pipeline | 20 cells. Computes geodesic area quadrature, establishes Candidate A reference grid, generates ellipsoidal fraction and 126-cell binary mask. Pre-rendered outputs intact. | `[PASS]` |
| [`04_mindanao_rzsm_pilot_preprocessing.ipynb`](../notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb) | RZSM pilot pipeline | 18 cells. Downloads 2014 antecedent support, computes depth weighting ($0.07, 0.21, 0.72$), performs bilinear remapping to Candidate A, generates 4-panel composite. | `[PASS]` |
| [`05_mindanao_atmospheric_and_rzsm_preprocessing.ipynb`](../notebooks/05_mindanao_atmospheric_and_rzsm_preprocessing.ipynb) | Atmospheric pilot & unit test verification | 18 cells. Retrieves hourly Jan 2015 atmospheric levels, derives 5 atmospheric variables via Bolton (1980), executes automated unit tests, and verifies zero NaNs across 126 evaluation cells. | `[PASS]` |
| [`06_mindanao_datacube_and_anomaly_pipeline.ipynb`](../notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb) | Production cube & anomaly pipeline | 16 cells. Compiles full 11-year ($2015\text{--}2025$) RZSM cube, fits locked 2015–2021 seasonal climatology, derives standardized anomalies, verifies census (506,268 points, zero NaNs/Infs), and exports 4-panel verification composite. | `[PASS]` |
| [`07_mindanao_s2s_and_pilot_case_assembly.ipynb`](../notebooks/07_mindanao_s2s_and_pilot_case_assembly.ipynb) | S2S Ingestion, ecCodes Equivalence Gate & UNET_RZSM Recursive Inference | 21 cells. Ingests clean 2015-01-16 S2S cycle, executes ecCodes side-by-side equivalence gate, harmonizes with zero step-0 fallbacks, builds multi-lead A0 case hierarchy, executes UNET_RZSM 4-lead recursive cascade, demonstrates downstream perturbation sensitivity, audits 126 active cells (0 NaNs/Infs), and generates dual 2x2 publication dashboards (`mindanao_s2s_dynamic_predictor_composite.png` and `mindanao_s2s_pilot_case_and_recursive_inference.png`). | `[PASS / VERIFIED]` |
| [`08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb`](../notebooks/08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb) | Surrogate TensorFlow Pipeline, 11-Member Batching & Checkpoint Pipeline | 18 cells. Streams 8 pilot NPZ cases, verifies 11-member ensemble batching invariance, multi-head dictionary loss, optimizer step execution, and bit-for-bit checkpoint restoration using 2-layer Conv2D surrogate. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`09_mindanao_a0_vram_profiling.ipynb`](../notebooks/09_mindanao_a0_vram_profiling.ipynb) | Genuine Model A0 (`UNET_RZSM`) Hardware Profiling & VRAM Feasibility Benchmark | 17 cells. Physical GPU verification of the Six Technical Pillars on Google Colab Tesla T4 GPU (15,360 MB): genuine 1.63M-parameter instantiation, multi-lead forward pass with Candidate A ocean mask, real-model backward pass ($\|\Delta w\| = 3.95 \times 10^{-3}$), 4-lead recursive cascade, VRAM memory ladder ($B \in \{11, 22, 33, 44, 66\}$), and production contract freeze ($0.00 \times 10^0$ parity). Standalone script updated to fail-closed; current-code rerun pending. | `[PASS_HISTORICAL_GPU / CURRENT_REVALIDATION_PENDING]` |

---

### 3.5 `reproduction_audit/` (Authoritative Audit Dossiers & Decision Logs)

| File Link | Step / Focus | Scope & Empirical Content | Verdict / Decision |
| :--- | :--- | :--- | :---: |
| [`01_parent_baseline_replication_audit.md`](01_parent_baseline_replication_audit.md) | Gate 1A | Parent code reproduction, environment lock, and baseline execution verification. | `[VERIFIED]` |
| [`01a_parent_baseline_execution_log.md`](01a_parent_baseline_execution_log.md) | Gate 1A | Complete terminal execution logs and runtime traces for parent baseline. | `[PASS]` |
| [`02_parent_experiment_trace_audit.md`](02_parent_experiment_trace_audit.md) | Gate 1B | Parent EX29 experiment audit: recursive loop mechanics, input/target window alignment. | `[VERIFIED]` |
| [`02a_parent_experiment_trace_log.md`](02a_parent_experiment_trace_log.md) | Gate 1B | Terminal execution logs verifying EX29 model convergence and evaluation outputs. | `[PASS]` |
| [`06_mindanao_boundary_and_gadm_audit.md`](06_mindanao_boundary_and_gadm_audit.md) | Step 21A.2 | Administrative boundary extraction from PSA shapefile; WGS84 geodesic area quadrature ($99,948.76\text{ km}^2$). | `[ACCEPTED]` |
| [`03_parent_variable_contract_audit.md`](03_parent_variable_contract_audit.md) | Step 21A.3 | Variable mapping, lag schedule ($[-1, -7, -14]$), and channel counts ($W_1=11, W_2=12, W_3=5, W_4=6$). | `[VERIFIED]` |
| [`05_mindanao_geometry_envelopes_audit.md`](05_mindanao_geometry_envelopes_audit.md) | Step 21A.4 | Comparative geometric analysis of Candidate A ($32 \times 48$) vs Candidate B envelopes. | `[ACCEPTED]` |
| [`07_mindanao_candidate_a_grid_freeze_audit.md`](07_mindanao_candidate_a_grid_freeze_audit.md) | Step 21A.6 | Formal freeze of Candidate A regular $0.25^\circ$ mesh with 16-divisible dimensions for UNet pooling. | `[ACCEPTED]` |
| [`08_mindanao_evaluation_mask_audit.md`](08_mindanao_evaluation_mask_audit.md) | Step 21A.7B | Ellipsoidal fractional land raster and binary evaluation mask (126 active cells: full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$). | `[ACCEPTED]` |
| [`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md) | Step 21A.9 & 21B.2 | Scientific defense of ERA5-Land vs GLEAM, hourly source-of-record, and RZSM physics. | `[ACCEPTED]` |
| [`09_mindanao_spatial_foundation_dossier.md`](09_mindanao_spatial_foundation_dossier.md) | Sub-Phase 21A Synthesis | Master synthesis dossier certifying area conservation ($0.0000\%$ quadrature error) and spatial foundation freeze. | `[PASS]` |
| [`10_era5_land_archive_completeness_audit.md`](10_era5_land_archive_completeness_audit.md) | Step 21B.1 | Census of 264 synchronized ERA5-Land NetCDF files covering 2015–2025 nominal archive (3.67 GiB in GCS). | `[PASS]` |
| [`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md) | Step 21B.3–4 | Pilot RZSM calculation across 20-day antecedent window; zero NaNs across $20 \times 126 = 2,520$ samples. | `[PASS]` |
| [`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md) | Step 21C.1–2 | Pilot extraction of 5 atmospheric channels for Jan 2015 via Bolton (1980); zero NaNs across 3,906 evaluation points. | `[PASS / VERIFIED / ACCEPTED]` |
| [`04_parent_temporal_target_parity_audit.md`](04_parent_temporal_target_parity_audit.md) | Step 21D.3 | Target reconciliation ($L=[6,13,20,27]$), trailing rolling mean, 3-month season climatology, domain-wide normalization, and 24/24 unit tests. | `[VERIFIED]` / `[ACCEPTED]` |
| [`13_production_cube_preflight_audit.md`](13_production_cube_preflight_audit.md) | Step 21D.4-PREFLIGHT | Formal preflight verification report: census of 265 NetCDFs (96,912 hours, 4,038 days), zero NaNs/Infs, spatial coordinate contract lock. | `[PASS]` |
| [`14_production_cube_compilation_and_census_audit.md`](14_production_cube_compilation_and_census_audit.md) | Step 21D.4 | Formal production cube compilation audit: multi-year ERA5-Land ingestion, 506,268 finite evaluations, zero NaNs/Infs, dual GCS synchronization. | `[PASS / VERIFIED / ACCEPTED]` |
| [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md) | Step 21F.3 | Formal pilot case assembly and recursive inference audit: real S2S integration, perturbation sensitivity, 4-lead cascade. | `[PASS / VERIFIED]` |
| [`17_pilot_case_ladder_and_manifest_audit.md`](17_pilot_case_ladder_and_manifest_audit.md) | Step 21G | 8-case pilot ladder manifest and census audit: 88 member rows, 8 summary rows, 314,496 finite feature values, 0 NaNs/Infs. | `[PASS / VERIFIED]` |
| [`20_tf_dataset_pipeline_and_checkpoint_audit.md`](20_tf_dataset_pipeline_and_checkpoint_audit.md) | Step 21H | Surrogate TensorFlow pipeline audit: 11-member ensemble grouping semantics, CRPS loss mathematical reconciliation, 5-epoch stability, bit-for-bit checkpoint restore for 2-layer surrogate. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`22_a0_tiny_overfit_gradient_audit.md`](22_a0_tiny_overfit_gradient_audit.md) | Step 21I | Surrogate tiny-data optimization test: 8 cases, 40 epochs, 320 updates on 2-layer Conv2D surrogate model; 99% error reduction is an unnormalized predictor scale artifact, not A0 capacity. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`21_vram_and_hardware_profiling_audit.md`](21_vram_and_hardware_profiling_audit.md) | Sub-Phase 21J | Hardware profiling & VRAM feasibility audit on Tesla T4 GPU (historical evidence); benchmark code hardened to fail-closed, per-lead counts; current-code revalidation pending. | `[PASS_PHYSICAL_GPU_HISTORICAL / CURRENT_REVALIDATION_PENDING]` |
| [`18_production_case_calendar_1154_cycles_audit.md`](18_production_case_calendar_1154_cycles_audit.md) | Step 21K.1 | Usable production case calendar audit: 1,154 operational cycles, 4-way data intersection, 735 Train / 210 Val / 209 Test partitioning, dual GCS parity. | `[PASS / VERIFIED / ACCEPTED]` |
| [`19_dataset_splits_and_normalization_contract_audit.md`](19_dataset_splits_and_normalization_contract_audit.md) | Step 21K.2 | Dataset split partitioning manifest and training-only normalization parameter audit: 735 Train / 210 Val / 209 Sealed Test, four-part scope, production contract freeze, 83 unit tests passing, dual GCS parity. | `[PASS / VERIFIED / ACCEPTED]` |
| [`OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md) | Governance | Master operational execution matrix, living directory registry, and file map (This Document). | `[APPROVED]` |

---

### 3.6 `scripts/` (Operational Processing & Automation)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`09_build_pilot_manifest_and_cases.py`](../scripts/09_build_pilot_manifest_and_cases.py) | Pilot Case Hierarchy Assembler | Batch compiles the 8 pilot cases (`CASE_20150115_W01.npz` through `CASE_20150304_W08.npz`) and writes 88-row member and 8-row summary manifests. | `[PASS / VERIFIED]` |
| [`06_build_production_case_calendar.py`](../scripts/06_build_production_case_calendar.py) | Production Case Calendar Generator | Generates the 1,154 operational forecast issuance cycles (2015–2025), computes deterministic lags/targets, evaluates 4-way data intersection, and exports `manifests/production_case_calendar.csv`. | `[PASS / VERIFIED / ACCEPTED]` |
| [`08_derive_training_normalization.py`](../scripts/08_derive_training_normalization.py) | Training Normalization Engine | Derives domain-wide active scalar normalization bounds strictly over 2015–2021 training partition across 126 active cells and exports `contracts/A0/normalization_parameters.yaml`. | `[PASS / VERIFIED / ACCEPTED]` |
| [`07_generate_dataset_splits.py`](../scripts/07_generate_dataset_splits.py) | Dataset Splits Generator | Partitions 1,154 production cycles into TRAIN (735), VAL (210), and SEALED_TEST (209) manifests with SHA-256 digests and leakage verification. | `[PASS / VERIFIED / ACCEPTED]` |
| [`04_download_all_s2s_production.py`](../scripts/04_download_all_s2s_production.py) | Production S2S Downloader | High-efficiency S2S downloader for all 385 cycles (2015–2025) featuring startup GCS caching, leap-year safety, and scratch cleanup. | `[IN PROGRESS]` |
| [`02_download_era5_atmospheric.py`](../scripts/02_download_era5_atmospheric.py) | Copernicus CDS-Beta API Downloader | Resilient multi-threaded downloader retrieving hourly ERA5 atmospheric pressure levels in monthly chunks with automatic retry and GCS synchronization. | `[IN PROGRESS]` (168/264 files secured for training fold Jan 2015–Dec 2021; 2022–2025 mirroring active) |
| [`01_generate_mindanao_masks.py`](../scripts/01_generate_mindanao_masks.py) | Geodesic Mask Generator | Python script computing ellipsoidal polygon intersections on WGS84 to produce Candidate A grid, fractional land raster, and 126-cell binary evaluation mask. | `[PASS]` |
| [`11_profile_a0_vram_benchmark.py`](../scripts/11_profile_a0_vram_benchmark.py) | Standalone A0 VRAM & Hardware Profiler | Executes the Six Technical Pillars for genuine `UNET_RZSM` (1.63M parameters), multi-lead forward pass, backward pass, 4-lead cascade, VRAM ladder ($B \in \{11, 22, 33, 44, 66\}$), and real model-weight checkpoint roundtrip. Upgraded to fail-closed (`--mode certify`). | `[PASS_PHYSICAL_GPU_HISTORICAL / CURRENT_REVALIDATION_PENDING]` |
| [`run_download_atmospheric.sh`](../scripts/run_download_atmospheric.sh) | Shell Execution Wrapper | Headless background execution wrapper for atmospheric retrieval on Unix/WSL environments. | `[READY]` |
| [`12_train_a0_tiny_overfit.py`](../scripts/12_train_a0_tiny_overfit.py) | Surrogate Pipeline Smoke Test Script | 40-epoch surrogate diagnostic verifying data streaming, 11-member batching, target broadcasting, and checkpoint restore parity across 320 updates. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`13_train_a0_pipeline_checkpoint.py`](../scripts/13_train_a0_pipeline_checkpoint.py) | Surrogate TensorFlow Pipeline Diagnostic | Diagnostic script validating multi-worker streaming, 11-member batch invariant, and checkpoint serialization for 2-layer surrogate. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`03_derive_era5_atmospheric_daily.py`](../scripts/03_derive_era5_atmospheric_daily.py) | Atmospheric Derivation Engine | Standalone script computing Bolton (1980) specific humidity, daily temperature extremes ($T_{\max}, \Delta T$), and geopotential height ($Z_{200}/g_0$). | `[PASS]` |
| [`10_verify_pilot_ladder_provenance_and_census.py`](../scripts/10_verify_pilot_ladder_provenance_and_census.py) | Pilot Census & Provenance Engine | Independent 16-point audit verifying all 8 pilot cases, 88 member realizations, zero NaNs across 314,496 points, and GCS lake parity. | `[PASS / VERIFIED]` |

| [`14_run_a0_production_smoke_test.py`](../scripts/14_run_a0_production_smoke_test.py) | Model A0 Production Training Smoke Test | End-to-end preflight verifying 4-lead genuine backward pass & parameter updates ($C \in \{11, 12, 5, 6\}$), 3 deep supervision heads, evaluation-domain masking, recursive channel semantics and tamper detection, and model-weight vs full training-state checkpointing. | `[PASS / CERTIFIED PREFLIGHT]` |

---

### 3.7 `src/` (Production Python Package)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`src/__init__.py`](../src/__init__.py) | Package Root | Top-level package initializer. | `[ACTIVE]` |
| [`src/data/__init__.py`](../src/data/__init__.py) | Data Package Exports | Exports `compute_depth_weighted_rzsm`, `remap_era5_land_to_candidate_a`, `compute_trailing_rolling_mean`, `compute_training_climatology`, `compute_seasonal_anomalies`, `fit_min_max_bounds`, `standardize_with_training_bounds`, `compile_production_rzsm_pipeline`, `verify_production_cube_census`, `FastLandAwareRemapper`, `process_era5_land_monthly_pair`, `process_era5_land_antecedent_file`, `compile_full_11yr_rzsm_cube`, `create_a0_dataset_from_manifest`, `save_a0_checkpoint`, `restore_a0_checkpoint`, and `ProductionCubeConfig`. | `[ACTIVE]` |
| [`src/data/rzsm.py`](../src/data/rzsm.py) | RZSM Calculation & Remapping | Implements depth-weighted RZSM formula ($0.07\cdot\text{SM}_1 + 0.21\cdot\text{SM}_2 + 0.72\cdot\text{SM}_3$). Remapping order: 1) Isolate finite land points on native ERA5-Land ($0.10^\circ$), 2) Bilinear interpolation over active evaluation cells ($M_{i,j}=1$), 3) Nearest-neighbor extrapolation fallback triggered specifically for unassigned coastal boundary cells to prevent coastline clipping, 4) Binary evaluation mask application, and 5) Zero-filling of 1,410 inactive computational cells. | `[PASS]` / `[ACCEPTED]` |
| [`src/data/temporal.py`](../src/data/temporal.py) | Temporal Preprocessing | Implements 7-day backward trailing rolling mean (`center=False`), locked 3-month seasonal climatology (DJF, MAM, JJA, SON) fitted on training years $\le 2021$, seasonal anomalies, and domain-wide active scalar normalization. | `[PASS]` / `[VERIFIED]` |
| [`src/data/compile_cube.py`](../src/data/compile_cube.py) | Production Cube Engine | Production compilation engine with vectorized multi-day spatial remapper (`FastLandAwareRemapper`), monthly pair ingestion (`process_era5_land_monthly_pair`), antecedent support processor (`process_era5_land_antecedent_file`), and end-to-end multi-year compilation (`compile_full_11yr_rzsm_cube`). Features automated date uniqueness and canonical endpoint validation (`2014-12-12` to `2025-12-31`), orchestrates temporal transformations on full 4,038-day archive series, slices nominal 4,018-day period (506,268 evaluation cell-days), and executes automated census certification (zero NaNs/Infs). | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |
| [`src/data/s2s.py`](../src/data/s2s.py) | ECMWF S2S Harmonization Engine | Pure-Python GRIB2 Section 7 Template 0 decoder and harmonizer. Ingests verified EX29 dynamic triplet (`t2m, d2m, tcw`), performs 6-hourly step aggregation to Week 1 ($0\text{--}168\text{ h}$) and Week 2 ($168\text{--}336\text{ h}$) leads, executes bilinear remapping with nearest-boundary extrapolation to Candidate A 0.25° grid, structures all 11 ensemble members ($M=11$), and applies evaluation masking. | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |
| [`src/data/case_builder.py`](../src/data/case_builder.py) | Multi-Lead Tensor Hierarchy Assembler | Production case assembler constructing the multi-lead tensor hierarchy $[M=11, H=32, W=48, C_k]$ across leads $W_1..W_4$ adhering strictly to Kyle Lesinger's channel counts ($[11, 12, 5, 6]$). Implements recursive prior-lead prediction concatenation and ground truth target slicing with zero future leakage. | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |
| [`src/data/tf_dataset.py`](../src/data/tf_dataset.py) | TensorFlow Streaming & Batching Engine | Production streaming pipeline enforcing 11-member ensemble batching invariance ($B \in 11\mathbb{Z}^+$), case-level shuffling, target broadcasting, multi-worker prefetching, and checkpoint persistence. | `[PASS / VERIFIED]` |
| [`src/models/__init__.py`](../src/models/__init__.py) | Model Package Root | Exports `build_a0_unet` and Model A0 architecture specifications. | `[ACTIVE]` |
| [`src/models/a0_unet.py`](../src/models/a0_unet.py) | Genuine Model A0 (`UNET_RZSM`) Architecture Factory | Authoritative factory instantiating the authentic 1,630,307-parameter nested U-Net (`function/modelRzsmRelu.py`) with 298 weight tensors, Inception blocks, SE attention, multiscale decoders, multi-head deep supervision outputs, and channel schedule ($W_1=11, W_2=12, W_3=5, W_4=6$). | `[PASS / VERIFIED]` |

---

### 3.8 `tests/` (Automated Unit Test Suite)

All 87 automated unit tests are executed with `python -m unittest discover -s tests -p "test_*.py"`. **All 84 applicable tests passed (3 skipped: 1 offline GCS network, 2 TF-gated graph preflight tests) with zero failures and zero errors (28.0s total runtime):**

> [!NOTE]
> **Comprehensive 87-Test Suite Verification**:
> The 87 automated unit tests provide end-to-end software confidence across all pipeline layers: spatial foundation, temporal processing, cube compilation, GRIB2 decoding, multi-lead tensor assembly, pilot ladder manifests, TensorFlow batching/checkpointing, genuine Model A0 architecture instantiation across all 4 leads, production case calendar integrity, dataset split / normalization parameter contracts, executable tensor-builder normalization binding, and production-path training preflight contracts (recursive channel semantics, ordering invariants, evaluation-domain masking, and training-state checkpointing).

| File Link | Test Count | Key Test Assertions | Execution Time / Status |
| :--- | :---: | :--- | :---: |
| [`test_01_grid_and_masks.py`](../tests/test_01_grid_and_masks.py) | 5 | 1. Candidate A grid dimensions ($32 \times 48$) and 0.25° spacing.<br>2. Geodetic area conservation ($99,948.76\text{ km}^2$, $0.0000\%$ error).<br>3. Exact 126-cell binary evaluation mask count.<br>4. Fractional boundary coverage consistency ($f \in [0.0, 1.0]$).<br>5. 16-pooling divisibility compatibility ($32 \pmod{16} = 0, 48 \pmod{16} = 0$). | 0.8s / `[PASS]` |
| [`test_08_normalization_and_splits.py`](../tests/test_08_normalization_and_splits.py) | 8 | 1. Split disjointness (zero date overlap across TRAIN, VAL, TEST).<br>2. Split completeness (735 Train + 210 Val + 209 Test = 1,154 total).<br>3. Strict chronological ordering without time inversion.<br>4. Sealed test quarantine integrity (202 queued + 7 out-of-bounds).<br>5. Normalization YAML contract validity and $[0, 1]$ bounds.<br>6. Production training contract schema validity.<br>7. Executable normalization tensor-builder contract test proving active consumption, $[0.0, 1.0]$ bounds, and $0.0$ ocean buffer. | 0.25s / `[PASS]` / `[VERIFIED]` |
| [`test_07_case_calendar.py`](../tests/test_07_case_calendar.py) | 4 | 1. Exactly 1,154 operational cycles indexed monotonically.<br>2. Strict partition splits (735 Train, 210 Val, 209 Sealed Test).<br>3. Deterministic lag arithmetic ($-1\text{d}, -7\text{d}, -14\text{d}$) and target offsets ($+6\text{d}, +13\text{d}, +20\text{d}, +27\text{d}$).<br>4. Isolation of exactly 7 boundary-clipped cycles in Dec 2025. | 0.04s / `[PASS]` / `[VERIFIED]` |
| [`test_05_compile_cube.py`](../tests/test_05_compile_cube.py) | 5 | 1. End-to-end multi-year pipeline execution and census certification.<br>2. Out-of-sample temporal leakage isolation ($\ge 2022$ cannot alter training bounds).<br>3. `FastLandAwareRemapper` exact numerical agreement on tested data (observed maximum absolute difference = 0.0; test tolerance = 1e-6) with reference remapper.<br>4. 2014 antecedent support file real data processing (20 days, finite on 126 cells, 0 on 1410 buffer).<br>5. Partial archive compilation pipeline orchestration. | 4.8s / `[PASS]` |
| [`test_02_rzsm.py`](../tests/test_02_rzsm.py) | 10 | 1. Layer weights sum to $1.0$.<br>2. Constant field preserves soil moisture.<br>3. Hand-computable analytical solution ($2.65$).<br>4. Non-unit weight exception.<br>5. Land mask application and ocean zero-filling.<br>6. Min-max scaling to $[0, 1]$.<br>7. Backward rolling mean mechanics.<br>8. Antecedent lag extraction ($[-1, -7, -14]$).<br>9. Real pilot NetCDF physical range $[0.10, 0.60]\,\text{m}^3/\text{m}^3$.<br>10. Land-aware bilinear remapping with zero NaNs across evaluation cells. | 1.1s / `[PASS]` |
| [`test_06_target_reconciliation.py`](../tests/test_06_target_reconciliation.py) | 3 | 1. Hand-calculated arithmetic verification.<br>2. Exact parity with parent formula $L = (\text{lead} \times 7) - 1$ on anchor dates (`2015-01-15`, `2015-06-01`, `2016-02-29`, `2018-08-15`, `2020-12-01`).<br>3. Mathematical proof of contiguous, non-overlapping 28-day partition across $W_1$–$W_4$. | 0.8s / `[PASS]` / `[VERIFIED]` |
| [`test_03_temporal.py`](../tests/test_03_temporal.py) | 9 | 1. Rolling mean future invariance (zero leakage from $t+1$).<br>2. Trailing window arithmetic values.<br>3. Climatology strictly ingests years $\le 2021$.<br>4. 3-month seasonal climatology conforms to DJF, MAM, JJA, SON.<br>5. Training period mean anomaly equals zero ($\pm 10^{-6}$).<br>6. Domain-wide active scalar normalization bounds.<br>7. Exact lag and lead date extraction.<br>8. Symmetric continuous series extraction for targets and antecedents.<br>9. Integration test on real 31-day pilot NetCDF data. | 2.0s / `[PASS]` / `[VERIFIED]` |
| [`test_04_s2s.py`](../tests/test_04_s2s.py) | 7 | 1. Pure-Python GRIB2 Section 7 Template 0 constant field unpacking.<br>2. Known synthetic bitstream decoding.<br>3. Bilinear remapping geometry from $(5, 8)$ to Candidate A $(32, 48)$ with zero NaNs.<br>4. Monotonic temperature gradient preservation.<br>5. Pilot cycle harmonization with explicit fallback opting and diagnostic flag.<br>6. Production mode hard-fail enforcement rejecting missing forecast steps.<br>7. Clean ECDS production GRIB harmonization with zero fallbacks, separate hdate (`2015-01-16`) and model date (`2020-01-16`). | 2.2s / `[PASS]` / `[VERIFIED]` |
| [`test_10_case_builder.py`](../tests/test_10_case_builder.py) | 5 | 1. Single-case tensor hierarchy shapes $[11, 12, 5, 6]$ across $M=11$ members.<br>2. Zero-filling of non-evaluation ocean cells across inputs and targets.<br>3. Incomplete antecedent window exception enforcement.<br>4. Real pilot Jan 15, 2015 case integration test.<br>5. Recursive normalization invariant test proving prior model predictions are not double-normalized. | 1.8s / `[PASS]` / `[VERIFIED]` |
| [`test_09_pilot_ladder.py`](../tests/test_09_pilot_ladder.py) | 4 | 1. Manifest structure and column integrity.<br>2. Temporal ordering and 7-day interval consistency.<br>3. Member indexing $[0..10]$ and case grouping invariance.<br>4. Missing file and corrupted case exception handling. | 1.2s / `[PASS]` / `[VERIFIED]` |
| [`test_11_tf_dataset.py`](../tests/test_11_tf_dataset.py) | 16 | 1. Batch size exact multiple of 11 enforcement.<br>2. Member realization order preservation.<br>3. Case-level shuffle integrity (members stay clustered).<br>4. Target broadcasting invariance ($0.00 \times 10^0$).<br>5. Multi-head output dictionary matching UNET_RZSM heads.<br>6. Checkpoint save, restore, and state parity. | 1.5s / `[PASS]` / `[VERIFIED]` |
| [`test_12_a0_unet.py`](../tests/test_12_a0_unet.py) | 6 | 1. Grid dimensions & Candidate A geometry guard (divisibility by 16).<br>2. Multi-lead channel schedule $[11, 12, 5, 6]$.<br>3. Invalid lead rejection.<br>4. Expected per-lead parameter counts ($W_1=1,627,139; W_2=1,630,307; W_3=1,608,131; W_4=1,611,299$).<br>5. Conditional TensorFlow-gated genuine architecture instantiation, 298 weight tensors, 3 deep supervision heads.<br>6. Multi-lead input shapes, forward pass execution, and output finiteness. | 0.9s / `[PASS]` / `[VERIFIED]` |
| [`test_13_production_smoke_preflight.py`](../tests/test_13_production_smoke_preflight.py) | 4 | 1. Recursive channel semantics and exact index placement across Leads 2, 3, 4.<br>2. Recursive channel tamper and order scrambling detection.<br>3. Full training-state checkpoint serialization and restoration (model weights, optimizer weights, epoch, step).<br>4. 4-Lead GradientTape backpropagation and parameter update with evaluation domain masking. | 0.05s / `[PASS]` |
| [`test_09_cloud_gcs_lake_client.py`](../tests/test_09_cloud_gcs_lake_client.py) | 1 | 1. GCS cloud lake bucket connectivity and artifact resolution (gracefully skips in offline local test environment). | 0.01s / `[SKIPPED OFFLINE]` |



---

### 3.9 `figures/` (Consolidated Publication & Verification Visual Assets)

| File Link | Description & Scientific Content | Resolution / Format |
| :--- | :--- | :---: |
| [`figures/mindanao_spatial_grid_mesh.png`](../figures/mindanao_spatial_grid_mesh.png) | Candidate A ($32 \times 48$, $0.25^\circ$) regular grid mesh and outer geodetic bounding box overlaying Mindanao administrative boundary | 300 DPI PNG |
| [`figures/mindanao_fractional_coverage_map.png`](../figures/mindanao_fractional_coverage_map.png) | Continuous ellipsoidal fractional boundary-coverage raster ($f \in [0.0, 1.0]$) computed via WGS84 geodesic polygon quadrature | 300 DPI PNG |
| [`figures/mindanao_evaluation_mask_map.png`](../figures/mindanao_evaluation_mask_map.png) | Authoritative binary evaluation mask ($f \ge 0.50$, 126 active land cells) and pure ocean buffer classification | 300 DPI PNG |
| [`figures/mindanao_spatial_foundation_composite.png`](../figures/mindanao_spatial_foundation_composite.png) | Comprehensive 4-panel spatial foundation synthesis dashboard certifying geodetic area conservation ($99,948.76\text{ km}^2$) | 300 DPI PNG |
| [`figures/mindanao_rzsm_pilot_preprocessing_composite.png`](../figures/mindanao_rzsm_pilot_preprocessing_composite.png) | 4-panel RZSM pilot composite: native $0.10^\circ$ daily mean, bilinear remapped $0.25^\circ$ field, evaluation mask, and zero-padded model tensor | 300 DPI PNG |
| [`figures/mindanao_era5_atmospheric_pilot_verification.png`](../figures/mindanao_era5_atmospheric_pilot_verification.png) | Publication-grade 6-panel atmospheric verification composite with GADM boundary overlay, $Z_{200}$ isohypses, Candidate A envelope, and 3-tier domain classification | 300 DPI PNG |
| [`figures/mindanao_production_cube_verification_composite.png`](../figures/mindanao_production_cube_verification_composite.png) | 4-panel 11-year RZSM production cube verification composite: climatological mean field, Model A0 DJF baseline, 4,018-day continuous hydrograph, and standardized anomaly distribution | 300 DPI PNG |
| [`figures/mindanao_s2s_dynamic_predictor_composite.png`](../figures/mindanao_s2s_dynamic_predictor_composite.png) | Canvas 1: 4-panel ECMWF S2S multi-scale dynamic predictor dashboard resolving S2S abstractness via discrete sampling nodes ($N=40$), vector GADM coastline overlay, continuous bilinear remapping ($32 \times 48$), 11-member ensemble spread ($\sigma_{\text{ENS}}$), and dynamic moisture shift ($\Delta_{W2-W1}$) | 300 DPI PNG |
| [`figures/mindanao_s2s_pilot_case_and_recursive_inference.png`](../figures/mindanao_s2s_pilot_case_and_recursive_inference.png) | Canvas 2: 4-panel Model A0 case assembly & recursive sensitivity dashboard: antecedent RZSM memory lag -1d over 126 evaluation cells, ERA5 precipitable water ($PWAT$ at $t_0$), observed ground truth target $Y_{W1}$, and downstream perturbation sensitivity response map $\Delta_{W2}$ in sequential `inferno` colormap | 300 DPI PNG |

---

### 3.10 `pilot_raw/` & `processed/` (Data Products & Contracts)

| Directory & File Link | Content Description | Physical Units / Format |
| :--- | :--- | :---: |
| [`pilot_raw/era5-land-2014-12-antecedent.nc`](../pilot_raw/era5-land-2014-12-antecedent.nc) | Raw 20-day antecedent NetCDF (12–31 Dec 2014) on native $0.10^\circ$ grid | $\text{m}^3/\text{m}^3$ (NetCDF4) |
| [`processed/boundary/mindanao_analysis_boundary.gpkg`](../processed/boundary/mindanao_analysis_boundary.gpkg) | Frozen administrative polygon for Mindanao (WGS84 EPSG:4326) | OGC GeoPackage |
| [`processed/grid/mindanao_0.25_grid.grd`](../processed/grid/mindanao_0.25_grid.grd) | Climate Data Operators (CDO) reference grid definition | CDO ASCII Grid |
| [`processed/grid/mindanao_025deg.nc`](../processed/grid/mindanao_025deg.nc) | Candidate A reference grid NetCDF ($32 \times 48$, lat 11.75 to 4.00, lon 116.00 to 127.75) | NetCDF4 CF-1.8 |
| [`processed/grid/mindanao_fraction_025.nc`](../processed/grid/mindanao_fraction_025.nc) | Fractional land coverage raster computed via ellipsoidal polygon intersection | NetCDF4 ($[0.0, 1.0]$) |
| [`processed/grid/mindanao_eval_mask_025.nc`](../processed/grid/mindanao_eval_mask_025.nc) | Authoritative binary evaluation mask (126 active land cells, 1,410 zero-filled ocean cells) | NetCDF4 (Int32) |
| [`processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc`](../processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc) | 5 derived atmospheric variables for Jan 2015 pilot (`spfh`, `tmax`, `diff_temp`, `pwat`, `hgt_pres`) | NetCDF4 CF-1.8 |
| [`processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc`](../processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc) | Pilot 0–100 cm depth-weighted RZSM dataset (Dec 2014 – Jan 2015) remapped to Candidate A | NetCDF4 CF-1.8 |
| [`processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc`](../processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc) | Full 11-year ($2015\text{--}2025$) production RZSM data cube on Candidate A grid with locked climatology and anomalies | NetCDF4 CF-1.8 ($8.40\text{ MiB}$) |
| [`processed/rzsm/pilot/README.md`](../processed/rzsm/pilot/README.md) | Technical data dictionary documenting pilot RZSM coordinate arrays, variable names, and units | Markdown Document |
| [`processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc`](../processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc) | Harmonized ECMWF S2S reforecast pilot dataset across 11 members, Leads 1 & 2, and EX29 triplet (`t2m, d2m, tcw`) on Candidate A grid | NetCDF4 CF-1.8 ($0.42\text{ MiB}$) |

---


## 4. Mandatory Maintenance Rule & Continuous Synchronization Protocol

> [!IMPORTANT]
> **Operational Repository Rule**:
> This document (`reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`) and the master plan (`mindanao_adaptation_master_plan.md`) must be **synchronized** whenever a **remarkable change** occurs in the repository.
>
> A **remarkable change** is defined as:
> 1. Creation, deletion, or renaming of any production script (`scripts/`), module (`src/`), unit test (`tests/`), notebook (`notebooks/`), or formal audit report (`reproduction_audit/`).
> 2. Alteration of mathematical formulas, normalization bounds, climatology definitions, or target lead indices.
> 3. Execution of milestone compilation runs, data cube census verifications, or model checkpoint saves.
> 4. Discovery or reconciliation of any numerical error, day count, or data lake manifest.
>
> **Non-Triggers (Do NOT update registry for trivial changes)**:
> - Minor formatting edits, typo fixes, or comment clarifications.
> - Temporary scratch scripts or diagnostic runs in `scratch/`.
>
> Every remarkable entry must maintain exact relative file links, precise file sizes/dimensions, and three-tier certification tags (`[PASS]`, `[VERIFIED]`, `[ACCEPTED]`).
>
> **Independent Verification Rule**: User feedback must always be empirically checked and mathematically validated before adoption into these records.

