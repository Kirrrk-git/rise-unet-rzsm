<!-- markdownlint-disable -->
# Mindanao RISE-UNet Operational Execution Matrix, Audit Dossier Index & Repository File Registry

**Project**: Enhanced RISE-UNet for Subseasonal Root-Zone Soil Moisture (RZSM) Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Parent Baseline**: Lesinger & Tian (2025), *Nature Communications*, DOI: [`10.1038/s41467-025-62761-3`](https://doi.org/10.1038/s41467-025-62761-3)  
**Target Model**: **Mindanao Model A0** (Adapted from EX29 Recursive Hybrid RISE-UNet Baseline)  
**Document Classification**: Living Operational Matrix, Comprehensive Directory Registry & Master File Map  
**Last Updated**: 2026-09-12  

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
| **Gate 1A** | Parent Code Reproduction & Environment Freeze | EX29 Environment Parity | `[VERIFIED]` | [`GATE1A_AUTHORITATIVE_BASELINE_EVIDENCE_DOSSIER.md`](GATE1A_AUTHORITATIVE_BASELINE_EVIDENCE_DOSSIER.md) |
| **Gate 1B** | Parent Pipeline Trace & Recursive Loop Verification | EX29 Recursive Logic Parity | `[VERIFIED]` | [`GATE1B_AUTHORITATIVE_EXPERIMENT_EVIDENCE_DOSSIER.md`](GATE1B_AUTHORITATIVE_EXPERIMENT_EVIDENCE_DOSSIER.md) |
| **Step 21A.1** | Mindanao Boundary Geodesy & Area Quadrature | PSA Admin Boundaries / WGS84 | `[ACCEPTED]` | [`MINDANAO_REGIONAL_BOUNDARY_AUDIT_REPORT.md`](MINDANAO_REGIONAL_BOUNDARY_AUDIT_REPORT.md) |
| **Step 21A.2** | Candidate A $32 \times 48$ Reference Grid Freeze | 16-Divisible Spatial Mesh | `[ACCEPTED]` | [`MINDANAO_SPATIAL_GRID_FREEZE_REPORT.md`](MINDANAO_SPATIAL_GRID_FREEZE_REPORT.md) |
| **Step 21A.3** | Ellipsoidal Fractional Coverage & Binary Evaluation Mask | 126 Active Land Cells ($>10\%$) | `[ACCEPTED]` | [`MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md`](MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md) |
| **Phase 21A** | Spatial Foundation Synthesis & Geodetic Certification | Complete Geodetic Quadrature | `[PASS]` | [`SPATIAL_FOUNDATION_COMPREHENSIVE_VERIFICATION_DOSSIER.md`](SPATIAL_FOUNDATION_COMPREHENSIVE_VERIFICATION_DOSSIER.md) |
| **Step 21B.1** | ERA5-Land Archive Ingestion & GCS Mirroring | 265 NetCDF Files (12 Dec 2014–2025) | `[PASS]` | [`ERA5_LAND_ARCHIVE_INTEGRITY_COMPLETENESS_AUDIT.md`](ERA5_LAND_ARCHIVE_INTEGRITY_COMPLETENESS_AUDIT.md) |
| **Step 21B.2** | 2014 Antecedent Window Retrieval (20 Days) | 12–31 Dec 2014 Support Window | `[ACCEPTED]` | [`ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md`](ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md) |
| **Step 21B.3** | Daily Mean & Depth Weighting ($0.07, 0.21, 0.72$) | 0–100 cm Physical Profile | `[VERIFIED]` | [`ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md`](ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md) |
| **Step 21B.4** | Bilinear Remapping & Coastal Extrapolation Fallback | 2D Remapping to Candidate A | `[ACCEPTED]` | [`ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md`](ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md) |
| **Step 21C.1** | ERA5 Atmospheric Pilot Retrieval (Jan 2015) | Hourly Levels (CDS-Beta API) | `[PASS]` | [`ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md`](ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md) |
| **Step 21C.2** | 5-Channel Atmospheric Derivation & Census | Bolton (1980) Humidity & Extr. | `[VERIFIED]` | [`ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md`](ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md) |
| **Step 21C.3** | 11-Year ERA5 Atmospheric Archive Mirroring | Hourly Monthly NetCDFs to GCS | `[IN PROGRESS]` | ERA5 Atmospheric Ingestion Log |
| **Step 21D.1** | Depth-Weighted RZSM Modular Implementation | Vectorized `src/data/rzsm.py` | `[PASS]` | Automated Unit Tests (`test_rzsm.py`) |
| **Step 21D.2** | Automated RZSM Unit Test Suite | Precision $\le 10^{-7}$, Real Pilot | `[PASS]` | 10 Unit Tests Passing |
| **Step 21D.3** | Temporal Preprocessing & Target Lead Reconciliation | $L=[6,13,20,27]$, Trailing Rolling | `[PASS / VERIFIED]` | Satisfies software tests & explicitly reconciled parent contracts ([`Audit`](EX29_TEMPORAL_PREPROCESSING_AND_TARGET_PARITY_AUDIT.md)) |
| **Step 21D.4-PREFLIGHT** | Production Cube Preflight Verification Gate | 11-Point Preflight Census & Integrity Gate | `[READY FOR EXECUTION]` | Preflight Checklist (Census 265, 96,912 hours across 4,038 days, Bounds, Spatial Contract Consistency) |
| **Step 21D.4** | Full Production 11-Year RZSM Cube Compilation | 4,038 Archive / 4,018 Nominal Days | `[READY: PENDING PREFLIGHT]` | `notebooks/06_...ipynb` / Production NetCDF |
| **Sub-Phase 21E** | ECMWF S2S Reforecast Pilot Retrieval | Dynamic Triplet (`t2m, d2m, tcw`) | `[PLANNED]` | S2S Reforecast Acquisition Audit |
| **Gate 2** | Preprocessing Go/No-Go Certification | Multi-Channel Tensor Census | `[PLANNED]` | `GATE2_PREPROCESSING_AND_DATA_CUBE_AUDIT.md` |
| **Phase 22** | Spatio-Temporal Dataset Splitting & DataLoader | Train (15–21), Val (22–23), Test (24–25)| `[PLANNED]` | Dataset Split & Boundary Integrity Audit |
| **Phase 23** | Model A0 Training Across Leads W1–W4 | Recursive Hybrid ConvLSTM-UNet | `[PLANNED]` | Model Training & Loss Convergence Dossier |
| **Phase 24** | Model A1 Refinement & Enhanced Physics | Lead-Aware Residual Refinement | `[PLANNED]` | Model A1 Architecture & Parity Audit |
| **Phase 25** | Benchmark Baselines (Climatology, Persistence, XGBoost)| Standard S2S Drought Baselines | `[PLANNED]` | Comparative Baseline Evaluation Report |
| **Phase 26** | Explainability & Attribution Analysis | Gradient SHAP / Spatio-Temporal | `[PLANNED]` | Model Interpretability & Attribution Dossier |
| **Phase 27** | Final Unblinded Test Evaluation & Significance | Block-Aware Bootstrap ($p < 0.05$) | `[PLANNED]` | **Gate 3 Final Thesis Defense Dossier** |

---

## 3. Comprehensive Repository File Registry (The Complete Project Map)

This section catalogs every directory and file introduced for the Mindanao adaptation. Original parent codebase files (such as root notebooks `00_min_max...` through `10a_...`) are excluded to maintain absolute clarity on adaptation additions.

```
dl_dm_rzsm_subseasonal_forecast/
├── contracts/
│   └── spatial/
│       └── spatial_grid_contract.yaml
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
├── metadata/
│   └── grid_definition.yaml
├── notebooks/
│   ├── 03_mindanao_spatial_foundation_and_mask_pipeline.ipynb
│   ├── 04_mindanao_rzsm_pilot_preprocessing.ipynb
│   └── 05_mindanao_atmospheric_and_rzsm_preprocessing.ipynb
├── reproduction_audit/
│   ├── OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md (This File)
│   ├── ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md
│   ├── ERA5_LAND_ARCHIVE_INTEGRITY_COMPLETENESS_AUDIT.md
│   ├── ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md
│   ├── ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md
│   ├── EX29_TEMPORAL_PREPROCESSING_AND_TARGET_PARITY_AUDIT.md
│   ├── EX29_VARIABLE_CONTRACT_AUDIT.md
│   ├── GATE1A_AUTHORITATIVE_BASELINE_EVIDENCE_DOSSIER.md
│   ├── GATE1A_PARENT_BASELINE_EXECUTION_RECORD.md
│   ├── GATE1B_AUTHORITATIVE_EXPERIMENT_EVIDENCE_DOSSIER.md
│   ├── GATE1B_PARENT_EXPERIMENT_TRACE_RECORD.md
│   ├── MINDANAO_CANDIDATE_GEOMETRY_ENVELOPES.md
│   ├── MINDANAO_REGIONAL_BOUNDARY_AUDIT_REPORT.md
│   ├── MINDANAO_SPATIAL_GRID_FREEZE_REPORT.md
│   ├── MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md
│   └── SPATIAL_FOUNDATION_COMPREHENSIVE_VERIFICATION_DOSSIER.md
├── scripts/
│   ├── download_era5_atmospheric_mindanao.py
│   ├── generate_mindanao_masks.py
│   ├── run_download_atmospheric.sh
│   └── verify_and_derive_era5_pilot.py
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── rzsm.py
│       ├── temporal.py
│       └── compile_cube.py
├── tests/
│   ├── __init__.py
│   ├── test_rzsm.py
│   ├── test_temporal.py
│   ├── test_target_reconciliation.py
│   └── test_compile_cube.py
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
    │   ├── mindanao_fraction_025.qml / mindanao_eval_mask_025.qml
    │   └── figures/
    │       ├── mindanao_spatial_grid_mesh.png
    │       ├── mindanao_fractional_coverage_map.png
    │       ├── mindanao_evaluation_mask_map.png
    │       ├── mindanao_spatial_foundation_composite.png
    │       └── mindanao_rzsm_pilot_preprocessing_composite.png
    ├── atmospheric/
    │   ├── pilot/
    │   │   └── era5_atmospheric_pilot_2015_01.nc
    │   └── figures/
    │       └── mindanao_era5_atmospheric_pilot_verification.png
    └── rzsm/
        └── pilot/
            ├── era5_land_rzsm_pilot_2014_2015.nc
            └── README.md
```

---

### 3.1 `contracts/` (Spatial & Mathematical Specifications)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml) | Spatial grid specification contract | Locks Candidate A coordinates (`lat`: 11.75 to 4.00, `lon`: 116.00 to 127.75, shape $32 \times 48$), resolution ($0.25^\circ$), geodetic area tolerances ($99,948.76\text{ km}^2$), evaluation threshold ($f \ge 0.50$), and active cell count (126). | `[FROZEN]` |

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

---

### 3.5 `reproduction_audit/` (Authoritative Audit Dossiers & Decision Logs)

| File Link | Step / Focus | Scope & Empirical Content | Verdict / Decision |
| :--- | :--- | :--- | :---: |
| [`GATE1A_AUTHORITATIVE_BASELINE_EVIDENCE_DOSSIER.md`](GATE1A_AUTHORITATIVE_BASELINE_EVIDENCE_DOSSIER.md) | Gate 1A | Parent code reproduction, environment lock, and baseline execution verification. | `[VERIFIED]` |
| [`GATE1A_PARENT_BASELINE_EXECUTION_RECORD.md`](GATE1A_PARENT_BASELINE_EXECUTION_RECORD.md) | Gate 1A | Complete terminal execution logs and runtime traces for parent baseline. | `[PASS]` |
| [`GATE1B_AUTHORITATIVE_EXPERIMENT_EVIDENCE_DOSSIER.md`](GATE1B_AUTHORITATIVE_EXPERIMENT_EVIDENCE_DOSSIER.md) | Gate 1B | Parent EX29 experiment audit: recursive loop mechanics, input/target window alignment. | `[VERIFIED]` |
| [`GATE1B_PARENT_EXPERIMENT_TRACE_RECORD.md`](GATE1B_PARENT_EXPERIMENT_TRACE_RECORD.md) | Gate 1B | Terminal execution logs verifying EX29 model convergence and evaluation outputs. | `[PASS]` |
| [`MINDANAO_REGIONAL_BOUNDARY_AUDIT_REPORT.md`](MINDANAO_REGIONAL_BOUNDARY_AUDIT_REPORT.md) | Step 21A.1 | Administrative boundary extraction from PSA shapefile; WGS84 geodesic area quadrature ($99,948.76\text{ km}^2$). | `[ACCEPTED]` |
| [`MINDANAO_CANDIDATE_GEOMETRY_ENVELOPES.md`](MINDANAO_CANDIDATE_GEOMETRY_ENVELOPES.md) | Step 21A.2 | Comparative geometric analysis of Candidate A ($32 \times 48$) vs Candidate B envelopes. | `[ACCEPTED]` |
| [`MINDANAO_SPATIAL_GRID_FREEZE_REPORT.md`](MINDANAO_SPATIAL_GRID_FREEZE_REPORT.md) | Step 21A.2 | Formal freeze of Candidate A regular $0.25^\circ$ mesh with 16-divisible dimensions for UNet pooling. | `[ACCEPTED]` |
| [`MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md`](MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md) | Step 21A.3 | Ellipsoidal fractional land raster and binary evaluation mask (126 active cells, 86.46% land area). | `[ACCEPTED]` |
| [`SPATIAL_FOUNDATION_COMPREHENSIVE_VERIFICATION_DOSSIER.md`](SPATIAL_FOUNDATION_COMPREHENSIVE_VERIFICATION_DOSSIER.md) | Phase 21A | Master synthesis dossier certifying area conservation ($0.0000\%$ quadrature error) and spatial foundation freeze. | `[PASS]` |
| [`ERA5_LAND_ARCHIVE_INTEGRITY_COMPLETENESS_AUDIT.md`](ERA5_LAND_ARCHIVE_INTEGRITY_COMPLETENESS_AUDIT.md) | Step 21B.1 | Census of 265 synchronized ERA5-Land NetCDF files covering the 12 Dec 2014–31 Dec 2025 support window (3.67 GiB in GCS). | `[PASS]` |
| [`ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md`](ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md) | Step 21B.2 | Scientific defense of ERA5-Land vs GLEAM, Copernicus daily statistics derivation, and RZSM physics. | `[ACCEPTED]` |
| [`ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md`](ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md) | Step 21B.3–4 | Pilot RZSM calculation across 20-day antecedent window; zero NaNs across $20 \times 126 = 2,520$ samples. | `[PASS]` |
| [`ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md`](ERA5_ATMOSPHERIC_PILOT_AND_RZSM_PIPELINE_AUDIT.md) | Step 21C | Pilot extraction of 5 atmospheric channels for Jan 2015 via Bolton (1980); zero NaNs across 3,906 evaluation points. | `[PASS]` |
| [`EX29_VARIABLE_CONTRACT_AUDIT.md`](EX29_VARIABLE_CONTRACT_AUDIT.md) | Contracts | Variable mapping between parent EX29 dataset and Mindanao regional data ecosystem. | `[VERIFIED]` |
| [`EX29_TEMPORAL_PREPROCESSING_AND_TARGET_PARITY_AUDIT.md`](EX29_TEMPORAL_PREPROCESSING_AND_TARGET_PARITY_AUDIT.md) | Step 21D.3 | Target reconciliation ($L=[6,13,20,27]$), trailing rolling mean, 3-month season climatology, domain-wide normalization, and 24/24 unit tests. | `[VERIFIED]` / `[ACCEPTED]` |
| [`OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md) | Governance | Master operational execution matrix, living directory registry, and file map (This Document). | `[APPROVED]` |

---

### 3.6 `scripts/` (Operational Processing & Automation)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`download_era5_atmospheric_mindanao.py`](../scripts/download_era5_atmospheric_mindanao.py) | Copernicus CDS-Beta API Downloader | Resilient multi-threaded downloader retrieving hourly ERA5 atmospheric pressure levels in monthly chunks with automatic retry and GCS synchronization. | `[IN PROGRESS]` |
| [`generate_mindanao_masks.py`](../scripts/generate_mindanao_masks.py) | Geodesic Mask Generator | Python script computing ellipsoidal polygon intersections on WGS84 to produce Candidate A grid, fractional land raster, and 126-cell binary evaluation mask. | `[PASS]` |
| [`run_download_atmospheric.sh`](../scripts/run_download_atmospheric.sh) | Shell Execution Wrapper | Headless background execution wrapper for atmospheric retrieval on Unix/WSL environments. | `[READY]` |
| [`verify_and_derive_era5_pilot.py`](../scripts/verify_and_derive_era5_pilot.py) | Atmospheric Derivation Engine | Standalone script computing Bolton (1980) specific humidity, daily temperature extremes ($T_{\max}, \Delta T$), and geopotential height ($Z_{200}/g_0$). | `[PASS]` |

---

### 3.7 `src/` (Production Python Package)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`src/__init__.py`](../src/__init__.py) | Package Root | Top-level package initializer. | `[ACTIVE]` |
| [`src/data/__init__.py`](../src/data/__init__.py) | Data Package Exports | Exports `compute_depth_weighted_rzsm`, `remap_era5_land_to_candidate_a`, `compute_trailing_rolling_mean`, `compute_training_climatology`, `compute_seasonal_anomalies`, `fit_min_max_bounds`, `standardize_with_training_bounds`, `compile_production_rzsm_pipeline`, `verify_production_cube_census`, `FastLandAwareRemapper`, `process_era5_land_monthly_pair`, `process_era5_land_antecedent_file`, `compile_full_11yr_rzsm_cube`, and `ProductionCubeConfig`. | `[ACTIVE]` |
| [`src/data/rzsm.py`](../src/data/rzsm.py) | RZSM Calculation & Remapping | Implements depth-weighted RZSM formula ($0.07\cdot\text{SM}_1 + 0.21\cdot\text{SM}_2 + 0.72\cdot\text{SM}_3$). Remapping order: 1) Isolate finite land points on native ERA5-Land ($0.10^\circ$), 2) Bilinear interpolation over active evaluation cells ($M_{i,j}=1$), 3) Nearest-neighbor extrapolation fallback triggered specifically for unassigned coastal boundary cells to prevent coastline clipping, 4) Binary evaluation mask application, and 5) Zero-filling of 1,410 inactive computational cells. | `[PASS]` / `[ACCEPTED]` |
| [`src/data/temporal.py`](../src/data/temporal.py) | Temporal Preprocessing | Implements 7-day backward trailing rolling mean (`center=False`), locked 3-month seasonal climatology (DJF, MAM, JJA, SON) fitted on training years $\le 2021$, seasonal anomalies, and domain-wide active scalar normalization. | `[PASS]` / `[VERIFIED]` |
| [`src/data/compile_cube.py`](../src/data/compile_cube.py) | Production Cube Engine | Production compilation engine with vectorized multi-day spatial remapper (`FastLandAwareRemapper`), monthly pair ingestion (`process_era5_land_monthly_pair`), antecedent support processor (`process_era5_land_antecedent_file`), and end-to-end multi-year compilation (`compile_full_11yr_rzsm_cube`). Orchestrates temporal transformations on full 4,038-day archive series, slicing nominal 4,018-day period (506,268 evaluation cell-days), and executing automated census certification (zero NaNs/Infs). | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |

---

### 3.8 `tests/` (Automated Unit Test Suite)

All 27 automated unit tests are executed with `python -m unittest discover -s tests -v`. **All 27 tests passed with zero failures and zero errors (10.3s total runtime):**

| File Link | Test Count | Key Test Assertions | Execution Time / Status |
| :--- | :---: | :--- | :---: |
| [`test_compile_cube.py`](../tests/test_compile_cube.py) | 5 | 1. End-to-end multi-year pipeline execution and census certification.<br>2. Out-of-sample temporal leakage isolation ($\ge 2022$ cannot alter training bounds).<br>3. `FastLandAwareRemapper` exact numerical bitwise parity ($10^{-6}$, max diff 0.0) with reference remapper.<br>4. 2014 antecedent support file real data processing (20 days, finite on 126 cells, 0 on 1410 buffer).<br>5. Partial archive compilation pipeline orchestration. | 4.8s / `[PASS]` |
| [`test_rzsm.py`](../tests/test_rzsm.py) | 10 | 1. Layer weights sum to $1.0$.<br>2. Constant field preserves soil moisture.<br>3. Hand-computable analytical solution ($2.65$).<br>4. Non-unit weight exception.<br>5. Land mask application and ocean zero-filling.<br>6. Min-max scaling to $[0, 1]$.<br>7. Backward rolling mean mechanics.<br>8. Antecedent lag extraction ($[-1, -7, -14]$).<br>9. Real pilot NetCDF physical range $[0.10, 0.60]\,\text{m}^3/\text{m}^3$.<br>10. Land-aware bilinear remapping with zero NaNs across evaluation cells. | 1.1s / `[PASS]` |
| [`test_target_reconciliation.py`](../tests/test_target_reconciliation.py) | 3 | 1. Hand-calculated arithmetic verification.<br>2. Exact parity with parent formula $L = (\text{lead} \times 7) - 1$ on anchor dates (`2015-01-15`, `2015-06-01`, `2016-02-29`, `2018-08-15`, `2020-12-01`).<br>3. Mathematical proof of contiguous, non-overlapping 28-day partition across $W_1$–$W_4$. | 0.8s / `[PASS]` / `[VERIFIED]` |
| [`test_temporal.py`](../tests/test_temporal.py) | 9 | 1. Rolling mean future invariance (zero leakage from $t+1$).<br>2. Trailing window arithmetic values.<br>3. Climatology strictly ingests years $\le 2021$.<br>4. 3-month seasonal climatology conforms to DJF, MAM, JJA, SON.<br>5. Training period mean anomaly equals zero ($\pm 10^{-6}$).<br>6. Domain-wide active scalar normalization bounds.<br>7. Exact lag and lead date extraction.<br>8. Symmetric continuous series extraction for targets and antecedents.<br>9. Integration test on real 31-day pilot NetCDF data. | 2.0s / `[PASS]` / `[VERIFIED]` |

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
| [`processed/rzsm/pilot/README.md`](../processed/rzsm/pilot/README.md) | Technical data dictionary documenting pilot RZSM coordinate arrays, variable names, and units | Markdown Document |

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

