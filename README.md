<!-- markdownlint-disable -->
# Tropical RISE-UNet Adaptation: Subseasonal Root-Zone Soil Moisture Forecasting over Mindanao, Philippines

[![Scientific Certification: 3-Tier Passing](https://img.shields.io/badge/Certification-3--Tier%20Pass-brightgreen.svg)](reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
[![Target Grid: Candidate A (32x48)](https://img.shields.io/badge/Grid-Candidate%20A%20(32%C3%9748)-blue.svg)](contracts/A0/mindanao_a0_production_contract.yaml)
[![Parent Reference: Nature Comms 2025](https://img.shields.io/badge/Parent%20Study-Lesinger%20%26%20Tian%20(2025)-orange.svg)](https://doi.org/10.1038/s41467-025-62761-3)
[![Test Suite: 114 Tests (108 Passed, 6 Skipped)](https://img.shields.io/badge/Tests-108%20Passed%20%7C%206%20Skipped-success.svg)](tests/)

---

## 1. Project Overview & Research Architecture

This repository hosts the **Mindanao Regional Adaptation (Track B)** of the **RISE-UNet (Recursive Integrated Spatiotemporal Evaluation U-Net)** deep learning architecture, originally published by Kyle Lesinger and Di Tian (2025) in *Nature Communications* for the Contiguous United States (CONUS).

The research objective is to adapt and validate deep learning subseasonal root-zone soil moisture (RZSM) forecasts (Weeks 1–4 leads) across the complex, cloud-dense, tropical archipelago domain of **Mindanao, Philippines** ($5.0^\circ\text{N} - 10.0^\circ\text{N}, 121.0^\circ\text{E} - 127.0^\circ\text{E}$).

### Dual-Track Repository Organization

To ensure complete scientific transparency and maintain clean separation of concerns:

- **Track A (Author Baseline)**: The complete, unmodified peer-reviewed CONUS codebase from Kyle Lesinger & Di Tian (2025) is fully preserved in [`parent_study_ex29/`](parent_study_ex29/).
- **Track B (Mindanao Adaptation)**: The regional adaptation, data pipelines, normalization contracts, 1,154-cycle operational calendar, and unit test suites are organized in standard modern scientific package directories (`src/`, `contracts/`, `manifests/`, `notebooks/`, `scripts/`, `tests/`).

```text
dl_dm_rzsm_subseasonal_forecast/
│
├── parent_study_ex29/                       <── TRACK A: Author's Original Peer-Reviewed Study
│   ├── README.md                            <── Original author documentation & CONUS guide
│   ├── notebooks/                           <── All 48 original CONUS notebooks (00_*, 02_*, etc.)
│   ├── function/                            <── Original author Python modules (modelRzsmRelu, losses)
│   ├── Data/                                <── Original author CONUS data scripts, EMOS, masks
│   └── conda_environment_setup.yaml        <── Original conda environment configuration
│
├── contracts/                               <── TRACK B: Frozen Machine-Readable Contracts
│   └── A0/                                  <── Model A0 contract, active normalization YAML, VERIFICATION_STATUS
├── manifests/                               <── Production Case Calendar & Split Partitions
│   ├── production_case_calendar.csv         <── Authoritative 1,154-cycle S2S forecast origin calendar
│   └── splits/                              <── Train (735), Validation (210), Sealed Test (209 scheduled census)
├── notebooks/                               <── Numbered End-to-End Mindanao Execution Pipeline
│   ├── 00_parent_freeze_and_inspection.ipynb
│   ├── 01_parent_experiment_trace.ipynb
│   ├── 02_mindanao_geometry_and_cascade_gate.ipynb
│   ├── 03_mindanao_spatial_foundation.ipynb
│   ├── 04_mindanao_elevation_and_soil_static_channels.ipynb
│   ├── 05_mindanao_atmospheric_and_rzsm_preprocessing.ipynb
│   ├── 06_mindanao_era5_land_daily_climatology_and_anomalies.ipynb
│   ├── 07_mindanao_s2s_and_pilot_case_assembly.ipynb
│   ├── 08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb
│   ├── 09_mindanao_a0_vram_profiling.ipynb
│   └── 10_mindanao_validation_atmospheric_pipeline.ipynb
├── processed/                               <── Verified Regional Data Artifacts
│   ├── grid/                                <── Candidate A grid ($32 \times 48$, 126 active cells)
│   ├── rzsm/                                <── Compiled 11-year RZSM NetCDF cube (2015–2025)
│   ├── atmospheric/                         <── Derived surface atmospheric pilot NetCDFs
│   └── s2s/                                 <── Verified ECMWF S2S reforecast structures
├── reproduction_audit/                      <── Three-Tier Scientific Audit Dossiers & Registry
│   ├── OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md
│   ├── 18_production_case_calendar_1154_cycles_audit.md
│   ├── 19_dataset_splits_and_normalization_contract_audit.md
│   └── ...
├── scripts/                                 <── Production CLI Tools & Preflight Engines
│   ├── 06_build_production_case_calendar.py    <── Generates 1,154-cycle operational calendar
│   ├── 07_generate_dataset_splits.py           <── Generates partitioned case manifests
│   ├── 08_derive_training_normalization.py     <── Fits active-domain scalar min-max bounds
│   ├── 11_profile_a0_vram_benchmark.py         <── Six-pillar fail-closed hardware benchmark
│   ├── 14_run_a0_production_smoke_test.py      <── Genuine 4-lead production preflight
│   └── 15_verify_validation_atmospheric_pipeline.py <── Validation atmospheric pipeline verification
├── src/                                     <── Production Python Package
│   ├── data/                                <── case_builder, cloud_lake, tf_dataset, atmospheric, rzsm
│   ├── models/                              <── genuine a0_unet factory, parameter counts
│   └── utils/                               <── Geospatial and metric utilities
├── tests/                                   <── Automated Unit Testing Framework (96 tests)
├── figures/                                 <── Publication-Grade Composite Verification Figures
├── AGENTS.md                                <── AI engineering safety boundaries & guidelines
└── README.md                                <── This Master Portal
```

---

## 2. Key Scientific & Architectural Invariants

1. **Frozen Spatial Domain (Candidate A)**:
   - Extent: $[5.0^\circ\text{N}, 10.0^\circ\text{N}] \times [121.0^\circ\text{E}, 127.0^\circ\text{E}]$
   - Grid Resolution: $0.25^\circ \times 0.25^\circ$ ($21 \times 25$ geographic bounding box)
   - UNet Dimensions: $32 \times 48$ with zero-padding (divisible by $2^4 = 16$ across 4 downsampling stages)
   - Active Evaluation Cells: Exactly **126 land cells** strictly verified against GADM Level 0/1 polygons.
2. **Forecast-Origin Partitioning with Target-Horizon Boundary Extension**:
   - Total Operational Cycles: **1,154** (reconciled against ECMWF CY48R1 operational reference calendar)
   - Train Split: **735 cycles** ($2015\text{--}2021$)
   - Validation Split: **210 cycles** ($2022\text{--}2023$; Val-A: 105, Val-B: 105)
   - Sealed Test Split: **209 scheduled cycles** cohort census ($2024\text{--}2025$; **202 usable sealed-test cases** forming evaluation denominator, **7 quarantined cases** in late Dec 2025 whose $W_4$ target extends into Jan 2026).
   - Non-Leakage Posture: No forecast-origin overlap and no future predictor information relative to each $t_0$.
3. **Immutable Normalization Contract**:
   - Parameters derived strictly from the 735 training cycles across active land cells only.
   - Actively consumed by [`src/data/case_builder.py`](src/data/case_builder.py) via [`contracts/A0/normalization_parameters.yaml`](contracts/A0/normalization_parameters.yaml).
4. **Authoritative Model A0 UNET_RZSM**:
   - Instantiated via [`src/models/a0_unet.py`](src/models/a0_unet.py), matching Kyle Lesinger's genuine architecture with verified per-lead trainable parameter counts:
     - Lead 1 ($W_1$): 1,627,139 parameters ($C_{in} = 11$)
     - Lead 2 ($W_2$): 1,630,307 parameters ($C_{in} = 12$)
     - Lead 3 ($W_3$): 1,608,131 parameters ($C_{in} = 5$)
     - Lead 4 ($W_4$): 1,611,299 parameters ($C_{in} = 6$)
     across 251 layers, 298 weight tensors, and 3 deep supervision output heads.

---

## 3. Running the Test Suite

All unit tests are automated and execute from the repository root:

```bash
# Run the complete test suite (114 tests: 108 passed, 6 skipped, 0 failed, 0 errors)
python -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Citation & Attribution

### Peer-Reviewed Parent Study

- Lesinger, K., & Tian, D. (2025). Subseasonal root-zone soil moisture forecasting with deep learning. *Nature Communications*, 16, 62761. DOI: [10.1038/s41467-025-62761-3](https://doi.org/10.1038/s41467-025-62761-3).

