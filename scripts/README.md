<!-- markdownlint-disable -->
# Operational CLI Scripts & Dependency Guide

This directory contains the production command-line scripts for the **Mindanao Tropical RISE-UNet Adaptation (Track B)**.

The numeric prefix is a stable **catalog identifier**, matching the repository's numbered notebooks and audits. It is not a promise that scripts `01` through `13` can always be run as one uninterrupted sequence: some scripts are optional, some inputs are pre-packaged, and the ERA5 and S2S acquisition branches can run independently. Follow the dependency workflows below rather than guessing from a historical phase or step identifier.

---

## 1. Choose the Right Starting Point

### Verify a fresh clone (recommended first action)

Do not run data downloaders or training scripts just to check that the clone is healthy. First run the hermetic unit suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The numbered test modules are arranged as a verification ladder, but are intentionally order-independent. See [`tests/README.md`](../tests/README.md).

### Use the committed/pre-packaged artifacts

The repository already contains the artifacts needed for the certification and smoke-test workflows. Start with the test suite above, then use only the diagnostic or training script that matches the question being investigated. Scripts `11`–`13` require GPU-capable TensorFlow and are not normal setup steps.

### Rebuild data artifacts from raw inputs

Use this dependency graph when deliberately regenerating artifacts. A node is runnable only after all of the nodes to its left are available.

```text
01_generate_mindanao_masks.py ───────────────> frozen grid and evaluation mask

06_build_production_case_calendar.py ────────> 04_download_all_s2s_production.py
02_download_era5_atmospheric.py ─────────────> 03_derive_era5_atmospheric_daily.py
raw ERA5-Land archive ───────────────────────> 05_verify_production_cube_preflight.py
06_build_production_case_calendar.py ────────> 07_generate_dataset_splits.py
production RZSM cube + splits ───────────────> 08_derive_training_normalization.py
grid + RZSM + atmospheric + S2S + contract ─> 09_build_pilot_manifest_and_cases.py
                                                   └> 10_verify_pilot_ladder_provenance_and_census.py
optional GPU diagnostics ────────────────────> 11_profile, 12_tiny_overfit, 13_pipeline
```
> [!NOTE]
> **Pre-Packaged Repository State**:
> In this repository, **the spatial, calendar, partition, normalization, and pilot-case outputs are already pre-computed and committed** in [`processed/`](../processed/), [`contracts/`](../contracts/), and [`manifests/`](../manifests/). 
> Users who wish to verify model functionality immediately can run the automated test suite (`python -m unittest discover -s tests -p "test_*.py"`) or select the diagnostic that matches their purpose.

---

## 2. Detailed Execution Guide by Stage

### Stage 1: Spatial Grid & Geometry Foundation
- **[`01_generate_mindanao_masks.py`](01_generate_mindanao_masks.py)**
  - **Purpose**: Intersects the Candidate A bounding box ($[4.00^\circ\text{N}, 11.75^\circ\text{N}] \times [116.00^\circ\text{E}, 127.75^\circ\text{E}]$, shape $32 \times 48$) with GADM Level 0/1 administrative boundaries to generate the official binary evaluation mask (126 active land cells) and fractional coverage grids.
  - **Inputs**: `raw/boundary/gadm41_PHL.gpkg`
  - **Outputs**: `processed/grid/mindanao_025deg.nc`, `processed/grid/mindanao_eval_mask_025.nc`, `processed/grid/mindanao_fraction_025.nc`
  - **Command**:
    ```bash
    python scripts/01_generate_mindanao_masks.py
    ```

---

### Stage 2: Raw Reanalysis & S2S Forecast Acquisition
- **[`02_download_era5_atmospheric.py`](02_download_era5_atmospheric.py)** / **[`run_download_atmospheric.sh`](run_download_atmospheric.sh)**
  - **Purpose**: Operational downloader for 11 years (2015–2025) of monthly ERA5 surface and pressure-level reanalysis files from Copernicus Climate Data Store (CDS).
  - **Outputs**: Raw monthly NetCDF files under `raw/era5_atmospheric/` and cloud lake synchronization.
  - **Command**:
    ```bash
    python scripts/02_download_era5_atmospheric.py --start-year 2015 --end-year 2025
    ```

- **[`04_download_all_s2s_production.py`](04_download_all_s2s_production.py)**
  - **Purpose**: Batch downloader and validator for ECMWF CY48R1 subseasonal reforecasts across all 1,154 production cycles, retrieving GRIB2 triplets (`t2m, d2m, tcw`) directly to the GCS lake.
  - **Prerequisites**: Requires `manifests/production_case_calendar.csv` (from Stage 4).
  - **Command**:
    ```bash
    python scripts/04_download_all_s2s_production.py --manifest manifests/production_case_calendar.csv
    ```

---

### Stage 3: Observation Harmonization & Preflight Verification
- **[`03_derive_era5_atmospheric_daily.py`](03_derive_era5_atmospheric_daily.py)**
  - **Purpose**: Ingests raw hourly ERA5 NetCDFs, computes daily diurnal aggregates (mean $T_{2m}$, $T_{max}$, $D_{2m}$, total precipitation $P$, and $200\,\text{hPa}$ geopotential height $Z_{200}$), and crops to the Mindanao Candidate A grid.
  - **Outputs**: `processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc`
  - **Command**:
    ```bash
    python scripts/03_derive_era5_atmospheric_daily.py
    ```

- **[`05_verify_production_cube_preflight.py`](05_verify_production_cube_preflight.py)**
  - **Purpose**: 11-point deterministic preflight verification gate certifying raw input file inventory (265 NetCDFs), hourly timestamp continuity (96,912 hours), leap day hygiene, and frozen spatial contract hashes before full-scale target compilation.
  - **Command**:
    ```bash
    python scripts/05_verify_production_cube_preflight.py
    ```

---

### Stage 4: Operational Calendar & Dataset Partitions
- **[`06_build_production_case_calendar.py`](06_build_production_case_calendar.py)**
  - **Purpose**: Generates the authoritative 1,154-cycle forecast-origin calendar matching the ECMWF CY48R1 reference operational schedule (bi-weekly Monday/Thursday cycles from 2015 to 2025).
  - **Outputs**: `manifests/production_case_calendar.csv`
  - **Command**:
    ```bash
    python scripts/06_build_production_case_calendar.py --output manifests/production_case_calendar.csv
    ```

- **[`07_generate_dataset_splits.py`](07_generate_dataset_splits.py)**
  - **Purpose**: Partitions the 1,154 cycles into Train (735 cycles, 2015–2021), Validation (210 cycles, 2022–2023), and Sealed Test (209 cycles, 2024–2025) with strict boundary isolation.
  - **Outputs**: `manifests/dataset_splits.json`
  - **Command**:
    ```bash
    python scripts/07_generate_dataset_splits.py
    ```

---

### Stage 5: Calibration & Normalization Parameters
- **[`08_derive_training_normalization.py`](08_derive_training_normalization.py)**
  - **Purpose**: Computes channel-wise minimum and maximum scalar parameters strictly across the 735 training cycles and active land evaluation cells, freezing them into the production contract.
  - **Outputs**: `contracts/A0/normalization_parameters.yaml`
  - **Command**:
    ```bash
    python scripts/08_derive_training_normalization.py
    ```

---

### Stage 6: Case Assembly & Provenance Certification
- **[`09_build_pilot_manifest_and_cases.py`](09_build_pilot_manifest_and_cases.py)**
  - **Purpose**: Assembles the 8 reference pilot cases (`CASE_20150115_W01.npz` through `CASE_20150304_W08.npz`) combining antecedent RZSM lags, atmospheric observations, and S2S forecast channels into normalized tensors.
  - **Outputs**: `processed/pilot_cases/CASE_*.npz`, `manifests/pilot_cases_manifest.csv`
  - **Command**:
    ```bash
    python scripts/09_build_pilot_manifest_and_cases.py
    ```

- **[`10_verify_pilot_ladder_provenance_and_census.py`](10_verify_pilot_ladder_provenance_and_census.py)**
  - **Purpose**: Validates bit-for-bit array integrity, SHA-256 hashes, zero NaN/Inf contamination, and temporal alignment of antecedent lags ($t_0-1\text{d}, t_0-7\text{d}, t_0-14\text{d}$) and rolling targets across all 8 pilot cases.
  - **Command**:
    ```bash
    python scripts/10_verify_pilot_ladder_provenance_and_census.py
    ```

---

### Stage 7: Hardware Profiling, Overfitting & Training Loop Verification
- **[`11_profile_a0_vram_benchmark.py`](11_profile_a0_vram_benchmark.py)**
  - **Purpose**: Profiles GPU peak VRAM and step latency across batch sizes $B \in \{11, 22, 33, 44, 55, 66\}$ on Model A0 to determine optimal hardware allocation.
  - **Command**:
    ```bash
    python scripts/11_profile_a0_vram_benchmark.py --device cuda:0
    ```

- **[`12_train_a0_tiny_overfit.py`](12_train_a0_tiny_overfit.py)**
  - **Purpose**: Validates backpropagation, multi-head loss computation, and optimizer convergence by confirming that Model A0 overfits a single batch to near-zero CRPS loss over 40 epochs.
  - **Command**:
    ```bash
    python scripts/12_train_a0_tiny_overfit.py --epochs 40
    ```

- **[`13_train_a0_pipeline_checkpoint.py`](13_train_a0_pipeline_checkpoint.py)**
  - **Purpose**: Verifies multi-epoch training mechanics, dynamic recursive prediction caching, and exact bit-for-bit checkpoint save/restore parity.
  - **Command**:
    ```bash
    python scripts/13_train_a0_pipeline_checkpoint.py --epochs 5 --batch-size 11
    ```

---

## 3. Script Catalog Quick Reference

| ID | Script | Run when | Prerequisite |
| :--: | :--- | :--- | :--- |
| 01 | [`01_generate_mindanao_masks.py`](01_generate_mindanao_masks.py) | Rebuilding the spatial foundation | Authoritative boundary GeoPackage |
| 02 | [`02_download_era5_atmospheric.py`](02_download_era5_atmospheric.py) | Acquiring atmospheric inputs | CDS or ARCO access |
| 03 | [`03_derive_era5_atmospheric_daily.py`](03_derive_era5_atmospheric_daily.py) | Deriving daily atmospheric predictors | Raw atmospheric files |
| 04 | [`04_download_all_s2s_production.py`](04_download_all_s2s_production.py) | Acquiring S2S inputs | `06` case calendar and CDS access |
| 05 | [`05_verify_production_cube_preflight.py`](05_verify_production_cube_preflight.py) | Before replacing the RZSM production cube | ERA5-Land archive and spatial contract |
| 06 | [`06_build_production_case_calendar.py`](06_build_production_case_calendar.py) | Creating/verifying the case calendar | None |
| 07 | [`07_generate_dataset_splits.py`](07_generate_dataset_splits.py) | Rebuilding data partitions | `06` case calendar |
| 08 | [`08_derive_training_normalization.py`](08_derive_training_normalization.py) | Rebuilding normalization bounds | `07` splits and RZSM cube |
| 09 | [`09_build_pilot_manifest_and_cases.py`](09_build_pilot_manifest_and_cases.py) | Assembling pilot tensors | Grid, RZSM, atmospheric, S2S, normalization |
| 10 | [`10_verify_pilot_ladder_provenance_and_census.py`](10_verify_pilot_ladder_provenance_and_census.py) | Certifying pilot outputs | `09` pilot cases |
| 11–13 | GPU diagnostic scripts | Diagnosing prepared model pipelines | GPU-capable TensorFlow and prepared cases |