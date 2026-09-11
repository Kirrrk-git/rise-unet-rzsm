<!-- markdownlint-disable -->
# Step 21D.4-PREFLIGHT: Production RZSM Data Cube Preflight Verification Report

**Domain**: ERA5-Land Archive Preflight Verification & Contract Consistency Gate  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Parent Study**: Lesinger & Tian (2025), *Nature Communications*, DOI: [`10.1038/s41467-025-62761-3`](https://doi.org/10.1038/s41467-025-62761-3)  
**Target Model**: **Mindanao Model A0** (Adapted from EX29 Recursive Hybrid RISE-UNet Baseline)  
**Milestone**: Sub-Phase 21D (Step 21D.4-PREFLIGHT Preflight Verification Gate)  
**Verification Status**: **`[PASS]` / `[VERIFIED]` / `[ACCEPTED]` — PREFLIGHT GATE CERTIFIED PASS**  
**Execution Script**: [`scripts/verify_step_21d4_preflight.py`](../scripts/verify_step_21d4_preflight.py)  
**Date of Audit**: 2026-09-12  

---

## 1. Executive Summary & Preflight Verdict

Before executing the multi-year production compilation of the 11-year Mindanao Root-Zone Soil Moisture (RZSM) spatio-temporal data cube (`era5_land_rzsm_production_2015_2025.nc`), a formal **Preflight Verification Gate (`Step 21D.4-PREFLIGHT`)** was conducted.

This deterministic preflight gate audits all 11 prerequisite computational, contractual, temporal, and spatial specifications.

```text
================================================================================
STEP 21D.4-PREFLIGHT: PRODUCTION CUBE VERIFICATION GATE
================================================================================
[Check 1 & 3] GCS Production Archive Census (265 files, swvl1-3) ........... PASS
[Check 2, 4, 5] Temporal Continuity, Hourly Timestamps (96,912 hrs) ........ PASS
[Check 6 & 7] Spatial Contract Consistency & SHA-256 Hashes ................ PASS
[Check 8, 9, 10, 11] Pipeline Configuration & Leakage Isolation ............ PASS
================================================================================
PREFLIGHT VERIFICATION STATUS: PASSED [CLEARED FOR PRODUCTION COMPILATION]
================================================================================
```

$$\Large\boxed{\textbf{PREFLIGHT VERDICT: STEP 21D.4-PREFLIGHT CERTIFIED PASS}}$$
*(All 11 prerequisites pass without discrepancies; the production compilation pipeline is certified ready for execution).*

---

## 2. 11-Point Preflight Verification Audit Matrix

| # | Preflight Dimension | Target Specification / Criterion | Observed / Verified Status | Certification Tier |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Input File Census** | Exactly 265 NetCDF files in `gs://rise-unet-rzsm/raw/era5_land/production/` | 132 monthly pairs (264 files) + 1 antecedent NetCDF = 265 files total (3.68 GiB) | `[PASS]` |
| **2** | **Archive Temporal Coverage** | 12 Dec 2014 to 31 Dec 2025 ($4,038$ continuous calendar days) | 4,038 calendar days continuous (0 missing days) | `[PASS]` |
| **3** | **Variable Completeness** | Volumetric soil water layers 1, 2, and 3 (`swvl1`, `swvl2`, `swvl3`) present | Verified across all monthly pairs and the 2014 antecedent support file | `[VERIFIED]` |
| **4** | **Hourly Timestamp Continuity** | Expected hourly records for every calendar day; 24 hourly timestamps per day; leap-day coverage intact | **$4,038 \times 24 = 96,912$ hourly timestamps** across support period; 29-day leap Februaries (2016, 2020, 2024 = 696 hrs each) intact | `[PASS]` |
| **5** | **Timestamp Hygiene** | Monotonic time coordinate with zero duplicate or misordered timestamps | Strictly monotonic datetime index; 0 duplicate timestamps | `[PASS]` |
| **6** | **Grid Geometry Compatibility** | Strict conformity to Candidate A cell centers (lat $11.75^\circ \to 4.00^\circ\text{N}$, lon $116.00^\circ \to 127.75^\circ\text{E}$, shape $32 \times 48$) | $32 \times 48$ Candidate A coordinates verified to machine precision ($\Delta = 0.0^\circ$) | `[ACCEPTED]` |
| **7** | **Spatial Contract Consistency** | Frozen contract values, grid coordinates, dimensions ($32 \times 48$), active evaluation cells (126), and referenced artifact hashes match | All 5 spatial foundation artifacts strictly match authoritative contract SHA-256 checksums | `[VERIFIED]` |
| **8** | **Training Period Isolation** | Nominal training fold strictly locked to 2015–2021 (7 full calendar years) | Config locked: `train_start_year=2015`, `train_end_year=2021` | `[PASS]` |
| **9** | **Normalization Scope** | Domain-wide scalar min-max bounds fitted strictly over active evaluation cells within 2015–2021 | 4-part scope enforced: training fold only, active cells only, domain-wide active scalar | `[VERIFIED]` |
| **10** | **Antecedent Support Isolation**| 2014 data (20 days) restricted to rolling memory initialization; never enters training statistics | Rolling mean calculated on full archive, then nominal period sliced (2015-01-01 to 2025-12-31) | `[ACCEPTED]` |
| **11** | **Output CF-1.8 Specification** | Variable names, coordinate conventions, and global metadata attributes frozen | NetCDF CF-1.8 compliant schema specified and tested | `[PASS]` |

---

## 3. Detailed Checkpoint Verifications

### 3.1 Checkpoint A: Hourly Timestamp Continuity & Temporal Accounting
* **Archive Support Window**: 12 December 2014 to 31 December 2025 = **$4,038$ calendar days**.
* **Total Expected Hourly Timestamps**:
  $$N_{\text{hours}} = 4,038 \times 24 = 96,912 \text{ hourly records}$$
* **Nominal Production Timeline**: 01 January 2015 to 31 December 2025 = **$4,018$ calendar days** ($4,018 \times 126 = 506,268$ nominal evaluation cell-days).
* **Leap Year Accounting**:
  - 2016 (Leap): 366 days ($29\text{ Feb} = 24\text{ hrs}$; Feb total = $696\text{ hrs}$)
  - 2020 (Leap): 366 days ($29\text{ Feb} = 24\text{ hrs}$; Feb total = $696\text{ hrs}$)
  - 2024 (Leap): 366 days ($29\text{ Feb} = 24\text{ hrs}$; Feb total = $696\text{ hrs}$)
  - Non-leap years (8 years $\times$ 365 days): 2,920 days.
  - Total nominal days: $8 \times 365 + 3 \times 366 = 2,920 + 1,098 = 4,018$ calendar days.
* **Antecedent Support Window**: 12 December 2014 to 31 December 2014 = **$20$ calendar days** ($20 \times 24 = 480\text{ hours}$).

### 3.2 Checkpoint B: Spatial Contract Consistency & SHA-256 Hash Verification
The deterministic preflight script computed the SHA-256 hashes of all referenced spatial foundation files and matched them against [`contracts/spatial/spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml):

| Artifact File | Expected Contract SHA-256 | Observed SHA-256 | Verification Result |
| :--- | :--- | :--- | :---: |
| `processed/grid/mindanao_025deg.nc` | `51a691994a16b75b3a103efdb539febce2695179dbb89f5cb346a477e694548f` | `51a691994a16b75b3a103efdb539febce2695179dbb89f5cb346a477e694548f` | ✅ **EXACT MATCH** |
| `processed/grid/mindanao_eval_mask_025.nc` | `d7fd80e1f95cdd840daded176b06b2a0d23557df555acfaf0c01d2d1964621f6` | `d7fd80e1f95cdd840daded176b06b2a0d23557df555acfaf0c01d2d1964621f6` | ✅ **EXACT MATCH** |
| `processed/grid/mindanao_fraction_025.nc` | `9294a1c74ec5aa2288c9e9bf54fc6763f905dbda1c4ff0ecda0a9b9ae9ff6325` | `9294a1c74ec5aa2288c9e9bf54fc6763f905dbda1c4ff0ecda0a9b9ae9ff6325` | ✅ **EXACT MATCH** |
| `processed/boundary/mindanao_analysis_boundary.gpkg` | `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2` | `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2` | ✅ **EXACT MATCH** |
| `processed/grid/mindanao_0.25_grid.grd` | `b8fabe5c21a6b5beda2df8b9ef4438d8b63d74427e86223137a55c3afff40049` | `b8fabe5c21a6b5beda2df8b9ef4438d8b63d74427e86223137a55c3afff40049` | ✅ **EXACT MATCH** |

* **Active Evaluation Cells**: Strictly 126 cells ($86,418.83\text{ km}^2$, 86.46% of regional land area).
* **Buffer / Ocean Cells**: Strictly 1,410 cells (zero-filled).
* **Total Grid Cells**: $32 \times 48 = 1,536$ cells.

### 3.3 Checkpoint C: Pipeline Configuration & Leakage Isolation
Verified via `ProductionCubeConfig` in [`src/data/compile_cube.py`](../src/data/compile_cube.py):
* `train_start_year`: 2015
* `train_end_year`: 2021
* `val_years`: `(2022, 2023)`
* `test_years`: `(2024, 2025)`
* `climatology_method`: `"season"` (Model A0 baseline)
* `rolling_window`: 7 days (`center=False`)
* Out-of-sample temporal leakage test `test_out_of_sample_leakage_isolation` confirmed that data from 2022 onwards causes strictly zero alteration to training climatology or normalization bounds.

---

## 4. Operational Clearance Verdict

With all 11 preflight checks certified as `[PASS]`, `[VERIFIED]`, or `[ACCEPTED]`:
1. **Interactive Colab Execution Asset Prepared**: [`notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb`](../notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb) (16 cells with official Colab badge).
2. **Deterministic Script Ready**: [`scripts/verify_step_21d4_preflight.py`](../scripts/verify_step_21d4_preflight.py).
3. **Step 21D.4 Execution Status**: **CLEARED TO PROCEED**.
