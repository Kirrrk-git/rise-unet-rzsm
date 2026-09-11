<!-- markdownlint-disable -->
# ERA5 Atmospheric Pilot & RZSM Pipeline Preprocessing Audit Report

**Domain**: ERA5 Atmospheric Reanalysis Acquisition, 5-Channel Derivation & RZSM Layer Preprocessing  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Milestones**: Sub-Phase 21C (Steps 21C.1 & 21C.2) & Sub-Phase 21D (Steps 21D.1 & 21D.2)  
**Authoritative Scripts**:  
- Acquisition Engine: [`scripts/download_era5_atmospheric_mindanao.py`](../scripts/download_era5_atmospheric_mindanao.py)  
- Verification Suite: [`scripts/verify_and_derive_era5_pilot.py`](../scripts/verify_and_derive_era5_pilot.py)  
- Production Module: [`src/data/rzsm.py`](../src/data/rzsm.py)  
- Unit Test Suite: [`tests/test_rzsm.py`](../tests/test_rzsm.py)  
**Status**: **EMPIRICALLY VERIFIED & CERTIFIED PASS**  
**Date**: 2026-09-11  

---

## 1. Executive Summary & Objective

Following the empirical completion of **Sub-Phase 21B** (ERA5-Land soil moisture remapping and antecedent window synchronization), this audit report formalizes the mathematical, computational, and physical verification for:

1. **Step 21C.1**: 1-month benchmark pilot retrieval of raw ERA5 atmospheric reanalysis variables (January 2015, 744 hours) from the Copernicus Climate Data Store (CDS API).
2. **Step 21C.2**: Mathematical derivation of the 5 parent-compatible RISE-UNet atmospheric predictor channels (`tmax`, `diff_temp`, `spfh`, `pwat`, `hgt_pres`), certification of $(31, 32, 48)$ tensor geometry, and assertion of **0 NaNs** across the 126 Mindanao evaluation land cells.
3. **Step 21D.1**: Modular implementation of layer-integrated Root-Zone Soil Moisture (RZSM 0–100 cm), 7-day trailing rolling mean, antecedent lag slicing (`[-1, -7, -14]` days), and min-max standardization in [`src/data/rzsm.py`](../src/data/rzsm.py).
4. **Step 21D.2**: Execution of automated unit testing in [`tests/test_rzsm.py`](../tests/test_rzsm.py), validating analytical formulas to machine precision ($10^{-7}$).

---

## 2. Mathematical Formulation & Channel Lineage

In strict accordance with the verified parent EX29 contract (Lesinger & Tian 2025; `function/addPredictors.py:L61-77` and `00_min_max_each_region_&reforecast.ipynb:L383-402`), five atmospheric predictor channels are derived from raw ECMWF reanalysis fields:

### 2.1 Daily Maximum 2m Temperature (`tmax`)
Extracted from analyzed hourly 2m temperature ($T_{\text{2m}}$, short name `2t` / `t2m`):
$$\text{tmax}_d = \max_{t \in \text{day } d} T_{\text{2m}}(t)$$

### 2.2 Diurnal Temperature Range (`diff_temp`)
Computed as the daily difference between maximum and minimum analyzed hourly temperature:
$$\text{tmin}_d = \min_{t \in \text{day } d} T_{\text{2m}}(t)$$
$$\text{diff\_temp}_d = \text{tmax}_d - \text{tmin}_d$$

### 2.3 Surface Specific Humidity (`spfh`) via Bolton (1980)
Calculated from hourly analyzed 2m dewpoint temperature ($T_d$ in °C) and surface barometric pressure ($p$ in Pa) using the Bolton (1980) / Tetens vapor pressure formulation:
$$e(t) = 611.2 \cdot \exp\left( \frac{17.67 \cdot T_d(t)}{T_d(t) + 243.5} \right)$$
$$q(t) = \frac{\epsilon \cdot e(t)}{p(t) - (1 - \epsilon) \cdot e(t)}, \quad \epsilon = 0.622$$
$$\text{spfh}_d = \frac{1}{24} \sum_{t=0}^{23} q(t) \quad [\text{kg/kg}]$$

### 2.4 Precipitable Water (`pwat`)
Calculated directly as the daily arithmetic mean of ECMWF Total Column Water Vapour (`tcwv`):
$$\text{pwat}_d = \frac{1}{24} \sum_{t=0}^{23} \text{tcwv}(t) \quad [\text{kg/m}^2]$$
*(Note: Total precipitation was deliberately excluded by Lesinger & Tian (2025); `pwat` is the authoritative moisture driver).*

### 2.5 Upper-Tropospheric Geopotential Height (`hgt_pres`)
Derived from ECMWF analyzed geopotential ($z$ in $\text{m}^2/\text{s}^2$) at the 200 hPa isobaric level, converted to geopotential meters (gpm) via WMO standard gravity ($g_0 = 9.80665\text{ m/s}^2$):
$$H_{200}(t) = \frac{z_{200}(t)}{9.80665}$$
$$\text{hgt\_pres}_d = \frac{1}{24} \sum_{t=0}^{23} H_{200}(t) \quad [\text{gpm}]$$

### 2.6 Depth-Weighted Root-Zone Soil Moisture (`RZSM_0_100`)
Top 100 cm volumetric soil water integration across ECMWF soil layers 1, 2, and 3:
$$\text{RZSM}_{0-100} = 0.07 \cdot \text{swvl}_1 + 0.21 \cdot \text{swvl}_2 + 0.72 \cdot \text{swvl}_3$$

---

## 3. Spatial Domain Parity & Candidate A Resolution

| Parameter | Specification | Parity Status |
| :--- | :--- | :---: |
| **Grid Framework** | Frozen Candidate A Mindanao Grid ($0.25^\circ \times 0.25^\circ$) | Exact Match |
| **Bounding Box** | `[North: 11.75°N, West: 116.00°E, South: 4.00°N, East: 127.75°E]` | Exact Match |
| **Latitude Vector** | 32 points: $11.75^\circ, 11.50^\circ, \dots, 4.00^\circ$ (descending) | Exact Match |
| **Longitude Vector** | 48 points: $116.00^\circ, 116.25^\circ, \dots, 127.75^\circ$ (ascending) | Exact Match |
| **Spatial Dimensions** | $32 \times 48 = 1,536$ total grid cells | Exact Match |
| **Active Evaluation Cells** | **126 land cells** ($f \ge 0.50$ coverage over Mindanao mainland) | Exact Match |
| **Zero-Padded Buffer** | **1,410 cells** (ocean, Visayas, Palawan, marine boundary) | Exact Match |

---

## 4. Empirical Verification Results (January 2015 Pilot)

### 4.1 Raw Retrieval Audit
- Single Levels File: `Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_2015_01.nc` (7,276,966 bytes = 7.28 MB)
- Pressure Levels File: `Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_2015_01.nc` (1,363,222 bytes = 1.36 MB)
- Timesteps: 744 hours ($31\text{ days} \times 24\text{ hours/day}$) — **100% complete, zero missing timesteps**.

### 4.2 Derived Channel Physical Distribution Census (Active 126 Cells)
Sample count: $31\text{ days} \times 126\text{ cells} = 3,906\text{ valid spatio-temporal samples}$.

| Channel | Variable Name | Min | Mean | Max | Std | NaNs | Infs | Units |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `tmax` | 294.28 | 301.65 | 307.36 | 1.97 | **0** | **0** | K |
| **—** | `tmin` (diagnostic) | 288.22 | 295.88 | 299.76 | 1.69 | **0** | **0** | K |
| **2** | `diff_temp` | 0.47 | 5.77 | 11.68 | 1.96 | **0** | **0** | K |
| **3** | `spfh` | 0.0146 | 0.0178 | 0.0204 | 0.0009 | **0** | **0** | kg/kg |
| **4** | `pwat` | 17.23 | 44.80 | 59.94 | 6.61 | **0** | **0** | kg/m² |
| **5** | `hgt_pres` | 12,411.5 | 12,442.8 | 12,481.9 | 16.30 | **0** | **0** | gpm |

**Verdict**: **ZERO NaNs and ZERO Infs** detected across all 3,906 evaluation sample points. All variables conform to physical tropical hydroclimatic bounds.

---

## 5. Automated Unit Test Suite Results (Sub-Phase 21D)

The automated unit test suite across [`tests/test_rzsm.py`](../tests/test_rzsm.py) and [`tests/test_temporal.py`](../tests/test_temporal.py) was executed to validate algorithmic integrity:

### 5.1 RZSM Formulation & Spatial Masking (`tests/test_rzsm.py`)
| Test Case | Component | Verification Focus | Result |
| :--- | :--- | :--- | :---: |
| `test_layer_weights_sum_to_one` | Depth Weighting | $0.07 + 0.21 + 0.72 = 1.0$ | ✅ **PASS** |
| `test_depth_weighted_rzsm_constant_field` | Depth Weighting | Uniform field $\text{SM}=c \implies \text{RZSM}=c$ ($10^{-7}$ precision) | ✅ **PASS** |
| `test_depth_weighted_rzsm_analytical_solution` | Depth Weighting | Exact analytical linear combination ($2.65$) | ✅ **PASS** |
| `test_invalid_weights_raise_error` | Input Validation | Non-unitary weights trigger `ValueError` | ✅ **PASS** |
| `test_backward_rolling_mean_mechanics` | Temporal | Trailing window=7 (`center=False`) step dynamics | ✅ **PASS** |
| `test_extract_antecedent_lags` | Temporal | Relative lag day offsets `[-1, -7, -14]` extraction | ✅ **PASS** |
| `test_apply_land_mask` | Spatial | Evaluation land preservation, dimension order & ocean zero-fill | ✅ **PASS** |
| `test_compute_min_max_scale` | Standardization | Range $[0, 1]$ mapping & inactive cell zero-fill | ✅ **PASS** |
| `test_pilot_rzsm_file_exists_and_passes_physics` | Regression | Real pilot NetCDF shape `(20, 32, 48)` and 0 NaNs | ✅ **PASS** |

### 5.2 Temporal Preprocessing, Climatology & Target Parity (`tests/test_temporal.py`)
| Test Case | Component | Verification Focus | Result |
| :--- | :--- | :--- | :---: |
| `test_trailing_window_values` | Rolling Mean | 7-day backward window values & 6 initial NaNs | ✅ **PASS** |
| `test_future_invariance_no_leakage` | Anti-Leakage | Zero future data leakage under forward perturbation | ✅ **PASS** |
| `test_training_period_filter` | Climatology | Strict isolation to training period ($\le 2021$) | ✅ **PASS** |
| `test_season_climatology_method` | Climatology | Conformance to DJF, MAM, JJA, SON groupings | ✅ **PASS** |
| `test_anomaly_subtraction_zero_mean_training`| Anomalies | Training fold anomaly mean zeroing | ✅ **PASS** |
| `test_exact_lag_and_lead_dates` | Multi-Lead | Date alignment for lags `[-1, -7, -14]` & leads `[7, 14, 21, 28]` | ✅ **PASS** |
| `test_target_parity_formula` | Target Parity | Targets and antecedents share identical transformation math | ✅ **PASS** |
| `test_fit_and_standardize` | Normalization | Training fold bounds fitting, $[0, 1]$ range & ocean zero-fill | ✅ **PASS** |
| `test_real_pilot_temporal_processing` | Pilot Integration | Real pilot NetCDF trailing rolling mean & zero NaNs | ✅ **PASS** |

**Summary**: **18/18 unit tests passed in 2.237 s (0 failures, 0 errors).**

---

## 6. Generated Visual & Data Artifacts

1. **Publication Verification Composite**:
   - Local: [`processed/atmospheric/figures/mindanao_era5_atmospheric_pilot_verification.png`](../processed/atmospheric/figures/mindanao_era5_atmospheric_pilot_verification.png) (390.7 KB, 300 DPI)
   - Six panels displaying monthly mean fields for `tmax`, `diff_temp`, `spfh`, `pwat`, `hgt_pres`, and the active evaluation mask.
2. **Verified Pilot NetCDF**:
   - Local: [`processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc`](../processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc) (1,165.33 KB, CF-1.8 compliant)
   - Shape: `(31, 32, 48)` with all 5 derived predictor channels and diagnostic `tmin`.
3. **Landing Zone Documentation**:
   - Local: [`Data/raw_downloads/ERA5_atmospheric/README.md`](../Data/raw_downloads/ERA5_atmospheric/README.md)
4. **Modular Packages & Automated Test Suites**:
   - Modular Production Engine: [`src/data/rzsm.py`](../src/data/rzsm.py) and [`src/data/temporal.py`](../src/data/temporal.py)
   - Comprehensive Unit Tests: [`tests/test_rzsm.py`](../tests/test_rzsm.py) and [`tests/test_temporal.py`](../tests/test_temporal.py)

---

## 7. Certification & Next Steps

Sub-Phases **21C** and **21D (Steps 21D.1, 21D.2, & 21D.3)** are formally **CERTIFIED PASS**.

- **Active Background Task**: Full multi-year raw atmospheric archive download (2015–2025) running via `download_era5_atmospheric_mindanao.py` in PowerShell terminal.
- **Next Milestone**: Step 21D.4 (Full-Scale 11-Year Data Cube Compilation & GCS Export).

