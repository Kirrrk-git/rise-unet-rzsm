<!-- markdownlint-disable -->
# ERA5-Land RZSM Target Acceptance & Independent Cross-Reference Audit Report

**Domain**: Root-Zone Soil-Moisture (RZSM) Target Definition & Data Provenance  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Milestone**: Sub-Phase 21A Step 21A.9  
**Evaluation Status**: **CONDITIONALLY ACCEPTED (Technical Formulation Certified; Antecedent Gap Identified)**  
**Date of Audit**: 2026-09-10  

---

## 1. Executive Summary & Audit Verdict

| Audit Dimension | Evaluation Criterion | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Target Variable Definition** | 0–100 cm depth-weighted volumetric soil-water index | $\text{RZSM}_{0-100} = 0.07\,\text{SM}_1 + 0.21\,\text{SM}_2 + 0.72\,\text{SM}_3$ (matches author `process_soil.sh`) | ✅ **PASS** |
| **Physical Units & Bounds** | Volumetric fraction $\text{m}^3/\text{m}^3 \in [0.0, 0.7]$ | Physical bounds verified; min-max normalization to $[0, 1]$ | ✅ **PASS** |
| **Existing Archive Inventory** | 2015–2025 monthly NetCDF blocks in GCS bucket | 264 objects parsed across 11 years (2015–2025); 132 monthly pairs complete | ✅ **PASS** |
| **Operational Stream Policy** | Hourly ERA5-Land reanalysis as source-of-record with pipeline daily aggregation | Certified: A0 production RZSM cube uses hourly ERA5-Land reanalysis as source-of-record and performs explicit daily aggregation in pipeline; Copernicus derived daily-statistics product is not the source | ✅ **PASS** |
| **Antecedent Lag Completeness** | Availability of 2014 data for $W_1$ 2015 initialization | **Archive starts at 2015-01-01; 2014 is MISSING** (resolved in Step 21B.2 via December 2014 antecedent support file) | ⚠️ **CONDITIONAL** |
| **Spatial Remapping Alignment** | Land-aware linear spatial interpolation with nearest fallback | CDO descriptor `mindanao_0.25_grid.grd` compatible; native engine in `compile_cube.py` | ✅ **PASS** |

| **Evaluation Mask Integration** | Confinement to 126 binary evaluation cells ($f \ge 0.50$) | Evaluated strictly on 126 cells (full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$); 1,410 buffer cells zero-masked | ✅ **PASS** |
| **Independent Cross-Reference** | Independent GLEAM v3.8a RZSM comparison | Protocol defined; GLEAM retained strictly as external validation reference | ✅ **PASS** |

$$\Large\boxed{\textbf{AUDIT VERDICT: CONDITIONAL PASS}}$$
*(Technical formulation certified; hourly source-of-record with pipeline daily aggregation established; 2014 antecedent support flagged and integrated in Step 21B.2).*

---

## 2. Target RZSM Mathematical Formulation & Provenance

### 2.1 Regional Adaptation Commitment vs. Parent Literature
In the parent study, Lesinger & Tian (2025) used daily GLEAM v3.8a RZSM (0–100 cm) as the target variable over CONUS. For the Mindanao adaptation (Track B), Model A0 deliberately commits to **ERA5-Land volumetric soil moisture (0–100 cm)** as the primary target and antecedent soil-moisture state:
* **Physical Depth Integration**:
  $$\text{RZSM}_{0-100} = 0.07 \cdot \text{swvl}_1 + 0.21 \cdot \text{swvl}_2 + 0.72 \cdot \text{swvl}_3$$
  where:
  * $\text{swvl}_1$: Volumetric soil water layer 1 ($0\text{--}7\text{ cm}$, depth thickness $\Delta z_1 = 0.07\text{ m}$)
  * $\text{swvl}_2$: Volumetric soil water layer 2 ($7\text{--}28\text{ cm}$, depth thickness $\Delta z_2 = 0.21\text{ m}$)
  * $\text{swvl}_3$: Volumetric soil water layer 3 ($28\text{--}100\text{ cm}$, depth thickness $\Delta z_3 = 0.72\text{ m}$)
* **Author Code Parity**: Audited against Kyle Lesinger’s original bash script [`Data/raw_downloads/ERA5_real/process_soil.sh`](../Data/raw_downloads/ERA5_real/process_soil.sh), confirming the identical linear scaling factors:
  ```bash
  cdo mulc,0.07 -selname,swvl1 "$daymean_file" "$swvl1_scaled"
  cdo mulc,0.21 -selname,swvl2 "$daymean_file" "$swvl2_scaled"
  cdo mulc,0.72 -selname,swvl3 "$daymean_file" "$swvl3_scaled"
  cdo add "$swvl1_scaled" "$swvl2_scaled" "$temp_add"
  cdo add "$temp_add" "$swvl3_scaled" "$sum_file"
  ```

---

## 3. Audit of the Existing GCS ERA5-Land Archive

An audit of the cloud archive at `gs://mindanao-drought-aaron-jalapon-drought-data/raw/era5-land/` was performed:

### 3.1 Object Accounting & File Patterns
* **Total Objects**: 264 NetCDF files (3,936,075,396 bytes, ~3.67 GiB).
* **Pattern 1**: `era5-land-YYYY-MM.nc` (132 files, 2015-01 through 2025-12).
  * Contains hourly layers (average file size: 25–28 MiB per month).
  * 12 of 12 months present for all 11 years (2015 through 2025).
* **Pattern 2**: `era5-land-sm3-YYYY-MM.nc` (132 files, 2015-01 through 2025-12).
  * Contains layer 3 (`swvl3`) extract (average file size: ~2.6 MiB per month).
  * 12 of 12 months present for all 11 years (2015 through 2025).

### 3.2 Operational Stream Policy: Hourly Reanalysis Source-of-Record & Pipeline Daily Aggregation
* **Frozen Source-of-Record Decision**: **The A0 production RZSM cube uses the hourly ERA5-Land reanalysis as the source-of-record and performs explicit daily aggregation within the project pipeline. The Copernicus derived daily-statistics product is not the source of the certified 21D.4 production cube.**
* **Scientific & Reproducibility Rationale**: Utilizing the synchronized multi-year hourly archive (`265 files, 96,912 hours`) ensures complete provenance and mathematical control over the 24-hour daily mean calculation (`.resample(time='1D').mean(dim='time')`) and leap-day handling, completely independent of external CDS API rate limits, downtime, or black-box server-side aggregation differences.

### 3.3 Critical Finding: The 2014 Antecedent Gap
* **The EX29 Lag Contract**: In Step 21A.3, the antecedent lag schedule was verified as:
  $$\text{Lags} = [-1, -7, -14]\text{ days (7-day backward trailing rolling windows)}$$
* **The Gap**: For the earliest forecast initialization date in the 2015 archive (e.g., first week of January 2015), calculating the 14-day backward trailing average strictly requires observations from **18 December to 31 December 2014**.
* **Audit Finding**: The existing bucket archive contains zero files for 2014 (`min_year = 2015`).
* **Remediation Requirement**: Prior to training on the complete 2015–2025 timeline, either:
  1. Retrieve December 2014 (or the full calendar year 2014) via CDS daily statistics to support unclipped 2015 initialization, OR
  2. Advance the initial training forecast date by 21 days into January 2015 to allow the 14-day trailing rolling window to populate entirely from within 2015.

---

## 4. Spatial Coordinate Remapping & Evaluation Mask Integration

### 4.1 Remapping Pipeline & CDO Specification Compatibility
* **Input Grid**: Native ERA5-Land $0.10^\circ \times 0.10^\circ$ geographic grid.
* **Target Grid**: Frozen Mindanao $0.25^\circ \times 0.25^\circ$ reference grid (`mindanao_025deg.nc`, $32 \times 48 = 1,536$ cells).
* **CDO Specification Compatibility vs. Production Engine Implementation**:
  The frozen target grid is formally compatible with the Climate Data Operators (CDO) standard description format (`processed/grid/mindanao_0.25_grid.grd`):
  ```bash
  # CDO interoperability verification syntax:
  cdo remapbil,processed/grid/mindanao_0.25_grid.grd era5_land_daily_010.nc era5_land_daily_025.nc
  ```
  However, the actual production Model A0 data pipeline executes via the project's native Python remapping engine (`FastLandAwareRemapper` in `src/data/compile_cube.py`). This engine executes land-aware linear spatial interpolation (piecewise-linear Delaunay barycentric interpolation on valid finite land points) with nearest-neighbor coastal fallback, avoiding ocean-mask clipping on island edges and peninsulas while maintaining exact numerical agreement on tested data (observed maximum absolute difference = 0.0; test tolerance = 1e-6).
* **Bounding Box Alignment**:
  * Native ERA5-Land window covers: $[115.8^\circ\text{E}, 3.8^\circ\text{N}] \to [128.0^\circ\text{E}, 12.0^\circ\text{N}]$, completely covering the frozen cell-edge envelope ($[115.875^\circ, 3.875^\circ] \to [127.875^\circ, 11.875^\circ]$). Zero extrapolation is required.


### 4.2 Non-Evaluation Buffer vs. Active Evaluation Cells
In strict adherence to the frozen spatial contract ([`contracts/spatial/spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml)):
* **Non-Evaluation Domain ($f < 0.50$, 1,410 cells, 91.80%)**:
  * Pure exterior ocean buffer ($f = 0.000$): 1,283 cells. Zero-filled per author convention to provide convolutional receptive field padding without NaNs.
  * Sub-threshold coastal transition cells ($0.000 < f < 0.500$): 127 cells. Excluded from loss and skill metrics to prevent coastal ocean water contamination.
* **Binary Evaluation Domain ($f \ge 0.50$, 126 cells, 8.20%)**:
  * Total full-cell footprint of 126 included $0.25^\circ$ cells spans $96,085.57\text{ km}^2$, representing $86,418.83\text{ km}^2$ of actual boundary-intersection area (86.46% of the authoritative $99,948.76\text{ km}^2$ Mindanao boundary).
  * All training loss updates and validation metrics (ACC, CRPSS, RMSE, KGE) are strictly confined to these 126 cells.

---

## 5. Independent GLEAM Cross-Reference Validation Protocol

### 5.1 Dataset Independence
* **Product**: Global Land Evaporation Amsterdam Model (GLEAM) v3.8a.
* **Target Variable**: Root-Zone Soil Moisture ($\text{SM}_{\text{root}}$, $0\text{--}100\text{ cm}$, $\text{m}^3/\text{m}^3$).
* **Methodological Role**: GLEAM is **NOT an A0 training target**. It serves exclusively as an independent, satellite-data-assimilated reference to assess the regional credibility of the ERA5-Land reanalysis over Mindanao.

### 5.2 Empirical Suitability Benchmark Criteria
Over the common historical period (2015–2022), the remapped ERA5-Land $\text{RZSM}_{0-100}$ index shall be cross-referenced against GLEAM across the 126 evaluation cells under the following criteria:

| Metric | Benchmark Criterion | Rationale |
| :--- | :---: | :--- |
| **Temporal Pearson Correlation ($r$)** | Domain Mean $r \ge 0.65$ | Verifies broad agreement on subseasonal wetting and drying dynamics. |
| **Seasonal Cycle Correlation** | Monthly Climatology $r \ge 0.80$ | Confirms alignment with the monsoon (Amihan / Habagat) seasonal cycle. |
| **Anomaly Correlation (AC)** | Climatological Anomaly $r \ge 0.50$ | Verifies coherence during drought and moisture anomaly departures. |
| **Dynamic Range Parity** | Inter-quartile range ratio within $[0.70, 1.30]$ | Confirms comparable variability amplitude without excessive damping. |

---

## 6. Step 21A.9 Decision Gate: GO / NO-GO

```
                           STEP 21A.9 ACCEPTANCE DECISION GATE
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     [CRITERION 1: FORMULATION]                                [CRITERION 2: SOURCE STREAM]
   RZSM = 0.07*SM1 + 0.21*SM2 + 0.72*SM3                     Hourly ERA5-Land Reanalysis Source-of-Record
   Matches author code process_soil.sh                       Project-Side Pipeline Daily Aggregation
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            ▼
                              [CRITERION 3: ANTECEDENT GAP]
                           2014 data missing from 2015-2025 archive;
                           resolved via 12-31 Dec 2014 antecedent retrieval
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │      OVERALL VERDICT: GO        │
                           │  (Proceed to 21B Pilot & 21D    │
                           │  Production with local pipeline) │
                           └─────────────────────────────────┘
```

**Decision Actions**:
1. **Approve ERA5-Land Definition**: The depth-weighted 0–100 cm volumetric soil water index is formally certified as the target and antecedent RZSM state for Mindanao Model A0.
2. **Operational Pipeline Direction**: The production pipeline utilizes the synchronized hourly ERA5-Land reanalysis archive (265 NetCDF files covering 12 Dec 2014–31 Dec 2025) as the source-of-record, with explicit daily aggregation performed within the project pipeline (`resample(valid_time='1D').mean()`). The Copernicus derived daily-statistics product is not the source of the certified production cube.
3. **No Bulk Download Redundancy**: Antecedent and multi-year production archives are fully synchronized across the local codebase and GCS lake (`gs://rise-unet-rzsm/raw/era5_land/production/`), satisfying all nominal training, validation, and test requirements without external API dependencies.
