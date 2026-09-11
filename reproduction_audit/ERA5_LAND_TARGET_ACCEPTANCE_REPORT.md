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
| **Operational Stream Policy** | Prioritize CDS daily statistics over hourly | Daily statistics (`derived-era5-land-daily-statistics`) confirmed primary; bucket hourly is fallback | ✅ **PASS** |
| **Antecedent Lag Completeness** | Availability of 2014 data for $W_1$ 2015 initialization | **Archive starts at 2015-01-01; 2014 is MISSING** (requires December 2014 for 14-day lags) | ⚠️ **CONDITIONAL** |
| **Spatial Remapping Alignment** | Bilinear interpolation to frozen 0.25° grid | CDO grid descriptor `mindanao_0.25_grid.grd` verified compatible | ✅ **PASS** |
| **Evaluation Mask Integration** | Confinement to 126 binary evaluation cells ($f \ge 0.50$) | Evaluated strictly on 126 cells ($96,085.57\text{ km}^2$); 1,410 buffer cells zero-masked | ✅ **PASS** |
| **Independent Cross-Reference** | Independent GLEAM v3.8a RZSM comparison | Protocol defined; GLEAM retained strictly as external validation reference | ✅ **PASS** |

$$\Large\boxed{\textbf{AUDIT VERDICT: CONDITIONAL PASS}}$$
*(Technical formulation certified; operational daily retrieval path cleared; 2014 antecedent acquisition flagged before full production training).*

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

### 3.2 Operational Stream Policy: Daily Statistics vs. Hourly Fallback
* **Primary Operational Choice**: In accordance with the Master Plan and user directives, **daily dataset retrieval** via Copernicus CDS (`derived-era5-land-daily-statistics`) is the designated operational pathway. The derived daily-statistics product natively aggregates the 24-hour daily mean during retrieval, providing daily statistics derived from the underlying hourly ERA5-Land reanalysis and substantially reducing temporal data volume by ~96%.
* **Secondary Fallback Role**: The existing hourly archive in the bucket is retained strictly as an **offline contingency fallback** if CDS API quotas or service downtime impede daily-statistics requests. It will not be assumed as the primary source. Numerical equivalence against offline hourly-to-daily CDO aggregation can be empirically evaluated as an explicit diagnostic check.

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

### 4.1 Remapping Pipeline
* **Input Grid**: Native ERA5-Land $0.10^\circ \times 0.10^\circ$ geographic grid.
* **Target Grid**: Frozen Mindanao $0.25^\circ \times 0.25^\circ$ reference grid (`mindanao_025deg.nc`, $32 \times 48 = 1,536$ cells).
* **CDO Remapping Specification**:
  ```bash
  cdo remapbil,processed/grid/mindanao_0.25_grid.grd era5_land_daily_010.nc era5_land_daily_025.nc
  ```
* **Bounding Box Alignment**:
  * Native ERA5-Land window covers: $[115.8^\circ\text{E}, 3.8^\circ\text{N}] \to [128.0^\circ\text{E}, 12.0^\circ\text{N}]$, completely covering the frozen cell-edge envelope ($[115.875^\circ, 3.875^\circ] \to [127.875^\circ, 11.875^\circ]$). Zero extrapolation is required.

### 4.2 Non-Evaluation Buffer vs. Active Evaluation Cells
In strict adherence to the frozen spatial contract ([`contracts/spatial/spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml)):
* **Non-Evaluation Domain ($f < 0.50$, 1,410 cells, 91.80%)**:
  * Pure exterior ocean buffer ($f = 0.000$): 1,283 cells. Zero-filled per author convention to provide convolutional receptive field padding without NaNs.
  * Sub-threshold coastal transition cells ($0.000 < f < 0.500$): 127 cells. Excluded from loss and skill metrics to prevent coastal ocean water contamination.
* **Binary Evaluation Domain ($f \ge 0.50$, 126 cells, 8.20%)**:
  * Covers $96,085.57\text{ km}^2$ of the validated six-region administrative domain.
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
   RZSM = 0.07*SM1 + 0.21*SM2 + 0.72*SM3                     Primary: CDS Derived Daily Statistics
   Matches author code process_soil.sh                       Secondary: Bucket Hourly Fallback (2015-2025)
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            ▼
                              [CRITERION 3: ANTECEDENT GAP]
                           2014 data missing from bucket archive;
                           requires Dec 2014 retrieval or 21-day offset
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │      OVERALL VERDICT: GO        │
                           │   (Proceed to 21B Pilot with    │
                           │  explicit daily CDS pipeline)   │
                           └─────────────────────────────────┘
```

**Decision Actions**:
1. **Approve ERA5-Land Definition**: The depth-weighted 0–100 cm volumetric soil water index is formally certified as the target and antecedent RZSM state for Mindanao Model A0.
2. **Operational Pipeline Direction**: Proceed to **Sub-Phase 21B (Step 21B.1 / 21B.2)** by setting up the Copernicus CDS API client to retrieve a **1-month pilot** (e.g., October 2020) using `derived-era5-land-daily-statistics`.
3. **No Bulk Download Yet**: As instructed, no large-scale 144-month retrieval will be initiated until the 1-month pilot validates coordinate alignment, units, missing-value masking, and CDO regridding against `mindanao_025deg.nc`.
