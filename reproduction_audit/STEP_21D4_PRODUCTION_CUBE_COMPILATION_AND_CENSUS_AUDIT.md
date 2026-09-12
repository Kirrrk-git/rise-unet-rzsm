<!-- markdownlint-disable -->
# Step 21D.4: 11-Year Production RZSM Data Cube Compilation & Anomaly Pipeline Audit Report

```text
Document Authority: Operational Execution Matrix & Artifact Registry
Contract References: contracts/spatial_foundation_contract.yaml, contracts/variable_contract_mapping.yaml
Milestone: Sub-Phase 21D (Step 21D.4 Production Cube Compilation & Census Certification)
Status: [PASS] / [VERIFIED] / [ACCEPTED]
```

---

## 1. Executive Summary

This dossier delivers the formal scientific audit, numerical census certification, and artifact inventory for **Step 21D.4** of the Mindanao RISE-UNet adaptation: the multi-year production compilation of the **11-year ($2015\text{--}2025$) Root-Zone Soil Moisture (RZSM) spatio-temporal data cube** (`era5_land_rzsm_production_2015_2025.nc`).

> [!IMPORTANT]
> **Source-of-Record Specification**:
> **The A0 production RZSM cube uses the hourly ERA5-Land reanalysis as the source-of-record and performs explicit daily aggregation within the project pipeline. The Copernicus derived daily-statistics product is not the source of the certified 21D.4 production cube.**

The pipeline was executed end-to-end via [`notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb`](../notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb) on a Google Colab High-RAM runtime, processing all **265 synchronized ERA5-Land NetCDF archives** ($3.67\text{ GiB}$) covering the full support window from **12 Dec 2014 through 31 Dec 2025** ($96,912$ hourly records across $4,038$ calendar days; 264 files for the nominal 2015–2025 archive + 1 file for the late-2014 antecedent support window).

### Pipeline Architecture & Explicit Responsibility Split
To preserve scientific accountability and modular software design, the production pipeline enforces a strict three-tier division of responsibility:
1. **`21D.4-PREFLIGHT`**: Validates raw source archive integrity (265 NetCDFs, 96,912 hours, zero missing dates), coordinate conventions, and spatial contract locks prior to compilation.
2. **`compile_cube.py`**: The modular compilation engine implementing vectorized spatial remapping (`FastLandAwareRemapper`), trailing rolling averages, and out-of-sample temporal leakage isolation.
3. **`STEP_21D4_PRODUCTION_CUBE_COMPILATION_AND_CENSUS_AUDIT` (This Dossier)**: Directly audits and certifies the resulting production NetCDF artifact (`era5_land_rzsm_production_2015_2025.nc`), establishing empirical verification through an exhaustive zero-tolerance census.

The resulting compiled production dataset was formally audited against all mathematical and spatial contracts, achieving **100% compliance with zero NaNs, zero Infs, no observed active-cell remapping gaps (with coastal fallback applied where required), and strict causal structure across all $506,268$ active evaluation points (where temporal transformations are causal by construction, and leakage-isolation tests confirm out-of-sample data do not alter fitted training climatology or normalization bounds)**.

$$\Large\boxed{\textbf{STEP 21D.4 PRODUCTION VERDICT: CERTIFIED PASS}}$$

---

## 2. Scientific & Architectural Foundations

Every operational transformation implemented in Step 21D.4 strictly enforces the frozen Mindanao adaptation contracts:

### 2.1 Spatial Grid & Geodetic Quadrature Terminology
To prevent ambiguity in downstream evaluation and thesis documentation, spatial area metrics are strictly classified under four distinct physical definitions:
* **Authoritative Regional Boundary Area**: $99,948.76\text{ km}^2$ (derived from official PSA / GeoRiskPH administrative boundaries via WGS84 geodesic polygon quadrature).
* **Total Full-Cell Footprint of 126 Selected Cells**: $96,085.57\text{ km}^2$ (the sum of the complete $0.25^\circ \times 0.25^\circ$ rectangular ground footprints for all 126 included grid cells, each spanning $\approx 762.58\text{ km}^2$).
* **Actual Boundary-Intersection Area within Selected Cells**: $86,418.83\text{ km}^2$ (the exact geographical landmass intersection contained within the 126 active cells, representing **$86.46\%$** of the authoritative $99,948.76\text{ km}^2$ boundary).
* **Fractional-Overlap Area across ALL 1,536 Grid Cells**: $\sum_{i,j} f_{i,j} \cdot A_{i,j} = \mathbf{99,948.76\text{ km}^2}$ ($0.0000\%$ quadrature discrepancy, demonstrating **area-conservative fractional discretization** under WGS84 ellipsoidal geodesics).
* **Binary Evaluation Domain vs. Mindanao Territory**: The 126 cells constitute the **binary evaluation domain** defined by the contractual majority rule ($f \ge 0.50$); they do not encompass all land territory, as the remaining $13.54\%$ of the boundary area lies across 127 sub-threshold coastal and islet transition cells ($0 < f < 0.50$).
* **UNet Pooling Compliance**: Candidate A dimensions ($32 \times 48 = 1,536$ total cells) are divisible by $16$ ($2^4$), enabling 4-level deep UNet encoder-decoder downsampling/upsampling without coordinate truncation or padding distortions.
* **Buffer/Ocean Cells**: Strictly 1,410 inactive computational cells zero-filled across all variables.

### 2.2 Vertical Depth Weighting (0–100 cm Soil Profile)
Root-zone soil moisture ($0\text{--}100\text{ cm}$) is derived from the native ERA5-Land volumetric soil water layers (`swvl1`, `swvl2`, `swvl3`) via depth-integrated trapezoidal weights matching layer thicknesses:
$$\text{RZSM}_{0\text{-}100} = \frac{7}{100}\,\text{swvl}_1 + \frac{21}{100}\,\text{swvl}_2 + \frac{72}{100}\,\text{swvl}_3 = 0.07\,\text{swvl}_1 + 0.21\,\text{swvl}_2 + 0.72\,\text{swvl}_3$$
Weights sum to identically $1.0000$, preserving physical volumetric units ($\text{m}^3/\text{m}^3$).

### 2.3 Land-Aware Linear Spatial Remapping & Coastline Preservation
Native ERA5-Land $0.10^\circ$ land points are remapped to Candidate A $0.25^\circ$ grid cells via a 2-stage land-aware procedure (`FastLandAwareRemapper`):
1. **Piecewise-Linear Interpolation**: Evaluated over active evaluation cells via Delaunay triangulation simplex barycentric interpolation (`LinearNDInterpolator`) using finite native land points (ocean cells in ERA5-Land are NaNs).
2. **Nearest-Neighbor Extrapolation Fallback**: For target coastal edge cells lying outside the convex hull of valid land points, `NearestNDInterpolator` provides extrapolation from adjacent valid land observations, ensuring island coastlines and peninsulas (Zamboanga, Surigao, Davao Oriental) avoid remapping dropouts. No active-cell remapping gaps were observed on the evaluation domain; coastal fallback was applied where required.
3. **Strict Zero-Fill Padding**: All 1,410 non-evaluation buffer/ocean cells are padded strictly to 0.0.
4. **Numerical Agreement Validation**: Observed maximum absolute difference = 0.0 on tested data; formal assertion test tolerance = 1e-6 relative to reference spatial remapper (`griddata(method='linear')`).
5. **CDO Compatibility**: The frozen target grid is compatible with the CDO grid descriptor (`processed/grid/mindanao_0.25_grid.grd`), while the production A0 pipeline uses the project's native `FastLandAwareRemapper`.

### 2.4 Antecedent Warmup & Causal Temporal Preprocessing
* **Trailing Backward Rolling Mean**: Computed as a 7-day backward-looking rolling average (`center=False`). For day $t$, the value incorporates days $[t-6, t]$ (causal by construction).
* **Antecedent Support Window**: Sourced from late-2014 data ($12\text{--}31$ Dec 2014, 20 calendar days). This ensures the nominal opening days of the record ($1\text{--}6$ Jan 2015) have a fully warmed, finite 7-day antecedent memory without requiring lookahead or producing initial NaNs under the verified EX29 Lag $-14$ contract (provisional on final case calendar frozen in Sub-Phase 21G).
* **Locked Model A0 Seasonal Climatology**: 3-month seasonal baselines (DJF: Dec–Feb, MAM: Mar–May, JJA: Jun–Aug, SON: Sep–Nov) fitted **strictly on training years $\le 2021$** ($2015\text{--}2021$, 7 full years). Validation ($2022\text{--}2023$) and test ($2024\text{--}2025$) years are evaluated against this locked baseline with zero data snooping.
* **Leakage-Isolation Proof**: Explicit unit tests (`test_out_of_sample_leakage_isolation`) confirm that out-of-sample data ($\ge 2022$) do not alter fitted training climatology or normalization bounds.
* **Domain-Wide Active Normalization**: Linear min-max scaling to $[0, 1]$ anchored to training distribution extremes ($[0.2319, 0.5898]$):
$$\text{RZSM}_{\text{norm}} = \frac{\text{RZSM}_{\text{anomaly}} - \min_{\text{train}}}{\max_{\text{train}} - \min_{\text{train}}}$$


---

## 3. Formal Numerical Census Certification

The compiled production cube was audited directly via `verify_production_cube_census`:

| Census Metric | Nominal Expectation | Measured Value | Compliance Verdict |
| :--- | :--- | :--- | :---: |
| **Nominal Timesteps** | $4,018$ days ($2015\text{--}2025$) | $4,018$ days | ✅ **EXACT MATCH** |
| **Active Evaluation Cells** | Strictly 126 cells | 126 cells | ✅ **EXACT MATCH** |
| **Total Active Evaluations** | $4,018 \times 126 = 506,268$ | $506,268$ points | ✅ **EXACT MATCH** |
| **NaNs across Active Cells** | Exactly 0 | 0 | ✅ **ZERO NANS** |
| **Infs across Active Cells** | Exactly 0 | 0 | ✅ **ZERO INFS** |
| **Nonzero Ocean Cell Leakage**| Exactly 0 across 1,410 buffer cells | 0 | ✅ **ZERO LEAKAGE** |
| **Active Normalized Min** | $\ge 0.0$ | $0.0000$ | ✅ **IN BOUNDS** |
| **Active Normalized Max** | $\le 1.0$ | $1.0000$ | ✅ **IN BOUNDS** |
| **Active Normalized Mean** | Nominal $\approx 0.52$ | $0.5205$ | ✅ **BALANCED DISTRIBUTION** |
| **Active Normalized Median** | Nominal $\approx 0.53$ | $0.5295$ | ✅ **SYMMETRIC ANOMALY** |

---

## 4. Verification Composite & Publication Figure

The 4-panel publication-grade verification composite was generated and validated:

* **File Location (Codebase)**: [`figures/mindanao_production_cube_verification_composite.png`](../figures/mindanao_production_cube_verification_composite.png) ($423,468\text{ bytes}$)
* **File Location (GCS Cloud Lake)**: `gs://rise-unet-rzsm/figures/mindanao_production_cube_verification_composite.png`
* **Composite Panel Breakdown**:
  * **Panel (A)**: 11-Year Climatological Mean RZSM ($2015\text{--}2025$) over Mindanao active land cells with PSA administrative boundary overlay and Candidate A bounding box. Active land mean: $0.418\text{ m}^3/\text{m}^3$ (range $[0.28, 0.50]$).
  * **Panel (B)**: Locked Model A0 DJF Northeast Monsoon Climatological Baseline ($2015\text{--}2021$).
  * **Panel (C)**: 11-Year Daily 7-Day Trailing Rolling RZSM Hydrograph with color-coded partition spans (Training: 2015–2021, Validation: 2022–2023, Test: 2024–2025) and compact 2-column lower-right legend.
  * **Panel (D)**: Standardized Seasonal Anomaly Distribution $[0, 1]$ over all $506,268$ finite points (Mean: $0.5205$, Median: $0.5295$).

---

## 5. Dual Codebase & Cloud Lake Artifact Inventory

Full dual parity has been established across both the repository and Google Cloud Storage:

| Artifact | Local Repository Path | GCS Cloud Lake URI | Size / Format | Status |
| :--- | :--- | :--- | :--- | :---: |
| **11-Year Production NetCDF Cube** | `processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc` | `gs://rise-unet-rzsm/processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc` | $8.40\text{ MiB}$ (CF-1.8 NetCDF) | `[PASS / CERTIFIED ARTIFACT]` |
| **Production Verification Figure** | `figures/mindanao_production_cube_verification_composite.png` | `gs://rise-unet-rzsm/figures/mindanao_production_cube_verification_composite.png` | $413.5\text{ KiB}$ (PNG, 300 DPI) | `[PASS]` |
| **Interactive Pipeline Notebook** | `notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb` | `gs://rise-unet-rzsm/notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb` | 16 Cells (Full Colab Outputs) | `[PASS]` |
| **Preflight Verification Script** | `scripts/verify_step_21d4_preflight.py` | — | Standalone Python CLI | `[PASS]` |
| **Modular Production Engine** | `src/data/compile_cube.py` | — | Vectorized Python Module | `[PASS]` |
| **Automated Unit Test Suite** | `tests/test_compile_cube.py` | — | 27 Unit Tests Passing (5 pipeline tests; validates software implementation confidence) | `[PASS]` |

---

## 6. Sign-off & Milestone Advancement

All criteria for **Step 21D.4 (Data Preparation & Multi-Year Compilation)** have been satisfied and certified under the three-tier standard:
* `[PASS]`: Software implementation confidence established via 27 passing unit tests; production NetCDF data cube artifact independently audited and certified with zero failures (506,268 finite evaluation cell-days, 0 NaNs/Infs, 0 nonzero buffer leakage).
* `[VERIFIED]`: Temporal processing and target reconciliation contracts strictly match parent EX29 parity for audited components.
* `[ACCEPTED]`: Candidate A spatial grid and Mindanao geodetic boundary remapping fully validated and accepted for regional execution.
* *(Methodological Note)*: **Scientific forecasting validity** is NOT claimed at this stage; it will be evaluated and established when Baseline Model A0 is trained and verified on full multi-source tensors in Sub-Phase 21K and evaluated against benchmark comparators in Phase 22/26.

**Sub-Phase 21D is formally COMPLETE.** The workspace is cleared to proceed with dynamic S2S integration (**Sub-Phase 21E** pilot acquisition) and full atmospheric reanalysis ingestion (**Step 21C.3**).
