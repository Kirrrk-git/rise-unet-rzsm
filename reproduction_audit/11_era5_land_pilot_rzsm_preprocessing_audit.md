<!-- markdownlint-disable -->
# ERA5-Land Soil Moisture Pilot Preprocessing & Verification Audit Report

**Domain**: Depth-Weighted Root-Zone Integration, 0.25° Bilinear Remapping & Evaluation Masking  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Milestones**: Sub-Phase 21B Steps 21B.3 & 21B.4  
**Validation Notebook**: [`notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb`](../notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb)  
**Status**: **EMPIRICALLY VERIFIED & CERTIFIED PASS**  
**Date**: 2026-09-11  

---

## 1. Executive Summary & Objective

Following the empirical completion of **Step 21B.1** (archive integrity verification) and **Step 21B.2** (targeted 20-day 2014 antecedent ingestion and 100% GCS synchronization), this audit report establishes the formal mathematical, computational, and spatial protocol for:
1. **Step 21B.3**: Pilot depth-weighted Root-Zone Soil Moisture (RZSM, 0–100 cm) calculation and bilinear remapping to our frozen $0.25^\circ$ reference grid ([`processed/grid/mindanao_0.25_grid.grd`](../processed/grid/mindanao_0.25_grid.grd)).
2. **Step 21B.4**: Spatial masking against the frozen binary evaluation mask ([`processed/grid/mindanao_eval_mask_025.nc`](../processed/grid/mindanao_eval_mask_025.nc), 126 active cells) and physical distribution verification.

---

## 2. Mathematical Formulation & Code Lineage

### 2.1 Depth-Weighted Root-Zone Soil Moisture Integration
In accordance with Kyle Lesinger's parent study implementation ([`Data/raw_downloads/ERA5_real/process_soil.sh:L27-35`](../Data/raw_downloads/ERA5_real/process_soil.sh#L27-L35)), the volumetric soil water content across the top 100 cm is computed as the depth-weighted sum of the three upper ECMWF model layers:

$$\text{RZSM}_{0-100} = 0.07 \cdot \text{swvl}_1 + 0.21 \cdot \text{swvl}_2 + 0.72 \cdot \text{swvl}_3$$

Where:
* $\text{swvl}_1$: Volumetric soil water layer 1 ($0\text{--}7\text{ cm}$, thickness $\Delta z_1 = 7\text{ cm} \to 7\% = 0.07$)
* $\text{swvl}_2$: Volumetric soil water layer 2 ($7\text{--}28\text{ cm}$, thickness $\Delta z_2 = 21\text{ cm} \to 21\% = 0.21$)
* $\text{swvl}_3$: Volumetric soil water layer 3 ($28\text{--}100\text{ cm}$, thickness $\Delta z_3 = 72\text{ cm} \to 72\% = 0.72$)

### 2.2 Temporal Daily Aggregation
Before or during depth weighting, each calendar day's 24 hourly values ($t = 00:00, \dots, 23:00\text{ UTC}$) are averaged via unweighted arithmetic mean:

$$\bar{x}_{\text{daily}} = \frac{1}{24}\sum_{t=0}^{23} x_{\text{hourly}}(t)$$

This matches author line [`process_soil.sh:L20`](../Data/raw_downloads/ERA5_real/process_soil.sh#L20) (`cdo daymean "$file" "$daymean_file"`).

---

## 3. Spatial Bilinear Remapping Architecture

### 3.1 Source Grid vs. Target Grid
* **Source Grid**: Native ERA5-Land $0.10^\circ \times 0.10^\circ$ regular Gaussian-derived grid ($71 \times 111 = 7,881$ cells, extent $[116.5^\circ\text{E}, 4.0^\circ\text{N}] \to [127.5^\circ\text{E}, 11.0^\circ\text{N}]$).
* **Target Grid**: Authoritative frozen **Candidate A Reference Grid** ($32 \times 48 = 1,536$ cells, extent $[116.00^\circ\text{E}, 11.75^\circ\text{N}] \to [127.75^\circ\text{E}, 4.00^\circ\text{N}]$ with cell step $0.25^\circ$).

### 3.2 Interpolation Operator
Bilinear interpolation (`cdo remapbil`) weights the four nearest source cell centers to compute values at the target $0.25^\circ$ cell centers:

$$I(x, y) \approx \sum_{k=1}^4 w_k \cdot S(x_k, y_k), \quad \sum_{k=1}^4 w_k = 1$$

---

## 4. Evaluation Masking & Computational Zeroing Protocol

Per the parent RISE-UNet contract (Lesinger & Tian 2025):
1. **Active Evaluation Cells ($N = 126$, $8.20\%$ of grid)**: Confined strictly to cells where fractional administrative boundary coverage $f \ge 0.50$ ([`processed/grid/mindanao_eval_mask_025.nc`](../processed/grid/mindanao_eval_mask_025.nc)).
2. **Buffer / Ocean Cells ($N = 1,410$, $91.80\%$ of grid)**: Padded to constant `0.0` in model input tensors, isolating model training and metric computation strictly to valid regional land.
3. **NaN Enforcement**: Zero NaNs or Infs permitted across the 126 evaluation cells for any time step.

---

## 5. Verification Protocol & Empirical Results

| Step | Verification Dimension | Target Specification | Observed Result | Verdict |
| :---: | :--- | :--- | :---: | :---: |
| **21B.3** | **Daily Timestep Count** | Dec 12–31, 2014 ($T=20$ days) | 20 daily timesteps ($480 \to 20$) | ✅ **PASS** |
| **21B.3** | **Execution Speed** | Vectorized interpolation throughput | 2.05 s for 20 days (0.103 s/day) | ✅ **PASS** |
| **21B.3** | **Remapped Tensor Shape** | Target Candidate A grid | $(20, 32, 48)$ exact | ✅ **PASS** |
| **21B.4** | **Active Cell Integrity** | 126 evaluation cells across 20 days | 2,520 valid samples; **0 NaNs, 0 Infs** | ✅ **PASS** |
| **21B.4** | **Ocean Padding** | 1,410 buffer/ocean cells | Exactly $0.0$ (0 non-zero cells) | ✅ **PASS** |
| **21B.4** | **Physical Bounds** | $0.10 \le \text{RZSM} \le 0.60\text{ m}^3/\text{m}^3$ | Min: 0.2488, Max: 0.5087, Mean: 0.4156 | ✅ **PASS** |
| **21B.4** | **Visual Verification** | 4-panel composite map | Generated & saved (300 DPI) | ✅ **PASS** |
| **21B.4** | **Artifact Export** | NetCDF CF-1.8 pilot dataset | Saved in `processed/rzsm/pilot/` | ✅ **PASS** |

---

## 6. Generated Visual Artifact

The 4-panel diagnostic composite map has been generated and validated at 300 DPI with true geographic coordinates and administrative boundary overlays:
* Local Path: [`processed/grid/figures/mindanao_rzsm_pilot_preprocessing_composite.png`](../processed/grid/figures/mindanao_rzsm_pilot_preprocessing_composite.png)
* Cloud Path: `gs://rise-unet-rzsm/figures/spatial/mindanao_rzsm_pilot_preprocessing_composite.png`
* Panels:
  * **(A) Native ERA5-Land RZSM ($0.10^\circ$)**: Raw depth-weighted 0–100 cm daily field for Dec 25, 2014 with red Mindanao boundary overlay.
  * **(B) Bilinear Remapped Grid ($0.25^\circ$ Candidate A)**: Smooth continuous interpolation across regional land cells with ocean cleanly masked (eliminating artificial triangulation across open sea).
  * **(C) Frozen Binary Evaluation Mask Contract**: Categorical map displaying active evaluation cells ($N = 126$, $f \ge 0.50$, dark blue), sub-threshold fringe cells ($N = 127$, orange), and ocean cells ($N = 1,283$, off-white).
  * **(D) Model-Ready RISE-UNet Input Tensor ($32 \times 48$)**: Final model input showing true volumetric soil moisture values across the 126 evaluation cells with the 1,410 ocean/buffer cells padded strictly to $0.0$ and styled as neutral zero-padded buffer.

---

## 7. Empirical Certification & Closure Record

| Execution Track | Environment | Runtime (20 Days) | Numerical Census (Active Cells) | NetCDF Export | GCS Upload | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Local Host** | Windows 11 / Python 3.14 | 2.12 s (0.106 s/day) | 2,520 samples (0 NaNs, 0 Infs) | `processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc` (130.6 KB) | Verified | ✅ **PASS** |
| **Remote Cloud** | Google Colab / Tesla T4 | 1.48 s (0.074 s/day) | 2,520 samples (0 NaNs, 0 Infs) | CF-1.8 NetCDF validated (`commit f2a346a`) | Uploaded to `gs://rise-unet-rzsm/processed/rzsm/pilot/` | ✅ **PASS** |

$$\Large\boxed{\textbf{SUB-PHASE 21B (STEPS 21B.3 \& 21B.4) CERTIFIED COMPLETE}}$$
*(Depth-weighted RZSM integration and land-aware bilinear remapping verified across both local and Colab environments; output tensors conform to Candidate A geometry with zero NaNs; preprocessed pilot NetCDF, usage documentation, and 919 KB publication composite synchronized across Git and Google Cloud Storage; pipeline cleared for Sub-Phase 21C ERA5 atmospheric pilot acquisition).*

