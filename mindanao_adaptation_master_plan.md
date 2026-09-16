<!-- markdownlint-disable -->
# RISE-UNet Mindanao Adaptation & Enhancement — Phase 21 to Completion Master Plan

**Project**: Enhanced RISE-UNet with Lead-Aware Recursive Residual Refinement for Subseasonal Root-Zone Soil-Moisture Drought Forecasting in Mindanao  
**Thesis Track**: Track B (Mindanao Regional Adaptation & Proposed Enhancements)  
**Parent Study**: Lesinger & Tian (2025), *Nature Communications*, DOI: `10.1038/s41467-025-62761-3`  
**Active Repository**: `https://github.com/Kirrrk-git/rise-unet-rzsm.git`  
**GCS Bucket**: `gs://rise-unet-rzsm/`  
**Status**: Sub-Phases 21A, 21B, 21C (Training Fold Jan 2015 – Dec 2021 + Dec 2014 antecedent secured; 2022–2023 Validation Fold Secured & Certified under Pre-Production Gate 1; 2024–2025 Test Fold mirroring in progress), 21D, 21E, 21F (`[PASS / VERIFIED]`), 21G (`[PASS / VERIFIED]`), 21H (`[SURROGATE_ONLY: Infrastructure Smoke Test]`), 21I (`[SURROGATE_ONLY: Optimization Smoke Test]`), 21J / Pre-Production Gate 2 (`[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]`), 21K.1 (`[PASS / VERIFIED / ACCEPTED: 2026-09-14]`), 21K.2 (`[PASS / VERIFIED / ACCEPTED: 2026-09-14]`), Pre-Production Gate 1 (`[PASS / VERIFIED / ACCEPTED: 2026-09-16]`), and Step 21K.3-pre / Pre-Production Gate 3 (`[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]`). Authoritative status tracking governed by [`contracts/A0/VERIFICATION_STATUS.yaml`](contracts/A0/VERIFICATION_STATUS.yaml). All 3 Pre-Production Gates cleared; Full 3-Seed Model A0 Production Training is formally AUTHORIZED (`Step 21K.3: AUTHORIZED: 2026-09-17`).


## Executive Verification & Methodological Alignment

Following the formal empirical certification of **Gate 1A** (parent architecture integrity) and **Gate 1B** (parent EX29 configuration and recursive pipeline trace), this master plan establishes the execution framework for adapting the RISE-UNet deep learning forecasting model to Mindanao, Philippines.

### Three-Tier Scientific Certification Standard

To ensure thesis-level rigor and prevent conflating software execution with published methodological fidelity, all milestones, findings, and technical assertions throughout this project are classified under an explicit three-tier taxonomy:
1. **`[PASS]` (Software / Numerical Integrity)**:
   Automated unit tests, execution assertions, and numerical scripts ran cleanly with zero failures, zero errors, finite outputs, and machine-precision arithmetic compliance. Proves the code works as implemented.
2. **`[VERIFIED]` (Parent Source-Code Reconciliation)**:
   Direct empirical reconciliation against Kyle Lesinger's authoritative parent EX29 codebase (`dl_dm_rzsm_subseasonal_forecast`). Proves that an algorithmic or mathematical specification reproduces the exact logic of the peer-reviewed parent baseline.
3. **`[ACCEPTED]` (Methodological Regional Adaptation)**:
   Deliberate, justified scientific and engineering decisions adopted specifically for the Mindanao regional adaptation (Track B) where the regional setting, tropical climatology, or target definition intentionally diverges from or extends the parent CONUS implementation.

### Document Authority Hierarchy

To guarantee architectural coherence and eliminate competing sources of truth, this project enforces an explicit three-tier authority hierarchy:

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

### Independent Empirical Verification of User Feedback

> [!CRITICAL]
> **User Feedback is Advisory, NOT Ground Truth**:
> All suggestions, critique, arithmetic, and feedback provided by the user must be treated as advisory review and expert guidance—**never as unverified ground truth**.
> 
> Before incorporating any user claim, critique, or suggested number into the codebase, contracts, or audit records:
> 1. Independently calculate and verify the arithmetic (e.g., date ranges, cell counts, tensor shapes).
> 2. Cross-reference the claim against actual source code, raw data files, and published literature (Lesinger & Tian, 2025).
> 3. Execute unit tests or empirical scripts to confirm software and mathematical behavior.
> 4. Explicitly state the verification findings before committing changes.

> [!NOTE]
> **Living Operational Registry & Map**:
> The complete directory map, operational matrix, and comprehensive file registry for all files introduced in this project are tracked in [`reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md). Under workspace governance rules, this registry is synchronized upon remarkable changes (excluding formatting, typos, or temporary scratch scripts).

---

### Core Methodological Pillars

1. **Parent-Consistent A0 Adaptation**:
   * **Regional Adaptation, Not Literal Reproduction**: Model A0 adapts the verified EX29-derived recursive RISE-UNet framework from Lesinger & Tian (2025) to the Mindanao domain.
   * **Explicit Scope Declaration (Forecast Horizon W1–W4)**: The parent study evaluated forecast leads 1 to 5. The Mindanao thesis adaptation explicitly focuses on **Weeks 1 to 4** ($W_1 \to W_4$), which defines the operational subseasonal drought horizon for the region; the fifth parent lead is formally declared outside the project scope.
   * **Three-Way Methodological RZSM Architecture**:
     * *Parent Baseline Reference (CONUS)*: Lesinger & Tian (2025) utilized daily GLEAM v3.8a RZSM (0–100 cm depth integration combining surface 0–10 cm and root-zone 10–100 cm) as the primary RZSM target and antecedent reference, hourly ERA5 atmospheric reanalysis for 5 predictor variables, and GEFSv12 / ECMWF S2S dynamic model reforecasts.
     * *Mindanao A0 Target/Reference RZSM Definition [ACCEPTED]*: Model A0 deliberately departs from the parent GLEAM target by formally defining the regional RZSM target and antecedent state as the ERA5-Land-derived 0–100 cm depth-weighted volumetric soil-water index:
       $$\text{RZSM}_{0-100} = 0.07 \cdot \text{SM}_1 + 0.21 \cdot \text{SM}_2 + 0.72 \cdot \text{SM}_3$$
       This represents an intentional Track B methodological commitment rather than mere data convenience, providing a physically consistent, continuous reference for both antecedent predictors and future forecast targets over Mindanao. **The A0 production RZSM cube uses the hourly ERA5-Land reanalysis as the source-of-record and performs explicit daily aggregation within the project pipeline. The Copernicus derived daily-statistics product is not the source of the certified 21D.4 production cube.**
     * *Independent Cross-Reference Validation [ACCEPTED]*: GLEAM v3.8a RZSM is retained strictly as an **independent external reference** for regional suitability assessment (Step 21A.7), comparing spatial and temporal behavior against the ERA5-Land index over their common Mindanao period without using GLEAM as an A0 training target or treating it as an uncritical ground truth.
   * **Published Parent Properties vs. Implementation-Specific Contracts**:
     * *Independently Confirmed Parent Properties (Published Literature)*:
       1. RISE-UNet implements recursive DL-DM subseasonal forecasting where prior-lead predictions feed into subsequent lead predictor tensors.
       2. Predictors at distinct lead/lag combinations are ingested as individual channels (e.g., the paper explicitly notes six lag weeks forming six distinct channels in illustrative configurations).
       3. Demonstrated parent input tensor is $48 \times 96$, reaching $6 \times 12$ through pooling.
       4. Inputs and targets are min-max standardized to $[0, 1]$, and water/ocean bodies are masked with zero.
       5. Atmospheric reanalysis is sourced from ERA5, and dynamic reforecasts are sourced from S2S models.
     * *Implementation-Specific Contracts Verified via Gate 1B Audit & Step 21A.3 Reconciliation*:
       1. The Gate 1B and Step 21A.3 audits confirm that EX29 specifies exactly 3 antecedent RZSM weekly lags (`num_lags_obs_RZSM = 3`), mapped to offsets `[-1, -7, -14]` days (`channelExperiment.py:L107-124`).
       2. Lag locations and antecedent window definitions: Verified that lags represent 7-day backward trailing rolling window averages (`preprocessUtils.py:L88`) rather than discrete daily snapshots (`'RZSM_obs_lag-1': 'RZSM_obs_lag_Wk_1'`, `'RZSM_obs_lag-7': 'RZSM_obs_lag_Wk_2'`, `'RZSM_obs_lag-14': 'RZSM_obs_lag_Wk_3'`).
       3. ECMWF S2S dynamic predictor set corresponds strictly to the verified triplet (`t2m`, `d2m`, `tcw`), with precipitation explicitly excluded.
       4. The lead-dependent channel schedule ($W_1=11, W_2=12, W_3=5, W_4=6$) is verified as the authoritative implementation-specific EX29 contract through Step 21A.3 code audit (`loadDataAllWeeks.py:L710-721`).
     * **Parent Empirical Basis for Recursive Investigation (Lesinger & Tian 2025)**:
       The parent study demonstrates that the hybrid DL-dynamic RISE-UNet achieves skillful RZSM forecasts out to four weeks over CONUS, but critically reveals that prediction skill in Weeks 3–4 is primarily associated with the first two weeks of dynamic atmospheric forecasts combined with antecedent RZSM land memory. Because dynamic atmospheric predictability decays rapidly beyond Week 2, the model relies heavily on its recursive autoregressive chain ($\hat{y}_{W1} \to \hat{y}_{W2} \to \hat{y}_{W3} \to \hat{y}_{W4}$) to propagate soil memory forward. While this empirical finding from the parent CONUS study motivates our Track B research focus, it does not predetermine that the identical error compounding mechanism necessarily manifests in tropical Mindanao. Whether and to what degree recursive autoregressive degradation occurs over Mindanao is treated strictly as an open empirical question to be diagnosed and quantified in Phase 23, prior to the Gate 2 GO / NO-GO refinement decision.


2. **Architecture Compatibility & Geometry Validation**:
   * The four $2\times$ pooling stages of the frozen `UNET_RZSM` / `modelRzsmRelu` backbone impose a base divisibility requirement of 16 for both spatial dimensions ($H, W$).
   * However, divisibility by 16 is a necessary base condition, not by itself sufficient proof that the complete graph (including `Conv2DTranspose` upsampling, multi-scale skip concatenations, and deep-supervision heads) will execute cleanly.
   * Forward graph compatibility is mandatory; gradient propagation provides an additional integrity check. Final compatibility must be established empirically against the frozen implementation using synthetic tensors across candidate dimensions before the reference grid is frozen.

3. **Complete Purge of Legacy Satellite Routes**:
   * All experimental routes involving legacy satellite surface observation regridding, geometric spatial support operators ($H$, $D$, $M$), and observation dropout evaluations are permanently removed from the active roadmap.
   * The thesis direction focuses strictly on mitigating **recursive error propagation and autoregressive feedback degradation** in multi-week subseasonal forecasting.

4. **Post-A0 Roadmap (Recursive Degradation Diagnostic & Refinement)**:
   * **Phase 22**: Controlled reference comparators (B0 Climatology, B1 Persistence, B2 Feature-Engineered XGBoost).
   * **Phase 23**: Empirical Recursive Degradation Diagnostic (evaluating Standard Autoregressive Recursion vs. Oracle Counterfactual Diagnostic vs. Direct Non-Recursive vs. Controlled Error Injection with perturbation propagation tracking).
   * **Gate 2**: Recursive Refinement GO / NO-GO Decision Gate (evaluating statistical significance $p < 0.05$, effect size, confidence intervals, and cross-case consistency, with thresholds frozen prior to inspecting results).
   * **Phase 24**: Lead-Aware Recursive Residual Refinement (Model A1, keeping parent RISE-UNet backbone frozen; transition inputs isolated from verifying ground truth, trained without teacher-forcing).
   * **Phase 25**: Ablation and Robustness Studies.
   * **Phase 26**: Strict matched statistical evaluation on sealed held-out test data (provisional 2024–2025) with paired moving-block bootstrap.
   * **Phase 27**: Research Reproducibility Package and Thesis Archive.

---

## Controlled Experimental Hierarchy

```
                                  ┌────────────────────────────────────────┐
                                  │      GATE 1: PARENT BASELINE TRACE     │
                                  │         (CERTIFIED COMPLETE)           │
                                  └───────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │ PHASE 21: MINDANAO A0 BASELINE MODEL   │
                                  │    EX29-Derived Recursive RISE-UNet    │
                                  │ (ERA5-Land RZSM + ERA5 Atm + ECMWF S2S)│
                                  └───────────┬────────────────┬───────────┘
                                              │                │
                         ┌────────────────────┘                └────────────────────┐
                         │ (Independent Baseline Benchmark)                         │ (Scientific Diagnostic Pathway)
                         ▼                                                          ▼
         ┌───────────────────────────────┐                          ┌───────────────────────────────┐
         │ PHASE 22: REFERENCE BASELINES │                          │ PHASE 23: RECURSIVE DIAGNOSTIC│
         │ B0: Seasonal Climatology      │                          │ Quantify error accumulation:  │
         │ B1: RZSM Lag Persistence      │                          │ Recursive vs. Oracle vs.      │
         │ B2: Structured XGBoost GBDT   │                          │ Direct vs. Error Injection    │
         └───────────────┬───────────────┘                          └───────────────┬───────────────┘
                         │                                                          │
                         │                                                          ▼
                         │                                          ┌───────────────────────────────┐
                         │                                          │ GATE 2: REFINEMENT GO / NO-GO │
                         │                                          │ Effect size + CI + p < 0.05   │
                         │                                          └───────────────┬───────────────┘
                         │                                                          │ (GO)
                         │                                                          ▼
                         │                                          ┌───────────────────────────────┐
                         │                                          │ PHASE 24: MODEL A1 ENHANCEMENT│
                         │                                          │ Lead-Aware Recursive Residual │
                         │                                          │ Refinement (Parent UNet frozen│
                         │                                          │ + lead-conditioned correction)│
                         │                                          └───────────────┬───────────────┘
                         │                                                          │
                         │                                                          ▼
                         │                                          ┌───────────────────────────────┐
                         │                                          │ PHASE 25: ABLATION & STABILITY│
                         │                                          │ Lead-conditioning ablations,  │
                         │                                          │ error-damping sensitivity     │
                         │                                          └───────────────┬───────────────┘
                         │                                                          │
                         └────────────────────────────┬─────────────────────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ PHASE 26: FINAL TEST EVAL     │
                                      │ Sealed 2024–2025 hold-out,    │
                                      │ paired moving-block bootstrap │
                                      └───────────────┬───────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ PHASE 27: THESIS ARCHIVE      │
                                      └───────────────────────────────┘
```

---

## Master Milestone & Progress Tracking Checklist

### Phase 21: Mindanao Adaptation & Baseline Model A0

#### Sub-Phase 21A: Spatial Foundation, Target Acceptance & Full-Cascade Geometry Gate
**Objective**: Establish branch `mindanao-adaptation`, reconcile the EX29 channel/variable contract from parent code, derive candidate spatial grids from official boundary geometry, execute the architecture compatibility gate across the complete 4-week recursive cascade, construct the authoritative 0.25° Mindanao reference grid, evaluate ERA5-Land target acceptance against technical criteria and independent GLEAM cross-reference, construct evaluation masks, and formally freeze the spatial/target contract.

- [x] **Step 21A.1**: Create and switch to isolated Mindanao adaptation Git branch.
  * Command: `git checkout -b mindanao-adaptation`
  * Pass Criterion: Branch created cleanly from certified commit `4adb9fa` on `parent-reproduction`. [PASSED: commit `4adb9fa`, pushed to `origin/mindanao-adaptation`]
- [x] **Step 21A.2**: Obtain authoritative Mindanao administrative reference and GIS boundary.
  * Administrative Identity Authority: Official Philippine Statistics Authority (PSA) Philippine Standard Geographic Code (PSGC) 2Q 2026 (`PSGC_2Q2026_Publication_Datafile.xlsx`, SHA-256 `31892bc2bdde3ea0682562d9412b5bab4d45a0be5e5a5b4f6c9d7714b94bca5d`) covering all 6 Mindanao regions: Region IX (`090000000`), Region X (`100000000`), Region XI (`110000000`), Region XII (`120000000`), Region XIII (`160000000`), and BARMM (9-digit `150000000` / 10-digit `1900000000`). [PASSED: 2026-09-10]
  * GIS Boundary Polygon Authority: Philippine Geoportal / GeoRiskPH `regionalboundary_20210504` regional boundary layer (`geoportal-regionalboundary_20210504_export.geojson`, 519,138,911 bytes, SHA-256 `62847e530c0d46df9ef6dc31ef64e372032ef7c0defc05d4758f4bf8c429ae9c`), cross-referenced to official PSA PSGC standards.
  * Harmonization & Audit: Filtered 6 Mindanao regions from 3,745 source features using `extract_mindanao.py` $\to$ 1,384 selected records, 1,378 polygon features written, 6 label points skipped, 0 unmatched records (`PSA_Geoportal_Regional_20210504_Mindanao.gpkg`, SHA-256 `a10204e1efe8f2ef11d15c17cce8291dfd74b21f4d2ca03badc238e8937f413f`). Verified via automated audit `verify_mindanao_codes.py` (0 errors, 0 missing).
  * Unified Boundary Dissolve: Executed QGIS GEOS native dissolve (`Keep disjoint features separate = False`) $\to$ `mindanao_analysis_boundary.gpkg` (Layer: `mindanao_analysis_boundary`, SHA-256 `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2`). Exactly 1 MultiPolygon feature, EPSG:4326, extent $[118.064236^\circ, 4.587294^\circ] \to [126.604966^\circ, 10.471595^\circ]$. GEOS Check Validity: VALID_COUNT: 1, INVALID_COUNT: 0, ERROR_COUNT: 0.
  * Audit Dossier: Formal certification recorded in `reproduction_audit/06_mindanao_boundary_and_gadm_audit.md`.
  * Pass Criterion: Certified PASS on procedural, administrative, and topological integrity.
- [x] **Step 21A.3**: Reconcile EX29 Channel, Lag & Variable Contract.
  * Literature & Code Reconciliation: Reconciled parent paper illustrative narrative (which mentions 6 lag weeks) with author code in `function/experimentType.py:L57` and `function/loadDataAllWeeks.py:L702-721`, which confirms the primary empirical recursive model EX29 uses `num_lags_obs_RZSM = 3`. [PASSED: 2026-09-10]
  * Antecedent Lag Schedule Verification: Verified EX29 specifies 3 antecedent RZSM weekly lags mapped to offsets `[-1, -7, -14]` via `channelExperiment.py:L107-124`. Confirmed lags represent 7-day backward rolling window averages (`center=False`) rather than instantaneous daily samples via `preprocessUtils.py:L88` and author renaming in `04d_Plot_permutation_test_results_UPDATE.ipynb:L403`.
  * Channel Schedule Verification: Formally confirmed lead-dependent channel schedule across $W_1 \to W_4$: **Lead 1 = 11 channels**, **Lead 2 = 12 channels**, **Lead 3 = 5 channels**, **Lead 4 = 6 channels** (`loadDataAllWeeks.py:L710-721`). Atmospheric obs (5) and dynamic S2S reforecasts (3) are included only for Leads 1 and 2; Leads 3 and 4 drop atmospheric predictors due to dynamic skill decay and ingest recursive prior-lead RZSM predictions.
  * Atmospheric Predictor Mapping: Confirmed parent `pwat` / `pwat_eatm` is Total Column Water Vapour (`tcwv` / `tcw`, $\text{kg/m}^2$) from ERA5 via `total_column_water_merged.nc4` and `addPredictors.py:L62-77`.
  * Specific Humidity Derivation: Documented author MetPy calculation in `preprocessUtils.py:L186` and exact Bolton (1980) vapor pressure / specific humidity thermodynamic equations ($e = 611.2 \exp(17.67(T_d-273.15)/(T_d-29.65))$, $q = \epsilon e / (p - (1-\epsilon)e)$ with $\epsilon \approx 0.62198$).
  * Geopotential Height Mapping: Confirmed raw CDS download is ERA5 `geopotential` ($z$, $\text{m}^2/\text{s}^2$) at 200 hPa (`multi_month_china_download_ERA.py:L102-125`), mapped to channel `z200` in `addPredictors.py:L70`, with conversion to standard geopotential meters ($Z = z / 9.80665$) documented.
  * Dynamic ECMWF S2S Triplet: Confirmed reforecast channels strictly ingest 3 variables (`t2m`, `d2m`, `tcw`) for Leads 1 and 2 (`loadDataAllWeeks.py:L708`), with precipitation explicitly excluded.
  * Audit Dossier: Detailed empirical evidence, code citations, and YAML contract specification recorded in `reproduction_audit/03_parent_variable_contract_audit.md`.
  * Pass Criterion: Certified PASS on channel schedule, lag temporal structure, and atmospheric derivations.
- [x] **Step 21A.4**: Define candidate 0.25° spatial envelopes and candidate model tensor dimensions.
  * Geographic Bounds: Derived candidate extents from validated boundary (`mindanao_analysis_boundary.gpkg`, $[118.064^\circ\text{E}, 4.587^\circ\text{N}] \to [126.605^\circ\text{E}, 10.472^\circ\text{N}]$), requiring minimum span of 24 latitude cells and 35 longitude cells at 0.25°. [PASSED: 2026-09-10]
  * Candidate Computational Domains: Enumerated 4 feasible padded rectangular domains satisfying base 16-pooling divisibility ($H, W \pmod{16} = 0$): Candidate A ($32 \times 48$, 1,536 cells, $4.00^\circ\text{--}11.75^\circ\text{N} \times 116.00^\circ\text{--}127.75^\circ\text{E}$), Candidate B ($32 \times 64$, 2,048 cells, $4.00^\circ\text{--}11.75^\circ\text{N} \times 114.50^\circ\text{--}130.25^\circ\text{E}$), Candidate C ($48 \times 48$, 2,304 cells), and Candidate D ($48 \times 64$, 3,072 cells).
  * Geometric Containment & Divisibility: Audited zero-truncation of the unified boundary across all 4 candidate domains (buffers range from $0.59^\circ$ to $3.65^\circ$). All candidate domains confirmed to yield integer spatial feature map dimensions across all 4 downsampling and upsampling levels ($H/16, W/16 \in \mathbb{Z}^+$).
  * Audit Dossier: Formal certification and coordinate tables recorded in `reproduction_audit/05_mindanao_geometry_envelopes_audit.md`.
  * Pass Criterion: Certified PASS on candidate envelope definition and zero-truncation containment.
- [x] **Step 21A.5**: Synthetic Candidate Tensor & 4-Lead Cascade Compatibility Gate.
  * Empirical Forward/Cascade Test: Traced layer-by-layer spatial feature dimensions across all 4 downsampling and upsampling levels of frozen `UNET_RZSM` (`modelRzsmRelu.py`), confirming that candidate domains yield exact integer feature maps ($H/16, W/16 \in \mathbb{Z}^+$) with zero concatenation mismatch errors. [PASSED: 2026-09-10]
  * Full-Cascade Execution Notebook: Constructed `notebooks/02_mindanao_geometry_and_cascade_gate.ipynb`, implementing the complete 4-week recursive cascade ($W_1 \to W_2 \to W_3 \to W_4$) on synthetic tensors $(M=2, H, W, C_k)$ using reconciled channel counts $[11, 12, 5, 6]$. Executed strictly in native TensorFlow 2.x / Keras using author's frozen `function/modelRzsmRelu.py` (`UNETRzsm.model_build_func`), verifying prior prediction channel concatenation and clean backpropagation gradient flow through all 298 trainable weight tensors with zero NaNs and non-zero updates.
  * Pass Criterion: Certified PASS across all candidate domains ($32 \times 48$, $32 \times 64$, $48 \times 48$, $48 \times 64$).
- [x] **Step 21A.6**: Final Coordinate Grid Selection & NetCDF Construction.
  * Selected Spatial Domain: Formally frozen **Candidate A ($H \times W = 32 \times 48$, 1,536 total cells)** based on full boundary containment, verified 16-pooling divisibility, minimal computational overhead (saving 33% cells relative to $32 \times 64$), and avoidance of unnecessary ocean ingestion while retaining $0.71^\circ\text{--}2.19^\circ$ protective convolutional buffers. [PASSED: 2026-09-10]
  * Explicit Coordinate Convention Decoupling:
    1. **Cell-Center Grid (Coordinate Variables)**: `lat` descending $11.75^\circ\text{N} \to 4.00^\circ\text{N}$ (32 points, step $-0.25^\circ$), `lon` ascending $116.00^\circ\text{E} \to 127.75^\circ\text{E}$ (48 points, step $+0.25^\circ$).
    2. **Cell-Edge Outer Bounding Envelope**: South Edge $3.875^\circ\text{N}$, North Edge $11.875^\circ\text{N}$, West Edge $115.875^\circ\text{E}$, East Edge $127.875^\circ\text{E}$, physically enclosing the entire Mindanao boundary ($[118.064^\circ\text{E}, 4.587^\circ\text{N}] \to [126.605^\circ\text{E}, 10.472^\circ\text{N}]$) with zero truncation.
  * Generated Artifacts:
    1. CF-1.8 NetCDF4 Grid: `processed/grid/mindanao_025deg.nc` (SHA-256: `51a691994a16b75b3a103efdb539febce2695179dbb89f5cb346a477e694548f`).
    2. CDO Grid Specification: `processed/grid/mindanao_0.25_grid.grd` (SHA-256: `b8fabe5c21a6b5beda2df8b9ef4438d8b63d74427e86223137a55c3afff40049`).
  * Audit Dossier: Formal certification and coordinate tables recorded in `reproduction_audit/07_mindanao_candidate_a_grid_freeze_audit.md`.
  * Pass Criterion: Certified PASS on coordinate definition, boundary containment, and NetCDF construction.
- [x] **Step 21A.7A**: Fractional Boundary-Coverage Mask (`processed/grid/mindanao_fraction_025.nc`).
  * Formulation: Compute the fractional overlap $f \in [0.0, 1.0]$ of each frozen 0.25° grid cell with the validated Mindanao administrative analysis boundary (`mindanao_analysis_boundary.gpkg`, SHA-256: `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2`), acknowledging this represents regional boundary coverage rather than a physical land-cover mask. [PASSED: 2026-09-10]
  * Geodetic Rigor: Computed via WGS 84 ellipsoidal geodesic quadrature (`pyproj.Geod`). Discrete integrated fractional cell area ($99,948.76\text{ km}^2$) matches authoritative boundary area ($99,948.76\text{ km}^2$) with area-conservative fractional discretization parity ($\Delta A = 0.0000\%$).
  * Benchmark Inspection: Evaluated 5 benchmark cells: Bukidnon interior ($f = 1.000000$), Cape San Agustin coastal tip ($f = 0.028912$), Surigao coastal land ($f = 0.556811$), Celebes ocean buffer ($f = 0.000000$), Dinagat archipelagic chain ($f = 0.228202$), and Jolo Island ($f = 0.596937$).
  * Output Artifact: `processed/grid/mindanao_fraction_025.nc` (SHA-256: `9294a1c74ec5aa2288c9e9bf54fc6763f905dbda1c4ff0ecda0a9b9ae9ff6325`).
- [x] **Step 21A.7B**: Binary Evaluation Mask (`processed/grid/mindanao_eval_mask_025.nc`).
  * Formulation: Explicit fractional inclusion rule: a grid cell is included in the evaluation mask if and only if fractional boundary coverage $f \ge 0.50$ ($M_{i,j} \in \{0, 1\}$), defining the binary evaluation domain for loss calculation and metric verification. [PASSED: 2026-09-10]
  * Cell Census Accounting: 126 binary evaluation cells ($8.20\%$ of $32 \times 48 = 1,536$ grid), with full-cell footprint of $96,085.57\text{ km}^2$, representing $86,418.83\text{ km}^2$ or $86.46\%$ of the $99,948.76\text{ km}^2$ authoritative boundary. 1,410 zero-padded buffer/ocean cells ($91.80\%$). Total active footprint ($f > 0$) is 253 cells ($16.47\%$), with 127 sub-threshold coastal/islet cells capturing the remaining $13.54\%$ ($13,529.93\text{ km}^2$) of the boundary.
  * Preservation Contract: Retained BOTH `mindanao_fraction_025.nc` (area-weighted sensitivity analyses) and `mindanao_eval_mask_025.nc` (loss and metric masking).
  * Output Artifact: `processed/grid/mindanao_eval_mask_025.nc` (SHA-256: `d7fd80e1f95cdd840daded176b06b2a0d23557df555acfaf0c01d2d1964621f6`).
  * Audit Dossier: Formal certification and census tables recorded in `reproduction_audit/08_mindanao_evaluation_mask_audit.md`.
  * Pass Criterion: Certified PASS on geodetic quadrature, benchmark inspection, and cell census.
- [x] **Step 21A.8**: Freeze Machine-Readable Spatial Contract.
  * Storage: `contracts/spatial/spatial_grid_contract.yaml` and `metadata/grid_definition.yaml`. [PASSED: 2026-09-10]
  * Contract Governance: Formally locks domain ID `Candidate_A_Mindanao_32x48`, coordinate conventions (cell centers: lat $11.75 \to 4.00$, lon $116.00 \to 127.75$), outer bounding box ($[115.875, 3.875] \to [127.875, 11.875]$), inclusion threshold ($f \ge 0.50$), active boundary footprint (253 cells), binary evaluation domain census (126 cells: full-cell footprint of $96,085.57\text{ km}^2$, representing $86,418.83\text{ km}^2$ or $86.46\%$ of the $99,948.76\text{ km}^2$ authoritative boundary), and SHA-256 hashes of all boundary, grid, CDO descriptor, mask, and generator script artifacts.
  * Pass Criterion: Certified PASS on machine-readable spatial contract freeze.
- [x] **Step 21A.9**: ERA5-Land RZSM Target Acceptance & Independent Cross-Reference.
  * Target Formulation Acceptance: Certified depth-weighted 0–100 cm volumetric soil water index ($\text{RZSM}_{0-100} = 0.07\,\text{swvl}_1 + 0.21\,\text{swvl}_2 + 0.72\,\text{swvl}_3$), strictly matching Kyle Lesinger's author code (`Data/raw_downloads/ERA5_real/process_soil.sh`). [PASSED: 2026-09-10]
  * Operational Dataset Standard: The A0 production RZSM cube uses the hourly ERA5-Land reanalysis as the source-of-record and performs explicit daily aggregation within the project pipeline. The Copernicus derived daily-statistics product is not the source of the certified 21D.4 production cube.
  * Dataset Splits Reconciled: Formally codified split architecture: **Nominal Training: 2015–2021** (7 years), **Validation: 2022–2023** (2 years), **Held-Out Test: 2024–2025** (2 years). Late 2014 data functions strictly as antecedent support data and is not part of the nominal training period.
  * Exact Antecedent Derivation: Verified that the EX29 trailing 14-day rolling window (`[-1, -7, -14]` days) requires daily observations extending exactly 20 calendar days prior to forecast issuance. For an earliest forecast issue date of 2015-01-01, the verified EX29 lag construction requires observations beginning 12 December 2014. The exact production requirement remains provisional on the finalized case calendar in Sub-Phase 21G, eliminating any requirement for full-year 2014 data.
  * Coordinate Alignment & Masking: Verified Candidate A CDO grid specification compatibility (`mindanao_0.25_grid.grd`), native land-aware linear spatial remapping in the pipeline, and confinement of loss/verification strictly to the 126 binary evaluation cells ($f \ge 0.50$, full-cell footprint: $96,085.57\text{ km}^2$, capturing $86,418.83\text{ km}^2$ or $86.46\%$ of the authoritative boundary).
  * Independent Cross-Reference Framework: GLEAM is established as an **independent cross-reference**, not a gold-standard acceptance oracle. Evaluation compares correlation, bias, RMSE, anomaly correlation, seasonal behavior, spatial agreement, and drought-event agreement, documenting discrepancies without arbitrary cutoffs.
  * Audit Dossier: Formal certification recorded in `reproduction_audit/12_era5_land_target_acceptance_audit.md`.
  * Pass Criterion: Technical target acceptance: PASS. Independent GLEAM suitability assessment: planned/ongoing validation framework, not a blocking prerequisite for using ERA5-Land unless predefined evidence later indicates a material inconsistency.

---

#### Sub-Phase 21B: ERA5-Land Soil Moisture Pilot Preprocessing & Verification
**Objective**: Audit existing 2015–2025 ERA5-Land archive integrity, lock the 20-day missing 2014 antecedent data specification (provisional on finalized case calendar), verify end-to-end depth-weighted integration on pilot data, execute land-aware linear spatial remapping to the frozen 0.25° grid, apply the binary evaluation mask, and validate physical distributions.

- [x] **Step 21B.1**: Audit cataloged 2015–2025 ERA5-Land archive integrity and completeness.
  * Inventory Scope: 11 full calendar years (2015–2025), 132 monthly pairs (264 files total, 3.67 GiB) covering volumetric soil water layers 1, 2, 3 (`swvl1`, `swvl2`, `swvl3`). [PASSED: 2026-09-10]
  * Diagnostic Audit: 100% monthly completeness across all 11 years (0 missing months); file sizes scale strictly with days per month (28d: ~23.88 MB, 29d: ~24.48 MB, 30d: ~24.99 MB, 31d: ~26.79 MB), confirming full 24h/day unclipped retention including leap days.
  * Audit Dossier: Formal certification recorded in `reproduction_audit/10_era5_land_archive_completeness_audit.md`.
  * Pass Criterion: Certified PASS (Full 144-month download averted; cleared for Step 21B.2 targeted 2014 antecedent specification).
- [x] **Step 21B.2**: Ingest, verify, and synchronize missing 2014 antecedent window.
  * Antecedent Scope: Strictly 12 December 2014 00:00 to 31 December 2014 23:00 UTC (20 calendar days, 480 hourly timesteps), resolving the EX29 trailing rolling window requirement for unclipped 1 January 2015 initiation (provisional on final case calendar). Full-year 2014 data excluded. [PASSED: 2026-09-11]
  * Retrieval & Ingestion: Retrieved `era5-land-2014-12-antecedent.nc` (6,185,331 bytes, SHA-256 `29292cf600a398ac61eabe111150e1aa3acf0f17356fb2da6a02c1dc92cd64fc`) covering all 3 volumetric soil moisture layers (`swvl1`, `swvl2`, `swvl3`) on the exact $0.10^\circ$ coordinate grid ($4.0^\circ\text{--}11.0^\circ\text{N} \times 116.5^\circ\text{--}127.5^\circ\text{E}$).
  * Production GCS Synchronization: Uploaded to primary thesis bucket `gs://rise-unet-rzsm/raw/era5_land/production/era5-land-2014-12-antecedent.nc`. Synchronized the full 2015–2025 archive (264 NetCDF files, 3.67 GiB at 511.2 MiB/s) to establish a 100% complete, self-contained archive of **265 synchronized ERA5-Land NetCDF files covering the 12 Dec 2014–31 Dec 2025 support window** in `gs://rise-unet-rzsm/raw/era5_land/production/`.
  * Pass Criterion: Certified PASS on antecedent window ingestion, variable completeness, and GCS synchronization.
- [x] **Step 21B.3**: Execute pilot RZSM depth-weighted integration and 0.25° land-aware linear spatial remapping.
  * Formulation: Calculate $\text{RZSM}_{0-100} = 0.07\,\text{swvl}_1 + 0.21\,\text{swvl}_2 + 0.72\,\text{swvl}_3$. [PASSED: 2026-09-11]
  * Remapping: Land-aware linear spatial interpolation (piecewise-linear Delaunay barycentric interpolation on valid finite land points with nearest-neighbor boundary fallback) to the frozen 0.25° reference grid via Candidate A geometry ($32 \times 48$). Ocean/buffer cells (1,410) are zero-filled. Remapping throughput: 2.12 s for 20 days (0.106 s/day).
  * Pass Criterion: Certified PASS (Output matches `(20, 32, 48)` tensor geometry with zero NaNs over active evaluation cells).
- [x] **Step 21B.4**: Verify coordinate alignment, spatial masking, and physical distributions.
  * Census & Distribution Audit: 20 daily dates (Dec 12–31, 2014), $20 \times 126 = 2,520$ valid evaluation samples; **Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational cells follow the frozen masking/zero-fill convention.** Volumetric soil moisture range $[0.2488, 0.5087]\text{ m}^3/\text{m}^3$ (mean: $0.4156\text{ m}^3/\text{m}^3$) conforms to tropical wet-season hydroclimatic physical bounds. [PASSED: 2026-09-11]
  * Artifacts & Cloud Synchronization: Preprocessed pilot NetCDF exported as `processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc` (130.6 KB, CF-1.8) tracked in Git with complete usage guide `processed/rzsm/pilot/README.md`. 4-panel 300 DPI publication composite generated at `figures/mindanao_rzsm_pilot_preprocessing_composite.png` (919 KB). GCS bucket `gs://rise-unet-rzsm/` fully synchronized with the 919 KB composite figure, auxiliary PAM XML (`mindanao_fraction_025.nc.aux.xml`), and pilot README.
  * Remote Colab Execution Record: Notebook 04 executed end-to-end on Google Colab with all 8 code cells recorded cleanly (`commit f2a346a`).
  * Audit Dossier: Formal certification recorded in `reproduction_audit/11_era5_land_pilot_rzsm_preprocessing_audit.md`.
  * Pass Criterion: Certified PASS (Sub-Phase 21B 100% complete; cleared for Sub-Phase 21C ERA5 atmospheric pilot acquisition).

---

#### Sub-Phase 21C: ERA5 Atmospheric Reanalysis Pilot Acquisition
**Objective**: Retrieve raw atmospheric reanalysis variables and derive the proposed/verified 21C atmospheric variable contract (5 model predictor channels) over Mindanao on the frozen Candidate A grid.

- [x] **Step 21C.1**: Execute 1-month pilot retrieval of raw ERA5 atmospheric variables. [PASSED: 2026-09-11]
  * Proposed/Verified 21C Atmospheric Variable Contract:
    - Raw ERA5 Single Levels: Analyzed hourly 2m temperature (`t2m`), 2m dewpoint temperature (`d2m`), surface pressure (`sp`), and total column water vapour (`tcwv`).
    - Raw ERA5 Pressure Levels: Analyzed geopotential (`z`) at 200 hPa.
  * Access Methods: Copernicus CDS API asynchronous client (`02_download_era5_atmospheric.py`).
  * Resolution & Pipeline Separation:
    - ERA5-Land (0.1° native) $\to$ RZSM depth-weighting $\to$ CDO bilinear remapping $\to$ 0.25° frozen grid (Sub-Phase 21B).
    - ERA5 Atmospheric (0.25° native) $\to$ atmospheric derivations $\to$ 0.25° Candidate A grid (direct coordinate co-location, Sub-Phase 21C).
  * Spatial Extent: Bounding box `[11.75°N, 116.0°E, 4.0°N, 127.75°E]` matching Candidate A $32 \times 48$ grid cell centers exactly.
  * Retrieval Certification: Retrieved `era5_single_levels_2015_01.nc` (7,276,966 bytes) and `era5_z200_2015_01.nc` (1,363,222 bytes) covering all 744 hourly timesteps of January 2015.
  * Storage: `Data/raw_downloads/ERA5_atmospheric/` (landing zone) with permanent data lake in `gs://rise-unet-rzsm/raw/era5/`.
  * Pass Criterion: Certified PASS (Raw atmospheric NetCDF files retrieved, 100% complete across 744 hours, conforming to bounds).
- [x] **Step 21C.2**: Implement atmospheric variable derivation and grid alignment. [PASS / VERIFIED / ACCEPTED: 2026-09-11]
  * Derived Model Channels (Distinction between Parent-Verified and Adaptation Decisions):
    1. Precipitable water (`pwat = tcwv`, [VERIFIED: Parent EX29 code `ERA_precipitable_water_obs`]).
    2. Specific humidity (`spfh`, derived from hourly `d2m` and `sp` via Bolton 1980 formulation, [VERIFIED: Parent EX29 thermodynamic variable]).
    3. Daily maximum temperature (`tmax = daily_max(hourly t2m)`, [ACCEPTED: Mindanao Implementation Decision] — operational adaptation aggregating hourly analyzed ERA5 surface temperatures to daily maximum).
    4. Diurnal temperature difference (`diff_temp = daily_max(hourly t2m) - daily_min(hourly t2m)`, [VERIFIED: Parent EX29 diurnal range contract]).
    5. Geopotential height at 200 hPa (`hgt_pres = z200 / 9.80665`, [VERIFIED: Parent EX29 geopotential conversion to gpm]).
  * Tensor Geometry & Missing Value Audit: All channels strictly conform to `(31, 32, 48)` (time $\times$ lat $\times$ lon). **Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational cells follow the frozen masking/zero-fill convention** across all $31 \times 126 = 3,906$ evaluation sample points.
  * Physical Distributions (Jan 2015):
    - `tmax`: $[294.28, 307.36]\text{ K}$ (mean: $301.65\text{ K}$)
    - `tmin`: $[288.22, 299.76]\text{ K}$ (mean: $295.88\text{ K}$)
    - `diff_temp`: $[0.47, 11.68]\text{ K}$ (mean: $5.77\text{ K}$)
    - `spfh`: $[0.0146, 0.0204]\text{ kg/kg}$ (mean: $0.0178\text{ kg/kg}$)
    - `pwat`: $[17.23, 59.94]\text{ kg/m}^2$ (mean: $44.80\text{ kg/m}^2$)
    - `hgt_pres`: $[12,411.5, 12,481.9]\text{ gpm}$ (mean: $12,442.8\text{ gpm}$)
  * Verified Artifacts: Exported pilot NetCDF `processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc` (1.17 MB) and 300 DPI composite figure `figures/mindanao_era5_atmospheric_pilot_verification.png` (1.27 MB publication edition).
  * Certification Status: [PASS: Software execution] + [VERIFIED: Variable schedule] + [ACCEPTED: Regional derivations].
- [ ] **Step 21C.3**: 11-Year ERA5 atmospheric archive mirroring (2015–2025). [TRAINING FOLD SECURED: 168/264 files (63.6%) + Dec 2014 Antecedent / Mirroring In Progress]
  * Objective: Retrieve complete hourly single-level and 200 hPa pressure-level fields across the 11-year nominal window via CDS-Beta API (`scripts/02_download_era5_atmospheric.py`) and mirror into Google Cloud Storage lake (`gs://rise-unet-rzsm/raw/era5/`).
  * Status: Exactly 168 of 264 monthly files (84 single-level + 84 pressure-level) covering January 2015 through December 2021 (84 months $\times$ 2 products = 168 files) are fully downloaded, verified, and mirrored to GCS (`gs://rise-unet-rzsm/raw/era5_atmospheric/`), securing 100% of the 7-year Model A0 training fold. The December 2014 antecedent support files (2 files: single-level and pressure-level) are tracked separately. The remaining 96 monthly files covering the validation and test partitions (2022–2025, 48 months $\times$ 2 products) are currently downloading via background Cloud Shell process. Full archive target is 264 monthly files.

---


#### Sub-Phase 21D: 0 to 100 cm RZSM Pipeline & Temporal Preprocessing
**Objective**: Build, unit-test, and verify the depth-weighted Root-Zone Soil Moisture calculation and author temporal preprocessing mechanics across separate pipelines for observations and dynamic forcings.

- [x] **Step 21D.1**: Implement ERA5-Land layer-integrated RZSM in `src/data/rzsm.py`. [PASS / ACCEPTED: 2026-09-11]
  * Formula: $\text{RZSM}_{0-100} = 0.07 \cdot \text{SM}_1 + 0.21 \cdot \text{SM}_2 + 0.72 \cdot \text{SM}_3$ ([ACCEPTED: Mindanao A0 Target Definition]).
  * Functions Authored: `compute_depth_weighted_rzsm`, `compute_backward_rolling_mean` (7-day trailing, center=False), `extract_antecedent_lags` (offsets [-1, -7, -14]), `apply_land_mask` (zero-filling ocean/buffer), `compute_min_max_scale` ([0, 1] scaling), and `remap_era5_land_to_candidate_a`.
  * Certification Status: [PASS: Vectorized NumPy/xarray implementation matching physical depth layer proportions].
- [x] **Step 21D.2**: Implement automated unit tests in `tests/test_02_rzsm.py`. [PASS: 2026-09-11]
  * Test Suite Scope: 10 automated unit tests across 5 test classes (`TestRZSMFormulation`, `TestTemporalPreprocessing`, `TestMaskingAndScaling`, `TestRealPilotDataIntegrity`, `TestLandAwareRemapping`).
  * Mathematical Accuracy: Verified depth integration to machine precision ($10^{-7}$), rolling mean step dynamics, lag slicing, zero-fill masking, and scaling properties.
  * Real Pilot Validation: Ran against `processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc` confirming 126 active cells, (20, 32, 48) shape, 0 NaNs, and physical bounds $[0.20, 0.60]\text{ m}^3/\text{m}^3$.
  * Test Results: 10/10 tests passed with 0 failures, 0 errors.
  * Certification Status: [PASS: Full unit test coverage with machine precision verification].
- [x] **Step 21D.3**: Implement distinct preprocessing transformation pipelines for inputs and targets. [PASS / VERIFIED / ACCEPTED: 2026-09-11]
  * **Modular Implementation**: Authored `src/data/temporal.py` and `src/data/compile_cube.py`, exported through `src/data/__init__.py`.
  * **Contract 1: Target Window Lead Offsets [VERIFIED: Parent EX29 Source-Code Parity]**:
    - S2S/SubX indexing convention: $L = (\text{lead} \times 7) - 1$ (`function/funs.py:L430-432`, `00_min_max_...:L244`).
    - Leads: $W_1 \to L=6$ (Days 0..6), $W_2 \to L=13$ (Days 7..13), $W_3 \to L=20$ (Days 14..20), $W_4 \to L=27$ (Days 21..27).
    - Rolling window: 7-day backward trailing mean (`center=False`).
    - Non-overlapping, contiguous 28-day partition: $[t_0, t_0+6] \cup [t_0+7, t_0+13] \cup [t_0+14, t_0+20] \cup [t_0+21, t_0+27]$.
    - Antecedent lags: $[-1, -7, -14]$ days (trailing rolling means ending at $t_0-1, t_0-7, t_0-14$).
  * **Contract 2: Climatology Definition [VERIFIED: Parent EX29 Source Code & ACCEPTED: Model A0 Baseline]**:
    - Model A0 strictly locks `climatology_method = "season"` (3-month seasons DJF, MAM, JJA, SON) fitted on training years $\le 2021$ (`preprocessUtils.py:L341-387`).
    - Methodological guardrail: Exploratory alternatives (such as daily DOY harmonic smoothing) are strictly quarantined to future Model A1 (Mindanao Enhancement).
  * **Contract 3: Min-Max Normalization Scope [VERIFIED: Parent Code & ACCEPTED: Adaptation Scope]**:
    - Four-part immutable scope:
      1. Training cases only ($2015 \le \text{year} \le 2021$).
      2. Evaluation cells only ($M_{i,j} = 1$, 126 active cells).
      3. Per-variable / per-lead channel.
      4. Domain-wide active scalar bounds (NOT pixel-wise per-cell bounds), matching Kyle Lesinger's parent code (`preprocessUtils.py:L739-757`: `train.sel(L=lead).max().compute()`).
      *(Note: The parent's implementation provides empirical evidence for the domain-wide scalar bound baseline; the argument that pixel-wise normalization would distort spatial soil moisture gradients is our scientific rationale/interpretation for why this baseline is well-suited for Mindanao's complex terrain)*.
  * **Contract 4: Precision Reporting Standard [ACCEPTED: Project Audit Standard]**:
    - Mandatory standard: **“Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational cells follow the frozen masking/zero-fill convention.”**
  * **Contract 5: Land-Aware Bilinear Remapping Specification [ACCEPTED: Mindanao spatial remapping implementation]**:
    - Source: ERA5-Land $0.10^\circ \times 0.10^\circ$ native grid.
    - Target: Candidate A $0.25^\circ \times 0.25^\circ$ ($32 \times 48$).
    - Remapping Order & Boundary Logic:
      1. Isolate finite land points on native ERA5-Land ($0.10^\circ$), ignoring native ocean NaNs.
      2. Bilinear interpolation (`scipy.interpolate.griddata(..., method="linear")`) evaluated over the 126 active evaluation cells ($M_{i,j}=1$).
      3. Nearest-neighbor extrapolation fallback (`NearestNDInterpolator`) triggered specifically for unassigned coastal boundary evaluation cells where the 4 surrounding points touch ocean nulls; no active-cell remapping gaps were observed, and coastal fallback was applied where required.
      4. Binary evaluation mask ($M_{i,j}=1$, 126 active cells) retains active values.
      5. Zero-fill strictly applied to all 1,410 inactive computational buffer/ocean cells.
  * **Automated Unit Test Verification Suite [PASS: 27 Tests Passed with Zero Failures and Zero Errors]**:
    - `tests/test_02_rzsm.py`: 10/10 tests passed (depth weighting, rolling mean, masking, scaling, real pilot, remapping).
    - `tests/test_03_temporal.py`: 9/9 tests passed (trailing invariance, training isolation $\le 2021$, zero-mean anomaly, parity).
    - `tests/test_06_target_reconciliation.py`: 3/3 tests passed (parent target construction == our target construction for hand-computable dates `2015-01-15`, `2015-06-01`, `2016-02-29`, `2018-08-15`, `2020-12-01`, manual arithmetic rolling verification, contiguous 28-day partition).
    - `tests/test_05_compile_cube.py`: 5/5 tests passed (full compilation pipeline, out-of-sample leakage isolation, census audit, `FastLandAwareRemapper` exact numerical agreement on tested data [observed maximum absolute difference = 0.0; test tolerance = 1e-6], antecedent real-data processing, partial compilation pipeline).
    - Execution Performance: **27 tests passed with zero failures and zero errors in 13.65s** (`python -m unittest discover -s tests -v`).
    - *Software Audit Conclusion & Coverage Calibration*: Step 21D.3 and the Step 21D.4 compilation engine satisfy all implemented software/numerical tests and explicitly reconciled parent-contract tests. Passing 27/27 automated unit tests establishes robust software implementation confidence; proof that the 11-year real data cube is correct is established separately by the Step 21D.4 production audit ([`reproduction_audit/14_production_cube_compilation_and_census_audit.md`](reproduction_audit/14_production_cube_compilation_and_census_audit.md)) via an exhaustive zero-tolerance numerical census across all 4,038 archive days and 506,268 nominal evaluation points.

- [x] **Step 21D.4-PREFLIGHT**: Production Cube Preflight Verification Gate. [PASSED: 2026-09-12]
  * Objective: Rigorous preflight checklist verifying all computational, contractual, and temporal prerequisites before launching the full 11-year data cube compilation:
    1. [x] **Input File Census**: Exactly 265 synchronized NetCDF files verified in `gs://rise-unet-rzsm/raw/era5_land/production/`.
    2. [x] **Archive Coverage**: 12 Dec 2014 to 31 Dec 2025 ($4,038$ continuous archive calendar days, 0 missing dates).
    3. [x] **Variable Completeness**: Soil moisture layers `swvl1`, `swvl2`, `swvl3` confirmed present across all archive files.
    4. [x] **Hourly Timestamp Continuity**: Expected hourly records for every calendar day; 24 hourly timestamps per day ($4,038 \times 24 = 96,912$ hourly timestamps total across the support period); leap-day coverage (29-day February in 2016, 2020, 2024 = 696 hrs each) intact.
    5. [x] **Timestamp Hygiene**: Monotonic time coordinate verified with zero duplicate timestamps.
    6. [x] **Grid Geometry Compatibility**: Strict conformity to Candidate A cell centers (lat $11.75^\circ \to 4.00^\circ\text{N}$, lon $116.00^\circ \to 127.75^\circ\text{E}$, shape $32 \times 48$).
    7. [x] **Spatial Contract Consistency**: Frozen contract values, grid coordinates, dimensions ($32 \times 48$), active evaluation cell count (126 cells), and referenced artifact SHA-256 hashes strictly match existing spatial foundation artifacts.
    8. [x] **Training Period Isolation**: Nominal training fold strictly locked to 2015–2021 (7 full calendar years).
    9. [x] **Normalization Scope Verification**: Domain-wide scalar min-max bounds fitted strictly over active evaluation cells within 2015–2021.
    10. [x] **Antecedent Support Isolation**: 2014 data (20 days) strictly restricted to rolling memory initialization; never enters training statistics or nominal production cube.
    11. [x] **Output CF-1.8 NetCDF Specification**: Variable names (`rzsm_0_100_raw`, `rzsm_0_100_rolling_7d`, `rzsm_0_100_seasonal_anomaly`, `rzsm_0_100_normalized`), coordinate conventions, and global attributes frozen.
  * Audit Dossier: Formal certification recorded in [`reproduction_audit/13_production_cube_preflight_audit.md`](reproduction_audit/13_production_cube_preflight_audit.md).
  * Interactive Asset Prepared: [`notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb`](notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb) (16 cells, Colab badge).
  * Pass Criterion: Certified PASS (All 11 preflight checks satisfied; cleared for Step 21D.4 production cube compilation).
- [x] **Step 21D.4**: Full production 11-year RZSM data cube compilation (2015–2025 with late-2014 antecedent support). [PASSED / VERIFIED / ACCEPTED: 2026-09-12]
  * **Temporal Partitioning & Explicit Scope Distinction**:
    - **Raw Support Archive**: 12 Dec 2014 – 31 Dec 2025 ($4,038$ calendar days total, $4,038 \times 126 = 508,788$ evaluation-cell-day samples across the **265 synchronized ERA5-Land NetCDF files covering the 12 Dec 2014–31 Dec 2025 support window** in `gs://rise-unet-rzsm/raw/era5_land/production/`).
    - **Production RZSM Cube**: 01 Jan 2015 – 31 Dec 2025 ($4,018$ calendar days, $4,018 \times 126 = 506,268$ active evaluation-cell-day samples across 11 calendar years, accounting for leap years 2016, 2020, and 2024: $8 \times 365 + 3 \times 366 = 4,018$).
    - **Purpose of 2014 Support**: Strictly antecedent initialization support only; provides the required historical window so that earliest nominal 2015 observations have complete 7-day trailing rolling memory. Those 20 support days (12–31 Dec 2014 = 20 days) are NOT independent production observations or training/evaluation data.
  * **Execution Strategy**: High-RAM Google Colab pipeline via `notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb` backed by automated local production module `src/data/compile_cube.py`.
  * **Verified Processing Sequence**:
    ```text
    12 Dec 2014 ─────────────── 31 Dec 2025 (4,038 archive days)
              ↓
    Depth-weighted RZSM construction (0.07*SM1 + 0.21*SM2 + 0.72*SM3)
              ↓
    Land-aware 2D bilinear remapping + coastal extrapolation fallback to 0.25° (32 x 48)
              ↓
    Continuous 7-day backward trailing rolling transformation (center=False, zero future leakage)
              ↓
    3-month seasonal climatology (DJF, MAM, JJA, SON, training fold <= 2021) & anomaly subtraction
              ↓
    Domain-wide active scalar min-max normalization (training active cells only)
              ↓
    Slice nominal production timeline: 01 Jan 2015 ─────────────── 31 Dec 2025 (4,018 days)
              ↓
    Production cube (506,268 nominal evaluation cell-days, 0 NaNs/Infs across 126 active cells)
    ```
  * **Validation & Certified Census**:
    - Nominal Timesteps: Exactly 4,018 days ($2015\text{--}2025$).
    - Active Cells: Exactly 126 cells.
    - Total Evaluations: $506,268$ evaluation cell-days.
    - NaNs across Active Cells: Strictly 0.
    - Infs across Active Cells: Strictly 0.
    - Ocean Buffer Cells: Strictly 0 nonzero values across 1,410 inactive cells.
    - Standardized Distribution: Min $0.0000$, Max $1.0000$, Mean $0.5205$, Median $0.5295$.
  * **Cloud Lake & Codebase Dual Parity**:
    - NetCDF Data Cube: `processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc` and `gs://rise-unet-rzsm/processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc` ($8.40\text{ MiB}$).
    - Verification Figure: `figures/mindanao_production_cube_verification_composite.png` and `gs://rise-unet-rzsm/figures/mindanao_production_cube_verification_composite.png` ($423,468\text{ bytes}$).
  * **Audit Dossier**: Certified in [`reproduction_audit/14_production_cube_compilation_and_census_audit.md`](reproduction_audit/14_production_cube_compilation_and_census_audit.md).
  * **Pass Criterion**: Zero NaNs/Infs across all 506,268 nominal evaluation points; 1,410 ocean cells zero-filled; full CF-1.8 NetCDF metadata. [CERTIFIED PASS]

---

#### Sub-Phase 21E: ECMWF S2S Reforecast Pilot Acquisition & Harmonization Engine
**Objective**: Ingest and harmonize ECMWF S2S subseasonal reforecasts using the verified parent EX29 dynamic triplet (`t2m`, `d2m`, `tcw`), with safe production defaults and zero-tolerance missing-step validation.

- [x] **Step 21E.1**: Authenticate and configure ECMWF Data Store (ECDS) API client. [PASS: 2026-09-12]
  * Dataset: Dedicated ECMWF S2S reforecast dataset (`s2s-reforecasts`).
  * Pass Criterion: Valid API connection and research authorization confirmed with unified CDS access token.
- [x] **Step 21E.2**: Retrieve single pilot issue cycle for available historical reforecast date. [PASS: 2026-09-12]
  * Issue Date: Selected based on verified Monday/Thursday model cycle calendar in ECDS.
  * Variables: Verified parent EX29 dynamic triplet: 2m temperature (`t2m`), 2m dewpoint temperature (`d2m`), and total column water (`tcw`). *(Total precipitation `tp` is explicitly excluded from S2S ingestion, adhering strictly to the author's EX29 ECMWF dynamic triplet contract)*.
  * Ensemble Realization: 11 ensemble members (Member 0 Control + Members 1–10 Perturbed) corresponding to the parent EX29 configuration.
  * Storage: `gs://rise-unet-rzsm/raw/ecmwf_s2s/production/`.
- [x] **Step 21E.3**: Harmonize and convert S2S GRIB to 0.25° grid xarray dataset with Safe Production Defaults. [PASS / VERIFIED: 2026-09-13]
  * Dimensions: `(lead=2, member=11, lat=32, lon=48)`.
  * **Lead Step Bins & Author Parity**:
    - Week 1 ($W_1$): daily forecast leads $L=0\text{--}6$, corresponding to forecast steps $0 \le \text{step} < 168\text{ h}$ (hours $0, 24, 48, 72, 96, 120, 144$ for daily resolution).
    - Week 2 ($W_2$): daily forecast leads $L=7\text{--}13$, corresponding to forecast steps $168 \le \text{step} < 336\text{ h}$ (hours $168, 192, 216, 240, 264, 288, 312$ for daily resolution).
    - Prevents double-counting step 168 and enforces strict mathematical symmetry across weekly windows.
  * **Production Safety Contract (`allow_step0_fallback=False`)**:
    - In production mode (`allow_step0_fallback=False`, default), missing required forecast steps raise a hard `ValueError`, immediately rejecting defective cases. Step-0 fallback is strictly forbidden in production.
    - Pilot/debug mode (`allow_step0_fallback=True`) is restricted to debugging legacy files, emits an explicit visible warning, and flags `attrs["contains_step0_fallback"] = True`.
  * **Date Semantics Preservation**:
    - Historical reforecast issue date (`hdate`, from Section 1) and operational model version date (`model_version_date` / `date`, from Section 4 PDT 60/61) are separately parsed and preserved in dataset metadata.
  * **Spatial Verification**:
    - Active evaluation cells ($N=126$) span $[5.75^\circ\text{N}, 9.75^\circ\text{N}]$ and $[121.00^\circ\text{E}, 126.50^\circ\text{E}]$, strictly bounded within native S2S coverage ($[4.5^\circ\text{N}, 10.5^\circ\text{N}]$, $[117.0^\circ\text{E}, 127.5^\circ\text{E}]$).
    - **Exactly 0 active evaluation cells require extrapolation**. Outer non-evaluation buffer cells ($461$ cells) are filled and zero-masked.
  * Transformations: Pure-Python GRIB2 Section 7 Template 0 decoding in `src/data/s2s.py`; bilinear spatial remapping to Candidate A 0.25° grid with ocean cell zero-filling.
  * Test Suite: Unit tests in `tests/test_04_s2s.py` verifying analytical decoding, monotonic gradient preservation, remapping dimensions, physical ranges, production hard-fail enforcement, and clean ECDS production parsing.
  * Storage & Dual Parity: Exported `processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc` (0.42 MB) and synchronized to `gs://rise-unet-rzsm/processed/s2s/pilot/`.
  * Certification: Formally certified in [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](reproduction_audit/16_ecmwf_s2s_pilot_and_recursive_inference_audit.md).

---

#### Sub-Phase 21F: Construct Single Complete EX29-Derived A0 Case
**Objective**: Assemble the complete multi-lead input-target tensor hierarchy for a single forecast cycle adhering strictly to the EX29 channel contracts, and execute the recursive forward inference gate.

- [x] **Step 21F.1**: Assemble lead-specific input tensors at issue time $\tau$. [PASS / VERIFIED: 2026-09-13]
  * Ensemble & Batch Distinction: Single-case tensor shape is $[M=11, H=32, W=48, C_k]$ across the $M=11$ ensemble members; mini-batch training dimension $B$ is handled separately during data pipeline construction.
  * **Lead 1 ($X_{W1}$)**: 11 channels $\to$ 3 antecedent weekly RZSM lag features, 5 ERA5 atmospheric (`pwat`, `spfh`, `tmax`, `diff_temp`, `hgt_pres`), 3 ECMWF S2S W1 predictions (`t2m`, `d2m`, `tcw`).
  * **Lead 2 ($X_{W2}$)**: 12 channels $\to$ 11 base channels + recursive W1 prediction channel $\hat{y}_{W1}$.
  * **Lead 3 ($X_{W3}$)**: 5 channels $\to$ 3 antecedent weekly RZSM lag features + recursive $\hat{y}_{W1}$ + recursive $\hat{y}_{W2}$.
  * **Lead 4 ($X_{W4}$)**: 6 channels $\to$ 3 antecedent weekly RZSM lag features + recursive $\hat{y}_{W1} + \hat{y}_{W2} + \hat{y}_{W3}$.
  * **Authoritative Antecedent Lag Contract**:
    - Expressed strictly in terms of rolling-window endpoint offsets: `[-1, -7, -14]` days.
    - Lag 1 (`-1`): $[t_0-7\text{d}, t_0-1\text{d}]$ (trailing 7 days).
    - Lag 2 (`-7`): $[t_0-13\text{d}, t_0-7\text{d}]$ (trailing 7 days).
    - Lag 3 (`-14`): $[t_0-20\text{d}, t_0-14\text{d}]$ (trailing 7 days).
  * Implementation: Vectorized NumPy/xarray case assembler authored in `src/data/case_builder.py` (`assemble_single_a0_case`), preserving `hdate` and `model_version_date` in `CaseTensorHierarchy`.
  * Pass Criterion: Zero future data leakage; tensor shapes $[M=11, H, W, C_k]$ strictly match EX29 specification.
- [x] **Step 21F.2**: Assemble ground truth target tensors $Y$ for leads W1 to W4. [PASS / VERIFIED: 2026-09-13]
  * Target Representation: Ground-truth target is represented once per forecast case ($[1, 32, 48, 1]$ per lead for $W_1..W_4$) and aligned/broadcast to the 11-member prediction structure only at the loss interface, according to the verified `crps2d_tf` contract.
  * **Verified Parent Target Index Contract**:
    - The parent EX29 implementation defines verification targets via 7-day backward trailing rolling means sampled at lead offsets $L = (\text{lead} \times 7) - 1$, yielding lead offsets `[6, 13, 20, 27]` days from $t_0$.
    - For issue date $t_0$, Week 1 target ($L=6$) covers $[t_0, t_0+6\text{d}]$ (Days 0–6); Week 2 target ($L=13$) covers $[t_0+7\text{d}, t_0+13\text{d}]$ (Days 7–13); Week 3 target ($L=20$) covers $[t_0+14\text{d}, t_0+20\text{d}]$ (Days 14–20); Week 4 target ($L=27$) covers $[t_0+21\text{d}, t_0+27\text{d}]$ (Days 21–27).
    - For pilot cycle $t_0 = \text{2015-01-16}$, verification target dates are strictly: W1: `2015-01-22`, W2: `2015-01-29`, W3: `2015-02-05`, W4: `2015-02-12`.
    - For cycle $t_0 = \text{2015-01-15}$, verification target dates are strictly: W1: `2015-01-21`, W2: `2015-01-28`, W3: `2015-02-04`, W4: `2015-02-11`.
  * Pass Criterion: Shape $[1, H, W, 1]$ per lead (broadcast to $[M=11, H, W, 1]$ at the loss layer).
  * Automated Unit Tests: Automated tests in `tests/test_10_case_builder.py` verifying synthetic and real pilot data assembly (`2015-01-15` and `2015-01-16` cases), zero-leakage isolation, ocean zero-filling, and explicit `target_dates` validation.
  * Full Suite Status: **38/38 automated unit tests passing** across the repository in 13.99s.
- [x] **Step 21F.3**: Real ECMWF Case, ecCodes Equivalence Gate & TensorFlow Recursive Forward Pass. [PASS / VERIFIED / ACCEPTED: 2026-09-13]
  * Implementation: Google Colab interactive notebook `notebooks/07_mindanao_s2s_and_pilot_case_assembly.ipynb` and dual 2x2 publication dashboards at 300 DPI (mirrored to `gs://rise-unet-rzsm/figures/`):
    - **Canvas 1 (`figures/mindanao_s2s_dynamic_predictor_composite.png`)**: ECMWF S2S Multi-Scale Dynamic Predictor Dashboard. Resolves ECMWF abstractness by superimposing discrete sampling nodes ($N=40$), vector PSA/NAMRIA coastline overlay, continuous bilinear remapped regional field ($32 \times 48$), 11-member ensemble spread ($\sigma_{\text{ENS}}$), and subseasonal dynamic moisture shift ($\Delta_{W2-W1}$).
    - **Canvas 2 (`figures/mindanao_s2s_pilot_case_and_recursive_inference.png`)**: Model A0 Case Assembly & Recursive Sensitivity Dashboard. Visualizes antecedent RZSM memory lag -1d over 126 evaluation cells, ERA5 precipitable water ($PWAT$ at $t_0$), observed ground truth target $Y_{W1}$ (Lead +6d, `2015-01-22`), and empirical downstream perturbation sensitivity response map $\Delta_{W2}$ in sequential `inferno` colormap certifying recursive feedback sensitivity.
  * Scope & Acceptance Criteria:
    1. **Real Clean ECDS Case Ingestion**: Ingest real clean ECDS production cycle (`2015-01-16`) under strict `allow_step0_fallback=False` (zero fallbacks, complete 14 TCW steps, 11 members, `Contains Fallback Flag = False`). [PASS]
    2. **Scoped Trusted-Decoder Equivalence Gate**: The custom pure-Python decoder reproduced official ecCodes (v2.48.2) values with near-exact machine precision across all 462 tested real messages in the 2015-01-16 production cycle using GRIB2 Template 7.0 simple packing ($\max |\Delta| = 3.81 \times 10^{-6} \le 10^{-5}$, max relative diff $= 5.95 \times 10^{-8}$). [PASS / VERIFIED]
    3. **TensorFlow UNET_RZSM Recursive Inference**: Execute the 4-lead recursive inference loop ($\hat{y}_{W1} \to X_{W2}, \hat{y}_{W2} \to X_{W3}, \hat{y}_{W3} \to X_{W4}$) on GPU in 2.762s ($690.5$ ms/lead across 11 ensemble members) using the authentic 1.63M-parameter architecture. [PASS]
    4. **Functional Computational Graph Sensitivity**: Demonstrate empirically that perturbations to $\hat{y}_{W1}$ ($\delta = +0.05$) produce measurable downstream response in $\hat{y}_{W2}$ ($\max |\Delta_{W2}| = 80.97$, Mean $= 4.99$, RMS $= 12.38$, 0 NaNs), confirming functional end-to-end graph connectivity across recursive unrolling (recursive degradation is separately evaluated in Phase 23). [PASS]
    5. **Zero-Tolerance Quality Census**: Zero NaNs, zero Infs across the 126 active evaluation cells; non-evaluation ocean buffer cells strictly zero-filled ($0.00 \times 10^0$). [PASS]
  * Formal Audit Dossier: Certified in [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](reproduction_audit/16_ecmwf_s2s_pilot_and_recursive_inference_audit.md).
  * Milestone Outcome: Certified PASS across all 5 verification criteria clears Sub-Phase 21F for **Case Assembly & Computational Integrity** and unlocks downstream execution, while model forecasting skill and trained Model A0 performance remain explicitly deferred to Sub-Phases 21H–21K.



---

#### Sub-Phase 21G: 5 to 10 Case Pilot Ladder & Manifest Generation
**Objective**: Verify pipeline stability across successive forecast issue cycles and establish the single source of truth for case tracking.

- [x] **Step 21G.1**: Build case manifest `manifests/cases_pilot_v001.csv`. [PASS / VERIFIED: 2026-09-13]
  * **Preflight Complete Intersection Audit**: Evaluated 21 candidate cycles across early 2015. Automatically rejected 4 cycles (`2015-01-01` to `2015-01-11`) whose 14-day antecedent lags preceded the 2015-01-01 production cube boundary, and identified 17 fully valid candidate cycles.
  * **Selected 8-Case Ladder**: Assembled 8 consecutive weekly cycles from January 15 to March 4, 2015 (`2015-01-15`, `2015-01-22`, `2015-01-29`, `2015-02-05`, `2015-02-12`, `2015-02-19`, `2015-02-26`, `2015-03-04`). In early March 2015, the next available operational ECMWF reforecast cycle occurred on Wednesday `2015-03-04` rather than Thursday `2015-03-05`, and both `2015-02-26` (Case 7) and `2015-03-04` (Case 8) are complete and included in the pilot ladder.
  * **Dual Manifest Emission**:
    - Member-level 88-row manifest (`manifests/cases_pilot_v001.csv`): 8 cycles $\times$ 11 ensemble members (0 CF, 1..10 PF), tracking individual member source URIs, split (`TRAIN`), target dates, and SHA-256 checksums.
    - Case-level 8-row summary manifest (`manifests/cases_pilot_summary_v001.csv`): Provides cycle-level file sizes, tensor shapes, and verification statuses.
  * Pass Criterion: 8 consecutive weekly cases indexed with full data-lineage provenance, 0 missing members, and zero target shifts ($L=[6, 13, 20, 27]$). [PASS]
- [x] **Step 21G.2**: Batch-generate pilot cases and store in GCS. [PASS / VERIFIED: 2026-09-13]
  * Batch Generation: Serialized all 8 cases into compressed NumPy archives (`processed/cases/pilot/CASE_*.npz`, ~122 KB each).
  * Mathematical Quality Census: Exactly **314,496 feature values** audited across 8 cases ($8 \text{ cases} \times 312 \text{ channels} \times 126 \text{ binary evaluation-domain cells}$), with exactly 0 NaNs and 0 Infs. All 1,410 ocean buffer cells strictly zero-filled ($0.00e-00$).
  * Cloud Lake Mirroring: All 8 case NPZ files and both manifest CSVs mirrored to `gs://rise-unet-rzsm/processed/cases/pilot/` and `gs://rise-unet-rzsm/manifests/`.
  * Independent 16-Point Audit: Validated by `scripts/10_verify_pilot_ladder_provenance_and_census.py` checking CF/PF source files, 11 distinct ensemble members, variable completeness, non-fallback production enforcement, hdate/model date fidelity, rolling targets, SHA-256 hashes, and local/GCS parity.
  * Automated Unit Tests: Added `tests/test_09_pilot_ladder.py`; all **42/42 unit tests passing** across repository in 28.2s.
  * Formal Audit Dossier: Certified in [`17_pilot_case_ladder_and_manifest_audit.md`](reproduction_audit/17_pilot_case_ladder_and_manifest_audit.md).
  * Milestone Outcome: Sub-Phase 21G certified `[PASS / VERIFIED] — Pilot Ladder Manifest & Pipeline Stability`. Unlocks Sub-Phase 21H (`tf.data` Pipeline & Checkpoint Test).

---

#### Sub-Phase 21H: Surrogate TensorFlow Pipeline & Checkpoint Infrastructure Smoke Test
**Objective**: Validate `tf.data.Dataset` streaming from GCS, mini-batch loss computation respecting ensemble grouping semantics, backpropagation, and state persistence using a lightweight 2-layer Conv2D surrogate architecture (did NOT instantiate genuine UNET_RZSM).

- [x] **Step 21H.1**: Implement high-throughput `tf.data` pipeline streaming from GCS. [PASS / VERIFIED: 2026-09-13]
  * Ensemble Grouping Constraint: Batch construction strictly enforces and tests the 11-member ensemble grouping invariant required by the verified `crps2d_tf` implementation; exact-multiple-of-11 batching ($B \in 11\mathbb{Z}^+$) across the complete set $\{11, 22, \dots, 110\}$ is enforced via `validate_batch_size(B)` with hard rejection on invalid sizes.
  * Shuffling & Ordering Semantics: Cycle-level shuffling preserves internal 11-member realization correspondence; member ordering is verified across the complete chain ($\text{NPZ} \to \text{generator} \to \text{tf.data} \to \text{batch} \to \text{model} \to \text{loss}$) with zero cross-case leakage or interleaving.
  * Target Alignment & Broadcasting Invariance: Verification targets $Y_{W_k} \in \mathbb{R}^{1 \times 32 \times 48 \times 1}$ are broadcast across all 11 members ($\mathbb{R}^{11 \times 32 \times 48 \times 1}$) at the loss interface; CRPS evaluated on single-target vs broadcast-target is empirically invariant with $0.00 \times 10^0$ discrepancy.
  * CRPS Mathematical Reconciliation: Independently verified against the exact analytical pairwise difference formula ($\text{CRPS}_{\text{exact}} = 0.045455$) and the author's spatial proxy ($\mathcal{L}_{\text{author}} = 0.123715$) on a deterministic toy ensemble.
  * Implementation: Modular batch generator and pipeline engine in `src/data/tf_dataset.py` with multi-head deep supervision outputs (`RZSM_output_1`, `RZSM_output_2`, `RZSM_output_3`).
  * Pass Criterion: Certified PASS (Automated unit tests in `tests/test_11_tf_dataset.py` verify $B=11$ and $B=22$ streaming with zero NaNs/Infs).
- [x] **Step 21H.2**: Execute 5-epoch training loop with Adam optimizer and CRPS loss. [PASS / VERIFIED FOR SURROGATE INFRASTRUCTURE: 2026-09-13]
  * Execution: Tested via `scripts/13_train_a0_pipeline_checkpoint.py` and interactive Colab notebook `notebooks/08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb` on NVIDIA Tesla T4 GPU using a 2-layer Conv2D surrogate model.
  * Loss & Gradient Verification: CRPS loss evaluates to strictly finite numbers across all epochs; on physical Colab GPU, mean CRPS loss decreased from $0.2233 \to 0.0748$ (66.5% reduction) with gradient norms in $[0.0468, 0.1637]$, non-zero weight updates occur across all layers, and zero gradient explosion.
  * Calibration & Scope: Certified strictly as an infrastructure, data-feeding, and numerical-stability smoke test (NOT evidence of predictive learning or genuine A0 forecasting skill).
  * Pass Criterion: Certified PASS (Finite execution, stable backpropagation, non-zero weight updates).
- [x] **Step 21H.3**: Verify checkpoint saving and restoration from GCS. [PASS / VERIFIED: 2026-09-13]
  * Checkpoint Artifacts: Exported `checkpoints/a0_pipeline_test/a0_test_epoch005.weights.*` and metadata JSON; Colab checkpoint serialized as `a0_tf_test_epoch005.weights.h5`.
  * Parity Assertion: Restoring checkpoint into a freshly initialized clean model instance yields $\max |\hat{Y}_{\text{original}} - \hat{Y}_{\text{restored}}| = 0.00 \times 10^0$ (exact bit-for-bit parity confirmed under both CPU and physical GPU runtimes).
  * Cloud Lake Mirroring: Checkpoint artifacts targeted for `gs://rise-unet-rzsm/checkpoints/a0_pipeline_test/`.
  * Pass Criterion: Certified PASS (Discrepancy $= 0.00 \times 10^0$ across all evaluation cells; unit tests passing).
  * Formal Audit Dossier: Certified in [`20_tf_dataset_pipeline_and_checkpoint_audit.md`](reproduction_audit/20_tf_dataset_pipeline_and_checkpoint_audit.md).

---

#### Sub-Phase 21I: Surrogate Tiny-Data Optimization Smoke Test
**Objective**: Empirically verify parameter updates and gradient descent mechanics on real Mindanao tensors using a lightweight 2-layer Conv2D surrogate model before hardware profiling and full training.

- [x] **Step 21I.1**: Train on 8 fixed Mindanao cases for 40 epochs. [PASS / VERIFIED FOR SURROGATE PURPOSE: 2026-09-14] (`CONDITIONAL GO → 21J`)
  * Model Topology: 2-layer Conv2D surrogate model (did NOT instantiate the genuine 1.63M-parameter UNET_RZSM).
  * Dataset & Budget: Exactly 8 fixed pilot forecast cycles (`CASE_20150115_W01.npz` to `CASE_20150304_W08.npz`) $\times$ 11 members = 88 samples; batch size $B=11$; exactly 320 parameter updates executed.
  * Optimizer & Loss: Adam ($\eta = 0.001$, $\beta_1=0.9, \beta_2=0.999$), deep supervision weighting ($1.0 \times \mathcal{L}_1 + 1.0 \times \mathcal{L}_2 + 1.0 \times \mathcal{L}_3$).
  * Training-Set Overfit Diagnostics (Historical Surrogate Diagnostic — Not Representative of A0 Training): Total loss reduced from $1.0791 \to 0.0105$ (99.03% reduction); active-cell MAE fell from $10.0671 \to 0.0428\,\text{m}^3/\text{m}^3$ (99.57% error reduction); active-cell target Euclidean distance dropped from $1060.08 \to 5.56$. As demonstrated by forensic audit, these massive numerical drops were scaling artifacts caused by raw unnormalized physical predictors (e.g., geopotential height $\sim 12,400\text{ gpm}$, temperature $\sim 300\text{ K}$) entering the surrogate, not proof of full RISE-UNet capacity.
  * Spatial Mask & Ocean Invariant: Evaluated over 126 active land cells; evaluation-domain postprocessing mask strictly sets $0.00 \times 10^0$ across all 1,410 ocean buffer cells (note: network graph outputs raw unmasked predictions; masking is an evaluation-domain step).
  * Checkpoint & Dispersion Diagnostics: Bit-for-bit restore parity verified ($0.00 \times 10^0$ error); per-case ensemble spread tracked diagnostically ($\bar{\sigma}_{\text{ens}}: 0.000325 \to 0.000387$, mean member correlation $r = 0.9847$).
  * Scope Limitation: Certified strictly as a surrogate optimization smoke test; provides zero claims of Model A0 architecture capacity, predictive skill, out-of-sample generalization, or probabilistic calibration.
- [x] **Step 21I.2**: Archive overfit verification dossier. [PASS / VERIFIED: 2026-09-14]
  * Storage: Formally audited and certified in [`reproduction_audit/22_a0_tiny_overfit_gradient_audit.md`](reproduction_audit/22_a0_tiny_overfit_gradient_audit.md) and [`logs/a0_tiny_overfit_execution.json`](logs/a0_tiny_overfit_execution.json).

---

#### Sub-Phase 21J: Genuine Model A0 (`UNET_RZSM`) Hardware Profiling & VRAM Feasibility Benchmark Gate
**Objective**: Empirically profile GPU VRAM footprint, host RAM pressure, forward/backward gradient dynamics, and epoch throughput for the authentic 1,630,307-parameter nested U-Net architecture (`UNET_RZSM`) on target NVIDIA Tesla T4 GPU hardware before committing to full multi-year production training (Sub-Phase 21K).

- [x] **Step 21J.PREFLIGHT**: Preflight Engineering, Unit Testing & Standalone Benchmark Architecture. [PASS / VERIFIED: 2026-09-14]
  * **Six Technical Pillars of 21J Gate**:
    1. **Pillar 21J.1 (Authentic Architecture Instantiation)**: Replaces miniature surrogate approximation with authentic Model A0 factory ([`src/models/a0_unet.py`](src/models/a0_unet.py)), directly delegating to Kyle Lesinger's `model_build_func` in `function/modelRzsmRelu.py`. Instantiates exactly 1,630,307 parameters across 298 trainable weight tensors for Lead 2, Inception blocks, Squeeze-and-Excitation attention, multiscale decoders, and 3 deep-supervision heads (`RZSM_output_1`, `RZSM_output_2`, `RZSM_output_3`).
    2. **Pillar 21J.2 (Real-Model Multi-Lead Forward Pass & Output Head Routing)**: Evaluates forward pass across all 4 lead channel schedules ($W_1=11, W_2=12, W_3=5, W_4=6$ channels) over Candidate A ($32 \times 48$). Routes primary inference to Stage 4 head (`preds[-1]`) and validates strict zero-filling on the 1,410 inactive ocean buffer cells.
    3. **Pillar 21J.3 (Real-Model Backward Pass & Gradient Stability)**: Executes full backpropagation using genuine multi-head CRPS loss ($\mathcal{L}_{\text{total}} = 1.0\mathcal{L}_1 + 1.0\mathcal{L}_2 + 1.0\mathcal{L}_3$) and Adam optimizer. Verifies strictly finite gradients ($\|\nabla_\theta\| > 0$, 0 NaNs/Infs) and measurable weight tensor updates across all 298 layers.
    4. **Pillar 21J.4 (4-Lead Autoregressive Recursive Cascade Compatibility)**: Evaluates the complete multi-lead recursive unrolling ($\hat{y}_{W1} \to X_{W2} \to \hat{y}_{W2} \to X_{W3} \to \hat{y}_{W3} \to X_{W4}$) on GPU, demonstrating non-zero downstream perturbation sensitivity ($\Delta_{W2} > 0$) without graph disconnection.
    5. **Pillar 21J.5 (VRAM Memory Ladder & Throughput Profiling)**: Profiles GPU memory allocation across candidate batch sizes $B \in \{11, 22, 33, 44\}$ (multiples of 11 preserving ensemble grouping). Establishes maximum safe batch size under the 15.0 GiB T4 VRAM budget.
    6. **Pillar 21J.6 (Production Contract Freeze & Checkpoint Serialization Parity)**: Verifies bit-for-bit checkpoint save/restore parity ($0.00 \times 10^0$ discrepancy), freezing the production training hyperparameters for Phase 21K.
  * **Automated Unit Test Suite**: Authored [`tests/test_12_a0_unet.py`](tests/test_12_a0_unet.py). All **83/83 unit tests passing** (82 passed, 1 skipped offline) across the repository.
  * **Benchmarking Artifacts**: Authored [`scripts/11_profile_a0_vram_benchmark.py`](scripts/11_profile_a0_vram_benchmark.py), preflight telemetry [`logs/A0_gpu_benchmark.json`](logs/A0_gpu_benchmark.json), and interactive Colab notebook [`notebooks/09_mindanao_a0_vram_profiling.ipynb`](notebooks/09_mindanao_a0_vram_profiling.ipynb).
  * **User Verdict**: **GO → Proceed to physical GPU execution of Notebook 09**.
- [x] **Step 21J.1**: Physical GPU Execution of 21J Six-Pillar Benchmark on Google Colab (Tesla T4 GPU). [PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]
  * **Execution Platform**: Google Colab NVIDIA Tesla T4 (15,360 MB VRAM), CUDA 12.5.1, cuDNN 9, TensorFlow 2.20.0, Python 3.13.15.
  * **Empirical Pillar Results**:
    1. *Pillar 21J.1 (Architecture)*: Instantiated genuine `UNET_RZSM` nested U-Net; Lead 1 = 1,627,139 params (11 channels); Lead 2 = 1,630,307 params (12 channels, exact parent EX29 parity); 298 trainable weight tensors, 132 non-trainable, 251 layers, 3 deep supervision heads. Reconciled per-lead contract in `EXPECTED_A0_PARAMETER_COUNTS`. [PASS]
    2. *Pillar 21J.2 (Multi-Lead Forward)*: Output shapes $(11, 32, 48, 1)$ across leads $1 \dots 4$; latencies 279.1 ms, 318.9 ms, 291.6 ms, 417.4 ms; evaluation-domain postprocessing mask strictly sets ocean buffer to $0.00 \times 10^0$ (UNET_RZSM outputs unmasked predictions). [PASS]
    3. *Pillar 21J.3 (Backward Pass)*: Multi-head representative MAE loss $= 2.8247$, global gradient norm $= 3.6176$, all 298 gradient tensors populated, weight delta $\|\Delta w\| = 3.95 \times 10^{-3} > 0$. Replicates parent executable gradient behavior. [PASS]
    4. *Pillar 21J.4 (4-Lead Cascade)*: Cascade latency $= 1112.82\text{ ms}$ ($278.20\text{ ms/lead}$); downstream perturbation sensitivity verified ($\Delta_{W1}=0.50 \implies \Delta_{W2}, \Delta_{W3}, \Delta_{W4} > 0$), confirming active recursive graph connection. [PASS]
    5. *Pillar 21J.5 (VRAM Ladder)*: Candidate batches $B \in \{11, 22, 33, 44, 66\}$ profiled on Tesla T4; peak VRAM $= 10320.12\text{ MB}$ to $10565.33\text{ MB}$, throughput up to $23.2\text{ samp/s}$, zero OOM faults. [PASS]
    6. *Pillar 21J.6 (Production Contract Freeze & Checkpoint Roundtrip)*: Real checkpoint save/restore executed; positional weight comparison verifies bit-for-bit exact restore with $0.00\text{e}+00$ discrepancy. [PASS]
  * **Milestone Outcome**: **[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]** Formally certified on physical NVIDIA Tesla T4 GPU in Google Colab with all 6 technical pillars passing autonomously under fail-closed certification (`--mode certify`). Telemetry synchronized to cloud lake (`gs://rise-unet-rzsm/reproduction_audit/A0_gpu_benchmark.json`). Pre-Production Gate 2 is fully cleared.
  * **Formal Audit Dossier**: Certified in [`reproduction_audit/21_vram_and_hardware_profiling_audit.md`](reproduction_audit/21_vram_and_hardware_profiling_audit.md) and telemetry synced to `gs://rise-unet-rzsm/logs/A0_gpu_benchmark.json`.

---

#### Sub-Phase 21K: Full Production Ingestion & Model A0 Training
**Objective**: Construct the verified multi-source case calendar, acquire historical archives, train Model A0 across multiple random seeds, and freeze the regional baseline reference.

- [x] **Step 21K.1**: Construct Usable Case Calendar First before bulk retrieval (`[PASS / VERIFIED / ACCEPTED: 2026-09-14]`).
  * Definition: $\text{Usable Case} = \text{Required Antecedent RZSM} \cap \text{Future Target RZSM} \cap \text{ERA5 Atmospheric} \cap \text{ECMWF S2S Issue/Lead Availability}$.
  * Mandatory Lead Availability: Required S2S lead-time coverage through W4 must be available for a case to enter the production calendar (preventing partial cases from entering the cohort).
  * Authoritative Calendar Definition: Enforces the **ECMWF CY48R1 Operational Schedule-Referenced Forecast Origin Calendar** (1,154 candidate cycles, matching Kyle Lesinger's parent study `download_data_update.py:L61-70`). In CY48R1, reforecasts are executed for historical years on the exact month/day corresponding to the twice-weekly reference operational schedule (105 runs/yr for hindcast years 2015–2023, 105 in 2024, 104 in 2025). This reconciles why 1,154 cycles exist rather than 1,148 calendar Mon/Thu occurrences, matching the actual Copernicus CDS lake reality (527/531 cached GCS cycles match Rule 2; only 154 match Rule 1).
  * Calendar Artifact: Generated and certified production case calendar in `manifests/production_case_calendar.csv` (1,154 operational forecast cycles spanning 2015–2025; SHA-256 verified; synced to `gs://rise-unet-rzsm/manifests/production_case_calendar.csv`).
  * Partition Distribution: Exact manifest accounting:
    - **TRAIN (2015–2021)**: 7 years $\times$ 105 cycles/year = **735 cycles** ($63.7\%$).
    - **VAL (2022–2023)**: 2 years $\times$ 105 cycles/year = **210 cycles** ($18.2\%$; exactly 105 in 2022 and 105 in 2023).
    - **SEALED_TEST (2024–2025)**: 105 in 2024 + 104 in 2025 = **209 scheduled cases** ($18.1\%$ manifest cohort census; **202 usable sealed-test cases** forming the evaluation denominator; exactly **7 quarantined cases** in late Dec 2025 where $W_4$ targets extend into Jan 2026 beyond the ERA5-Land cube).
    - **Grand Total**: $735 + 210 + 209 = \mathbf{1,154}$ operational cycles.
  * Boundary Truncation: Exactly 7 trailing cycles in Dec 2025 (`CASE_20251208_1148` to `CASE_20251229_1154`) marked `TARGET_OUT_OF_BOUNDS` as $W_4$ targets extend into Jan 2026. These 7 cases are permanently quarantined from final metric computation while retained in the manifest census for chronological traceability.
  * Formal Audit Dossier: [`reproduction_audit/18_production_case_calendar_1154_cycles_audit.md`](reproduction_audit/18_production_case_calendar_1154_cycles_audit.md).
  * Test Suite: Validated by `tests/test_07_case_calendar.py` (87 unit tests passing repo total).
- [x] **Step 21K.2**: Build training, validation, and untouched test splits (`[PASS_NORMALIZATION_FROZEN_MANIFESTS_VERIFIED: 2026-09-14]`).
  * Forecast-Origin Partitioning: Partitioning is strictly by forecast-origin issuance date ($t_0$), eliminating all shared issue cycles ($\mathcal{T}_{\text{train}} \cap \mathcal{T}_{\text{val}} = \emptyset$, $\mathcal{T}_{\text{val}} \cap \mathcal{T}_{\text{test}} = \emptyset$).
  * Target-Horizon Boundary Extension Audit: 8 trailing training cycles ($t_0 \in [\text{2021-12-05}, \text{2021-12-30}]$) have $W_4$ targets extending up to 25 days into January 2022; 8 trailing validation cycles ($t_0 \in [\text{2023-12-05}, \text{2023-12-28}]$) have $W_4$ targets extending up to 24 days into January 2024. Methodologically standard for S2S prediction (zero feature/predictor leakage across boundaries).
  * Training Split: 2015–2021 cases (735 forecast cycles, 63.7%; `manifests/splits/train_cases.csv`, SHA-256 `d0aea5558ca6eb24712f2acad83c849002c0ffbc97206966b8bb5456d16e8a59`).
  * Validation Split: 2022–2023 cases (210 forecast cycles, 18.2%; 105 in 2022, 105 in 2023; Val-A=105, Val-B=105; `manifests/splits/val_cases.csv`, SHA-256 `0ab5fbd430d20812f32c7ddce29450e8548d4958254518654f8e651ca5a586b0`; used for hyperparameter evaluation and checkpoint minimum-CRPS selection).
  * Sealed Test Split: 2024–2025 cases (209 scheduled cases cohort census, 18.1%; 105 in 2024, 104 in 2025; **202 usable sealed-test cases** forming the evaluation denominator, **7 quarantined cases** where $W_4$ target extends into Jan 2026; `manifests/splits/test_cases_sealed.csv`, SHA-256 `443d5af42bd41a5229b98d94811c0a92ca1a1108702fb61a11c173085d808f88`; strictly quarantined until Phase 26). Scientific publications must never report simply '209 cases evaluated', but explicitly declare the $N=202$ usable evaluation denominator.
  * Normalization Parameters & Executable Binding: Derived strictly and exclusively from the 2015–2021 training partition over 126 active cells (`contracts/A0/normalization_parameters.yaml` and `.json`). Actively consumed by `src/data/case_builder.py` when `normalize=True` and verified by executable unit test `test_tensor_builder_active_normalization_contract`. Zero future leakage.
  * Pre-Training Production Contract: Machine-readable specifications frozen in `contracts/A0/mindanao_a0_production_contract.yaml` (Adam, $\text{lr}=10^{-4}$, seeds $[42, 123, 456]$, batch sizes $B \in \{11, 22, 33, 44, 66\}$, spatial CRPS loss, multi-head weights $[0.2, 0.3, 0.5]$).
  * Formal Audit Dossier: [`reproduction_audit/19_dataset_splits_and_normalization_contract_audit.md`](reproduction_audit/19_dataset_splits_and_normalization_contract_audit.md).
  * Test Suite: Validated by `tests/test_08_normalization_and_splits.py` (87 unit tests passing repo total).
- [x] **Step 21K.3-pre**: Genuine Production-Path Model A0 Training Smoke Test across all 4 Leads with Evaluation Domain Masking. **`[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]`**
  * **Certification Findings**: Formally certified on physical NVIDIA Tesla T4 GPU in Google Colab across all 5 production stages with zero fallback invocations and strict fail-closed enforcement (Exit Code 0).
  * **Certified 5-Stage Scope**:
    - **Stage A**: Ingested real representative pilot case (`CASE_20150115_W01.npz`), normalized via frozen contract (`contracts/A0/normalization_parameters.yaml`), verifying exact parameter counts ($W_1=1,627,139$; $W_2=1,630,307$; $W_3=1,608,131$; $W_4=1,611,299$), finite positive gradients, and non-zero weight updates ($\|\Delta w\| \approx 0.11 - 0.12$) across all 4 leads.
    - **Stage B**: Genuine forward model inference cascade ($W_1 \to \hat{y}_1 \to W_2 \to \hat{y}_2 \to W_3 \to \hat{y}_3 \to W_4$) on real data, verifying exact recursive channel placement ($W_2$ ch 11; $W_3$ ch 3, 4; $W_4$ ch 3, 4, 5) and permutation tamper rejection ($\Delta_{\max} = 0.2135$).
    - **Stage C**: Explicit production loss path demonstrating 3-head deep supervision, proving active domain decoupling (mean unmasked MAE = 0.3201 vs mean masked MAE = 0.8847, domain discrepancy = 0.5646 / 0.6793).
    - **Stage D**: Strict checkpoint scoping proving model-weight parity ($0.00 \times 10^0$) and full training-state next-step trajectory roundtrip (loss delta = $0.00 \times 10^0$, weight delta = $6.70 \times 10^{-7} < 5 \times 10^{-6}$) using direct variable assignment.
    - **Stage E**: Standalone engine execution (`scripts/14_run_a0_production_smoke_test.py --mode certify`) with exit code 0; telemetry synchronized to `gs://rise-unet-rzsm/reproduction_audit/a0_production_smoke_test.json`.
    - **Audit Dossier**: [`reproduction_audit/23_production_smoke_preflight_audit.md`](reproduction_audit/23_production_smoke_preflight_audit.md).
- [ ] **Step 21K.3**: Train Model A0 across minimum three predeclared seeds (seeds 42, 123, 456). **`[AUTHORIZED: 2026-09-17]`**
  * **Prerequisites Status**: All three Pre-Production Gates cleared: (1) 2022–2023 atmospheric mirroring certified under Pre-Production Gate 1 [PASS / CERTIFIED: 2026-09-16], (2) 21J hardware profiling & VRAM benchmark certified on physical Tesla T4 under Pre-Production Gate 2 [PASS / CERTIFIED_ON_GPU: 2026-09-17], and (3) 21K.3-pre certified on physical Tesla T4 under Pre-Production Gate 3 [PASS / CERTIFIED_ON_GPU: 2026-09-17]. Full 3-Seed Model A0 Production Training is hereby **AUTHORIZED FOR IMMEDIATE EXECUTION**.
  * Checkpoint Selection Rule: Primary checkpoint per seed = minimum validation CRPS, subject to all integrity checks.
  * Performance Reporting: Report A0 reference performance per-seed, mean across seeds, and standard deviation across seeds.
  * Storage: `gs://rise-unet-rzsm/checkpoints/A0/`.
- [ ] **Step 21K.4**: Generate validation predictions and evaluate baseline metrics.
  * Metrics: ACC, MAE, RMSE, CRPS, and categorical drought Brier score (strictly conforming to `contracts/A0/metric_evaluation_contract.yaml`).
  * Storage: `gs://rise-unet-rzsm/predictions/A0/` and `metrics/A0/`.
- [ ] **Step 21K.5**: Assemble and freeze A0 Baseline Contract Package.
  * Artifacts: `contracts/A0/A0_Mindanao_Baseline_Contract.yaml`, model weights, and performance dossier.

---

### Phase 22: Reference Comparators Track (B0, B1, B2)

**Objective**: Implement controlled, non-neural reference baselines on the exact same Mindanao cases and evaluation masks to benchmark deep learning skill. Phase 22 executes independently of Phase 23/24/25 and feeds directly into comparative evaluation in Phase 26.

- [ ] **Step 22.1**: Implement Model B0 (Seasonal Climatology).
  * Definition: Forecast for calendar week $w$ is the training climatological mean for week $w$.
  * Pass Criterion: Validation ACC, MAE, RMSE, and CRPS computed as the zero-skill anchor.
- [ ] **Step 22.2**: Implement Model B1 (Persistence).
  * Definition: Forecast for all lead weeks W1–W4 is the latest valid antecedent weekly RZSM lag window.
  * Pass Criterion: Validation metrics computed; establishes the benchmark for memory dissipation.
- [ ] **Step 22.3**: Implement Model B2 (Feature-Engineered XGBoost Benchmark).
  * Explicit Feature Contract: Model B2 operates under a strictly matched information budget: identical forecast issue cases, identical target verification dates, identical provisional train/val/test splits, identical spatial evaluation mask, zero future information leakage, and explicit feature availability strictly bounded by issue date $\tau$.
  * Spatial Target Matching Rule: B2 must preserve the same spatial target/evaluation unit as A0 (predicting the same gridded target independently per cell or matched spatial representation), unless an explicitly justified alternate spatial formulation is predeclared.
  * Pass Criterion: Validated under matched training splits without future data leakage.

---

### Phase 23: Recursive Degradation Diagnostic (Core Scientific Investigation)

**Objective**: Empirically investigate whether Model A0 suffers from systematic error compounding along the autoregressive recursive prediction chain ($W_1 \to W_2 \to W_3 \to W_4$).

- [ ] **Step 23.1**: Define parallel diagnostic inference protocols.
  1. **Protocol 1 (Standard Autoregressive Recursion)**:
     Downstream models ingest prior predicted states:
     $$\hat{y}_{W1} \to X_{W2} \to \hat{y}_{W2} \to X_{W3} \to \hat{y}_{W3} \to X_{W4} \to \hat{y}_{W4}$$
  2. **Protocol 2 (Oracle Counterfactual Diagnostic)**:
     Downstream models ingest the *true verifying state* rather than prior predictions to isolate sensitivity to prior state error:
     $$y_{W1}^{\text{true}} \to X_{W2}^{\text{oracle}} \to \hat{y}_{W2}^{\text{oracle}} \to y_{W2}^{\text{true}} \to X_{W3}^{\text{oracle}} \to \dots$$
  3. **Protocol 3 (Direct Non-Recursive Baseline)**:
     Models trained and evaluated directly from antecedent conditions and S2S forcings without recursive feedback channels (serving as a diagnostic comparator rather than an assumed superior architecture).
  4. **Protocol 4 (Controlled Error-Injection & Perturbation Tracking)**:
     Inject controlled perturbations into the Week 1 prediction ($\hat{y}_{W1} \pm \epsilon$ for $\epsilon \in \{0.05, 0.10, 0.25, 0.50\}$) and measure downstream divergence in Leads 2, 3, and 4 relative to the unperturbed recursive run:
     $$D_k(\epsilon) = \operatorname{RMSE}\left(\hat{y}_k^{(\epsilon)}, \hat{y}_k^{(0)}\right)$$
     directly quantifying perturbation propagation across leads.
- [ ] **Step 23.2**: Execute multi-lead validation evaluation under all four protocols.
  * Compute lead-by-lead performance: $\text{RMSE}_k, \text{MAE}_k, \text{ACC}_k, \text{CRPS}_k$ for $k \in \{1, 2, 3, 4\}$.
- [ ] **Step 23.3**: Compute the Recursive Error Accumulation Gap.
  * Formula:
    $$\Delta E_k = E_k^{\text{recursive}} - E_k^{\text{oracle}}$$
  * Decomposition: Distinguish between irreducible loss of meteorological predictability vs. compounding recursive state error.
- [ ] **Step 23.4**: Document diagnostic findings in `reproduction_audit/GATE2_RECURSIVE_DEGRADATION_DIAGNOSTIC.md`.

---

### Gate 2: Recursive Refinement GO / NO-GO Decision Gate

**Objective**: Formal decision gate establishing whether empirical evidence of recursive degradation justifies the introduction of an architectural refinement mechanism.

- [ ] **Step G2.1**: Evaluate formal GO / NO-GO criteria against validation data:
  * **Predeclared Protocol Rule**: The decision criteria and practical effect thresholds must be **frozen before viewing A0 diagnostic results** to prevent post-hoc rationalization.
  * **Primary GO Criteria**:
    1. Recursive error gap $\Delta E_k = E_k^{\text{recursive}} - E_k^{\text{oracle}}$ is positive and practically meaningful across downstream leads.
    2. Supported by statistical significance ($p < 0.05$), non-trivial effect size, well-bounded confidence intervals, and consistency across random seeds and forecast cases.
  * **Supporting Evidence**:
    1. Downstream error sensitivity confirms that prior-lead errors affect subsequent lead accuracy (even if degradation partially saturates rather than strictly monotonically increases across all 4 leads).
    2. Controlled error-injection test confirms that recursive feedback actively propagates state errors downstream ($D_k(\epsilon)$ demonstrates systematic, practically meaningful downstream sensitivity beyond numerical/trivial variation).
    3. Replacing recursive predictions with oracle or damped states materially improves downstream skill.
  * **NO-GO Criteria**:
    1. Recursive penalty is negligible, unstable, or not specifically associated with recursive state errors.
    2. Degradation occurs equally under oracle counterfactual state injection.
  * **Predeclared Protocol Action**:
    * If **GO** is confirmed: Proceed to Phase 24 (Model A1 Lead-Aware Recursive Residual Refinement).
    * If **NO-GO** is confirmed: Formally halt neural architectural modifications; designate Model A0 as the authoritative regional benchmark, document that the tested evidence does not support recursive feedback as a sufficiently material or actionable source of degradation under the tested conditions, and retain A0 as the regional benchmark.
- [ ] **Step G2.2**: Publish formal Gate 2 Decision Dossier.
  * Storage: `reproduction_audit/GATE2_DECISION_DOSSIER.md`.
  * Pass Criterion: Clear empirical justification locked before Phase 24 architecture construction begins.

---

### Phase 24: Model A1 (Lead-Aware Recursive Residual Refinement)

**Objective**: Implement and train the proposed enhancement architecture: preserving the verified RISE-UNet backbone while introducing a lightweight, lead-conditioned residual correction module to suppress error compounding along the recursive chain.

- [ ] **Step 24.1**: Design Lead-Aware Recursive Residual Refinement module.
  * Architectural Principle: The verified RISE-UNet backbone remains **unchanged and frozen**; the refinement module is added externally to the recursive state transition.
  * Formulation: Before passing prediction $\hat{y}_k$ into the input tensor of Lead $k+1$, a lightweight convolutional refinement block estimates a calibrated residual:
    $$\tilde{y}_k = \hat{y}_k + \mathcal{R}_{\theta}(\hat{y}_k, \hat{y}_{k-1}, k)$$
  * Input Isolation Rule: The refinement module receives **only information available at the recursive transition**; verifying ground truth observations are used exclusively as training targets and never as transition inputs, preventing disguised teacher-forcing.
  * Training Regime Safeguard: A1 refinement training must not use teacher-forced intermediate predictions unless that is explicitly its intended training regime and evaluated separately; the refinement module must train under the same recursive transition conditions it receives during multi-week inference.
  * Lead Conditioning: Compare candidate encodings during development (normalized scalar $k/4$ vs. learned embedding / scale-bias block).
  * Training Protocol: Model A0 backbone weights remain **frozen**, training exclusively the refinement module $\mathcal{R}_{\theta}$ for clean causal attribution.
- [ ] **Step 24.2**: Execute A1 pilot ladder (single case $\to$ overfit $\to$ checkpoint verification).
  * Pass Criterion: Refinement module trains stably without numerical divergence; overfit test succeeds.
- [ ] **Step 24.3**: Production training of Model A1 across identical random seeds (seeds 42, 123, 456).
  * Storage: `gs://rise-unet-rzsm/checkpoints/A1/`.
  * Pass Criterion: Identical learning rate schedule, batch size, and loss formulation as Model A0.
- [ ] **Step 24.4**: Freeze A1 winning checkpoints based on validation CRPS.
  * Storage: `manifests/A1_frozen_checkpoints.yaml` and `contracts/A1/A1_Refined_Contract.yaml`.

---

### Phase 25: Ablation & Robustness Studies

**Objective**: Verify the specific mechanisms of the proposed refinement module and evaluate operational stability under edge conditions.

- [ ] **Step 25.1**: Execute Architectural Ablation Study.
  * Variant 1: Refinement module without lead conditioning (static residual).
  * Variant 2: Lead-aware refinement (full Model A1).
  * Pass Criterion: Quantifies the isolated contribution of lead-dependent calibration.
- [ ] **Step 25.2**: Execute Controlled Error-Damping Test.
  * Method: Inject synthetic perturbations into Lead 1 predictions and measure downstream divergence in Leads 2–4.
  * Pass Criterion: Model A1 demonstrably damps error propagation compared to unrefined Model A0.
- [ ] **Step 25.3**: Sub-Regional Performance Quantification.
  * Method: Performance variability is quantified across predefined geographic/hydroclimatic subregions derived from official administrative boundaries and river basin classifications.
  * Pass Criterion: Performance characteristics documented across diverse terrain and rainfall regimes without ad-hoc boundary definitions.

---

### Phase 26: Final Statistical Evaluation on Sealed Hold-Out

**Objective**: Unseal the held-out test period (provisional 2024 to 2025) and execute the authoritative statistical evaluation and hypothesis tests across all candidate models.

- [ ] **Step 26.1**: Strict Sealed Evaluation Protocol.
  * Non-Leakage Rule: **No threshold, architecture choice, preprocessing choice, hyperparameter, seed-selection rule, or refinement variant may be selected using 2024–2025 test data.**
  * Unseal test dataset and generate immutable predictions.
  * Models Evaluated: A0 (Adapted Baseline), A1 (Refined Model), B0 (Climatology), B1 (Persistence), B2 (XGBoost).
  * Storage: `gs://rise-unet-rzsm/predictions/test_holdout/`.
- [ ] **Step 26.2**: Compute deterministic forecast metrics.
  * Metrics: Spatial ACC, MAE, RMSE computed per lead week over active Mindanao land cells.
- [ ] **Step 26.3**: Compute probabilistic forecast metrics.
  * Metrics: CRPS and Continuous Ranked Probability Skill Score (CRPSS) referenced against Climatology (B0) and Baseline (A0).
- [ ] **Step 26.4**: Compute categorical drought event metrics.
  * Event Definition: Agricultural drought defined as training-period, grid-cell-specific, seasonally conditioned 20th percentile (derived exclusively from training data).
  * Metrics: Brier Score, Reliability Diagrams, ROC/AUC.
- [ ] **Step 26.5**: Execute paired moving-block bootstrap significance test.
  * Method: 1,000 bootstrap resamples using a paired moving-block bootstrap over temporally contiguous forecast cases, with block length predeclared to preserve temporal serial correlation.
  * Null Hypothesis: $\mathcal{H}_0: \text{CRPS}_{\text{A1}} \ge \text{CRPS}_{\text{A0}}$ vs. $\mathcal{H}_1: \text{CRPS}_{\text{A1}} < \text{CRPS}_{\text{A0}}$.
  * Decision Criteria: Report paired effect estimate, 95% confidence interval, p-value, and practical effect magnitude. Statistical significance ($p < 0.05$) alone does not establish meaningful improvement without practically relevant effect magnitude and well-bounded confidence intervals.
- [ ] **Step 26.6**: Generate publication-ready figures and tables.
  * Deliverables: Spatial skill maps, lead-time degradation curves, reliability diagrams, case study drought progression maps.
  * Storage: `gs://rise-unet-rzsm/figures/`, `metrics/`, and `logs/`.

---

### Phase 27: Research Reproducibility Package & Thesis Archive

**Objective**: Consolidate and package the final research artifacts, frozen weights, execution dossiers, and code into an immutable repository release.

- [ ] **Step 27.1**: Package final research reproducibility archive.
  * Release Tag: `v1.0.0-thesis-defense`.
  * Deliverable: Fully documented code, frozen weights, case manifests, and execution dossiers archived on GitHub and Zenodo/GCS.

---

## Operational Toolchain & Environment Reference

### 1. Git Branch Management
```bash
git checkout parent-reproduction
git pull origin parent-reproduction
git switch -c mindanao-adaptation
git push -u origin mindanao-adaptation
```

### 2. GCS Storage Structure (Verified Authenticated Bucket)
```text
gs://rise-unet-rzsm/
├── raw/
│   ├── boundaries/
│   │   └── mindanao/
│   │       ├── psgc/
│   │       │   └── PSGC_2Q2026_Publication_Datafile.xlsx
│   │       └── geoportal/
│   │           └── geoportal-regionalboundary_20210504_export.geojson
│   ├── era5_land/
│   │   ├── pilot/
│   │   └── production/
│   ├── era5/
│   │   ├── pilot/
│   │   └── production/
│   └── ecmwf_s2s/
│       ├── pilot/
│       └── production/
├── processed/
│   ├── boundaries/
│   │   └── mindanao/
│   │       ├── PSA_Geoportal_Regional_20210504_Mindanao.gpkg
│   │       └── mindanao_analysis_boundary.gpkg
│   ├── grid/
│   │   ├── mindanao_025deg.nc
│   │   ├── mindanao_fraction_025.nc
│   │   └── mindanao_eval_mask_025.nc
│   ├── rzsm/
│   ├── era5_atm/
│   ├── s2s/
│   └── cases/
│       ├── pilot/
│       └── production/
├── contracts/
│   ├── spatial/
│   ├── data/
│   ├── A0/
│   └── A1/
├── manifests/
│   ├── boundary_checksums.txt
│   ├── boundary_manifest.yaml
│   ├── production_case_calendar.csv
│   └── cases_pilot_v001.csv
├── checkpoints/
│   ├── A0/
│   └── A1/
├── predictions/
├── metrics/
├── figures/
├── logs/
└── support/
    ├── gis_scripts/
    │   ├── extract_mindanao.py
    │   ├── verify_mindanao_codes.py
    │   └── generate_boundary_manifest.py
    └── qgis/
        └── mindanao_boundary_validation.qgz
```

### 3. Colab High-Speed Storage Access
```python
from google.colab import auth
auth.authenticate_user()
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("rise-unet-rzsm")
```
