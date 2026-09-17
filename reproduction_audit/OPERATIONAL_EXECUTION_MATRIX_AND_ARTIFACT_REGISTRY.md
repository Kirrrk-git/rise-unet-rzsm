<!-- markdownlint-disable -->
# Mindanao RISE-UNet Operational Execution Matrix, Audit Dossier Index & Repository File Registry

**Project**: Enhanced RISE-UNet for Subseasonal Root-Zone Soil Moisture (RZSM) Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Parent Baseline**: Lesinger & Tian (2025), *Nature Communications*, DOI: [`10.1038/s41467-025-62761-3`](https://doi.org/10.1038/s41467-025-62761-3)  
**Target Model**: **Mindanao Model A0** (Adapted from EX29 Recursive Hybrid RISE-UNet Baseline)  
**Document Classification**: Living Operational Matrix, Comprehensive Directory Registry & Master File Map  
**Last Updated**: 2026-09-17 (Step 21K.3 Full Three-Seed Model A0 Production Training & GCS Parity Certified)  

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
| **Gate 1A** | Parent Code Reproduction & Environment Freeze | EX29 Environment Parity | `[VERIFIED]` | [`01_parent_baseline_replication_audit.md`](01_parent_baseline_replication_audit.md) |
| **Gate 1B** | Parent Pipeline Trace & Recursive Loop Verification | EX29 Recursive Logic Parity | `[VERIFIED]` | [`02_parent_experiment_trace_audit.md`](02_parent_experiment_trace_audit.md) |
| **Step 21A.1** | Isolated Mindanao Adaptation Git Branch Creation | Git Commit Integrity (`4adb9fa`) | `[PASS]` | Git branch `origin/mindanao-adaptation` |
| **Step 21A.2** | Official PSA Administrative Boundary Harmonization & Geodesy | PSA PSGC 2Q 2026 / GeoRiskPH | `[ACCEPTED]` | [`06_mindanao_boundary_and_gadm_audit.md`](06_mindanao_boundary_and_gadm_audit.md) ($99,948.76\text{ km}^2$) |
| **Step 21A.3** | Parent EX29 Channel, Lag & Variable Contract Reconciliation | EX29 Source-Code Contract Parity | `[VERIFIED]` | [`03_parent_variable_contract_audit.md`](03_parent_variable_contract_audit.md) ($W_1=11, W_2=12, W_3=5, W_4=6$) |
| **Step 21A.4** | Candidate 0.25° Spatial Envelopes & 16-Divisible Geometry Analysis | Divisibility $H, W \pmod{16} = 0$ | `[ACCEPTED]` | [`05_mindanao_geometry_envelopes_audit.md`](05_mindanao_geometry_envelopes_audit.md) |
| **Step 21A.5** | Synthetic Candidate Tensor & 4-Lead Cascade Compatibility Gate | Native TF2 Forward & Backward Flow | `[PASS]` | `notebooks/02_mindanao_geometry_and_cascade_gate.ipynb` |
| **Step 21A.6** | Candidate A ($32 \times 48, 0.25^\circ$) Grid Selection & NetCDF Construction | Candidate A Mesh Freeze | `[ACCEPTED]` | [`07_mindanao_candidate_a_grid_freeze_audit.md`](07_mindanao_candidate_a_grid_freeze_audit.md) |
| **Step 21A.7A** | Fractional Boundary-Coverage Mask (`mindanao_fraction_025.nc`) | WGS84 Ellipsoidal Geodesics | `[ACCEPTED]` | `processed/grid/mindanao_fraction_025.nc` ($99,948.76\text{ km}^2$, $0.0000\%$ error) |
| **Step 21A.7B** | Binary Evaluation Mask ($f \ge 0.50$, 126 Cells) | Majority Land Rule ($f \ge 0.50$) | `[ACCEPTED]` | [`08_mindanao_evaluation_mask_audit.md`](08_mindanao_evaluation_mask_audit.md) (126 cells: full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$) |
| **Step 21A.8** | Freeze Machine-Readable Spatial Contract | Immutable Spatial Governance | `[PASS]` | [`contracts/spatial/spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml) |
| **Step 21A.9** | ERA5-Land RZSM Target Acceptance & Independent Cross-Reference | Depth-Weighted Profile & Physics | `[ACCEPTED]` | [`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md) |
| **Sub-Phase 21A** | Spatial Foundation Synthesis & Geodetic Certification | Complete Geodetic Quadrature | `[PASS]` | [`09_mindanao_spatial_foundation_dossier.md`](09_mindanao_spatial_foundation_dossier.md) |
| **Step 21B.1** | ERA5-Land Archive Ingestion Audit (2015–2025 Nominal Archive) | 264 NetCDF Files (132 Monthly Pairs) | `[PASS]` | [`10_era5_land_archive_completeness_audit.md`](10_era5_land_archive_completeness_audit.md) |
| **Step 21B.2** | 2014 Antecedent Window Retrieval (20 Days) | +1 NetCDF File (265 Combined Files) | `[ACCEPTED]` | [`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md) (12–31 Dec 2014 Support Archive in GCS) |
| **Step 21B.3** | Pilot Daily Mean & Depth Weighting ($0.07, 0.21, 0.72$) | 0–100 cm Physical Profile | `[VERIFIED]` | [`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md) |
| **Step 21B.4** | Bilinear Remapping, Coastal Fallback & Pilot Census | Land-Aware Remapping & Masking | `[PASS]` | [`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md) (2,520 evaluations, zero NaNs) |
| **Step 21C.1** | ERA5 Atmospheric Pilot Retrieval (Jan 2015, 744 Hours) | Hourly Single & Pressure Levels | `[PASS]` | [`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md) |
| **Step 21C.2** | 5-Channel Atmospheric Derivation & Census | Bolton (1980) Humidity & Extr. | `[PASS / VERIFIED / ACCEPTED]` | [`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md) (3,906 evaluations, zero NaNs) |
| **Step 21C.3** | 11-Year ERA5 Atmospheric Archive Mirroring | Hourly Monthly NetCDFs to GCS | `[TRAINING_ARCHIVE_SECURED]` | Training fold (Jan 2015 – Dec 2021, 84 months $\times$ 2 products = 168 files) fully secured and verified in GCS lake; Dec 2014 antecedent support (2 files) tracked separately. Full 11-year archive target is 264 monthly files; validation/test partition (2022–2025, 96 files) background mirroring in progress |
| **Step 21D.1** | Depth-Weighted RZSM Modular Implementation | Vectorized `src/data/rzsm.py` | `[PASS]` | Automated Unit Tests (`test_02_rzsm.py`) |
| **Step 21D.2** | Automated RZSM Unit Test Suite | Precision $\le 10^{-7}$, Real Pilot | `[PASS]` | 10 Unit Tests Passing |
| **Step 21D.3** | Temporal Preprocessing & Target Lead Reconciliation | $L=[6,13,20,27]$, Trailing Rolling | `[PASS / VERIFIED]` | Satisfies software tests & explicitly reconciled parent contracts ([`Audit`](04_parent_temporal_target_parity_audit.md)) |
| **Step 21D.4-PREFLIGHT** | Production Cube Preflight Verification Gate | 11-Point Preflight Census & Integrity Gate | `[PASS]` | Verified 265 NetCDFs (96,912 hrs across 4,038 days), zero NaNs/Infs, verified spatial contract ([`Report`](13_production_cube_preflight_audit.md)) |
| **Sub-Phase 21E** | ECMWF S2S Dynamic Triplet Pilot Acquisition & Harmonization Engine | ECDS API (`t2m, d2m, tcw`), 11 Members, $1.5^\circ \to 0.25^\circ$ Remap, Hard-Fail Safe Production Default | `[PASS / VERIFIED]` | Pure-Python GRIB2 Section 7 decoder & harmonizer in `src/data/s2s.py` with `allow_step0_fallback=False` default; 7 unit tests passing; pilot artifact `processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc` (0.42 MB) synced to GCS; certified in [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md) |
| **Step 21F.1–2** | Single Complete EX29-Derived A0 Case & Target Assembly | Leads W1–W4 Multi-Lead Tensor Schema ($[11, 12, 5, 6]$), Exact Lags `[-1, -7, -14]`, Verified Target Offsets `[6, 13, 20, 27]` | `[PASS / VERIFIED]` | Multi-lead tensor hierarchy assembler in `src/data/case_builder.py`; verified $[M=11, 32, 48, C_k]$ across $W_1..W_4$; targets reconciled to parent EX29 $L = (\text{lead} \times 7) - 1$; 5 unit tests passing; 81 total unit tests passing |
| **Step 21F.3** | ecCodes Interoperability, Real Case Assembly & Recursive Graph Integrity | Scoped ecCodes comparison (462 msgs), UNET_RZSM 4-lead recursive cascade, downstream perturbation response, zero NaNs/Infs over 126 binary evaluation cells | `[PASS / VERIFIED]` | Certified in [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md) — Case Assembly & Computational Integrity (Forecasting skill deferred to 21H–21K); Interactive Colab notebook `notebooks/07_mindanao_s2s_and_pilot_case_assembly.ipynb` & dual 2x2 publication dashboards `figures/mindanao_s2s_dynamic_predictor_composite.png` and `figures/mindanao_s2s_pilot_case_and_recursive_inference.png` (dual synced to GCS) |
| **Sub-Phase 21G** | 8-Case Pilot Ladder Manifest Generation & Pipeline Stability | 8 Consecutive Cycles (Jan 15 – Mar 4, 2015), 88-Row Member Manifest, 0 NaNs across 314,496 values, GCS Parity, 16-point independent provenance verification | `[PASS / VERIFIED]` | Certified in [`17_pilot_case_ladder_and_manifest_audit.md`](17_pilot_case_ladder_and_manifest_audit.md); Member manifest `manifests/cases_pilot_v001.csv` (88 rows) & summary `manifests/cases_pilot_summary_v001.csv` (8 rows); 8 NPZ cases in `processed/cases/pilot/` synced to `gs://rise-unet-rzsm/processed/cases/pilot/`; 42/42 unit tests passing; verified by `scripts/10_verify_pilot_ladder_provenance_and_census.py` |
| **Sub-Phase 21H** | Surrogate TensorFlow Data Pipeline, Ensemble Grouping & Checkpoint Smoke Test | SURROGATE SMOKE TEST ONLY. Validated data feeding, 11-member batching, target broadcasting, and checkpoint saving on 2-layer Conv2D surrogate (did NOT instantiate genuine UNET_RZSM) | `[SURROGATE_ONLY]` | Certified in [`20_tf_dataset_pipeline_and_checkpoint_audit.md`](20_tf_dataset_pipeline_and_checkpoint_audit.md); Module `src/data/tf_dataset.py`; Automated test suite `tests/test_11_tf_dataset.py` (16/16 passing); Pipeline test script `scripts/13_train_a0_pipeline_checkpoint.py`; Interactive Colab notebook `notebooks/08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb`; Checkpoints in `checkpoints/a0_pipeline_test/` |
| **Sub-Phase 21I** | Surrogate Tiny-Data Optimization Smoke Test (8 Cases, 40 Epochs) | SURROGATE SMOKE TEST ONLY. 2-layer Conv2D surrogate on unnormalized inputs; 99% loss drop was an unnormalized scale artifact, NOT proof of genuine RISE-UNet capacity or overfit; reclassified universally as surrogate smoke test | `[SURROGATE_ONLY]` | Certified in [`22_a0_tiny_overfit_gradient_audit.md`](22_a0_tiny_overfit_gradient_audit.md); Script `scripts/12_train_a0_tiny_overfit.py`; Log `logs/a0_tiny_overfit_execution.json`; Checkpoint `checkpoints/a0_tiny_overfit/` |
| **Sub-Phase 21J** | Genuine Model A0 (`UNET_RZSM`) Hardware Profiling & VRAM Feasibility Benchmark | Hardware Feasibility & Production Contract Gate: Genuine 1.63M-parameter nested U-Net across Six Technical Pillars (21J.1 architecture: 1,627,139 W1 / 1,630,307 W2 params, 298 tensors, 3 deep supervision heads; 21J.2 multi-lead forward with evaluation-domain postprocessing mask $0.00 \times 10^0$; 21J.3 backpropagation with finite gradients under representative MAE workload; 21J.4 4-lead cascade with active perturbation propagation; 21J.5 VRAM ladder $B \in \{11, 22, 33, 44, 66\}$ from $2.08\text{ GB}$ to $10.57\text{ GB}$; 21J.6 production contract freeze & real weight roundtrip parity). Certified autonomously on physical Tesla T4 GPU | `[PASS / CERTIFIED_ON_GPU: 2026-09-17]` | Certified in [`21_vram_and_hardware_profiling_audit.md`](21_vram_and_hardware_profiling_audit.md); Interactive Colab notebook `notebooks/09_mindanao_a0_vram_profiling.ipynb`; Benchmark script `scripts/11_profile_a0_vram_benchmark.py`; Model factory `src/models/a0_unet.py`; Unit test `tests/test_12_a0_unet.py`; Physical GPU telemetry artifact `logs/A0_gpu_benchmark.json` |
| **Sub-Phase 21K** | Production Case Ingestion, Splits, Normalization & Baseline Training | 21K.1 Usable Case Calendar: 1,154 cycles indexed via ECMWF CY48R1 schedule-referenced forecast origin calendar; 21K.2 Dataset Splits (735 Train, 210 Val, 209 Scheduled Test Census: 202 Usable Sealed Test Denominator, 7 Quarantined) & Training Normalization Contract (`contracts/A0/normalization_parameters.yaml`) frozen strictly from 2015–2021; Dynamic tensor normalization actively bound in `case_builder.py`; 101 unit tests passing repo-wide; Step 21K.3-pre formally certified on Tesla T4 GPU; Step 21K.3 production training AUTHORIZED | `[IN PROGRESS]` (Steps 21K.1, 21K.2: `[PASS / VERIFIED / ACCEPTED]`; 21K.3-pre: `[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]`; 21K.3: `[AUTHORIZED: 2026-09-17]`) | Certified in [`18_production_case_calendar_1154_cycles_audit.md`](18_production_case_calendar_1154_cycles_audit.md), [`19_dataset_splits_and_normalization_contract_audit.md`](19_dataset_splits_and_normalization_contract_audit.md), and [`23_production_smoke_preflight_audit.md`](23_production_smoke_preflight_audit.md); Calendar in `manifests/`; Splits in `manifests/splits/`; Contracts in `contracts/A0/`; Verification Status in `contracts/A0/VERIFICATION_STATUS.yaml` |
| **Step 21K.3-pre** | Genuine Production-Path Model A0 Training Smoke Test across all 4 Leads | 5-Stage Preflight: 4-lead genuine backward updates on real pilot data (`CASE_20150115_W01.npz`), recursive autoregressive cascade on actual model inferences, downstream 126-cell masked multi-head loss, permutation tamper rejection, and full training-state step-2 trajectory roundtrip. Certified on physical Tesla T4 GPU with zero fallbacks, exit code 0, and telemetry synced to cloud lake | `[PASS / CERTIFIED_ON_GPU: 2026-09-17]` | Certified in [`23_production_smoke_preflight_audit.md`](23_production_smoke_preflight_audit.md); Standalone preflight engine `scripts/14_run_a0_production_smoke_test.py`; Contract test `tests/test_13_production_smoke_preflight.py`; Execution notebook `notebooks/11_mindanao_a0_production_smoke_preflight.ipynb`; Telemetry in `logs/a0_production_smoke_test.json` (synced to GCS) |
| **Step 21K.3** | Full Three-Seed Model A0 Production Training across all 4 Leads | Three independent training replicates (`[42, 123, 456]`), 4-lead ($W_1 \to W_4$) recursive cascade, deep supervision `[0.2, 0.3, 0.5]`, Adam $\eta=10^{-4}$, ReduceLROnPlateau, early stopping, validation checkpoint selection via spatial CRPS proxy ($\text{MAE} - 0.08\bar{\sigma}_{\text{spatial}}$), and exact analytical ensemble CRPS (Hersbach 2000). Formally amended batch size $B=11 \to B=33$ ($3\text{ cases} \times 11\text{ members}$) for GPU throughput and gradient stability. Executed on physical Tesla T4 GPU in Google Colab with all 12 model training runs completed and checkpoints synchronized to GCS. Parent EX29 Contextual Reference Comparison [CONTEXTUAL / REPORTED REFERENCE] demonstrates performance broadly comparable to reported contextual benchmark, with higher ACC across all four horizons and mixed RMSE differences: W1 ACC=$0.8544 \pm 0.0047$, Exact CRPS=$0.0330 \pm 0.0008$, RMSE=$0.0505 \pm 0.0009$; W2 ACC=$0.6824 \pm 0.0096$, Exact CRPS=$0.0506 \pm 0.0020$, RMSE=$0.0714 \pm 0.0015$; W3 ACC=$0.5629 \pm 0.0067$, Exact CRPS=$0.0582 \pm 0.0011$, RMSE=$0.0800 \pm 0.0006$; W4 ACC=$0.4948 \pm 0.0095$, Exact CRPS=$0.0613 \pm 0.0003$, RMSE=$0.0844 \pm 0.0005$. Effective sample is 871 cases (677 train, 194 val). 2024–2025 Sealed Test cohort ($N=202$) quarantined for Phase 26 | `[PASS / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]` | Training engine `scripts/16_train_a0_production.py`; Orchestration notebook `notebooks/12_mindanao_a0_production_training.ipynb`; 3-seed summary `logs/a0_production_3seed_summary.json`; Checkpoints `checkpoints/A0/` (Seeds 42, 123, 456; Leads 1-4) synced to GCS; Parity verification figure `figures/a0_production_benchmark_parity_comparison.png` & spatial evaluation `figures/a0_production_spatial_forecast_evaluation.png` |
| **Step 21K.5** | Assemble & Freeze A0 Baseline Contract Package & Performance Audit Dossier | Formal post-training baseline contract (`contracts/A0/A0_Mindanao_Baseline_Contract.yaml`) and comprehensive performance audit report (`reproduction_audit/25_a0_production_training_and_parity_audit.md`). Formally seals Phase 21 Model A0 baseline and unlocks Phase 23 (Recursive Degradation Diagnostic) | `[PASS / ACCEPTED: 2026-09-17]` | Baseline Contract [`contracts/A0/A0_Mindanao_Baseline_Contract.yaml`](../contracts/A0/A0_Mindanao_Baseline_Contract.yaml); Audit Dossier [`reproduction_audit/25_a0_production_training_and_parity_audit.md`](25_a0_production_training_and_parity_audit.md) |
| **Pre-Production Gate 1** | Validation Atmospheric Pipeline Preflight Certification | Authoritative verification across 2022–2023 atmospheric mirror: continuous 730-day calendar (0 missing/duplicate days), 24-file fail-closed whitelist, 210 validation cycles, 5 channels, 126 active cells, 3-way normalization diagnostics, and production `CaseBuilder` ingestion into `CaseTensorHierarchy` | `[PASS / VERIFIED / ACCEPTED]` | Executed and certified live in Google Colab (2026-09-16). Authoritative engine `scripts/15_verify_validation_atmospheric_pipeline.py`; Execution wrapper `notebooks/10_mindanao_validation_atmospheric_pipeline.ipynb`; Telemetry in `logs/gate1_validation_atmospheric_execution.json`; Diagnostic composite `figures/gate1_validation_atmospheric_spatial_sanity.png` (synced to GCS) |
| **Phase 23** | Recursive Degradation Diagnostic [Track 1 Core Scientific Investigation] | Immediate post-A0 empirical diagnostic: error compounding across leads ($W_1 \to W_4$), Protocols 1–4 (standard recursion vs oracle truth vs direct non-recursive comparator vs error injection $D_k(\epsilon)$ with $[0,1]$ boundary clipping), computing primary recursive gap $\Delta E_k = E_k^{\text{recursive}} - E_k^{\text{oracle}}$ via case-level moving-block bootstrap (MBB with centered null $p_{\text{boot}}$) and supplementary paired Wilcoxon tests. Pre-analysis specification frozen in `contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml` before execution | `[DIAGNOSTIC CONTRACT FROZEN - READY FOR VALIDATION EVALUATION]` | Diagnostic Contract [`contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml`](../contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml); Engine `scripts/19_run_recursive_degradation_diagnostic.py`; Notebook `notebooks/13_mindanao_recursive_degradation_diagnostic.ipynb` |

| **Post-A0 Gate 2 (G2-R)** | Recursive Refinement Decision Gate [Track 1 Scientific Gate] | Formal decision gate evaluating statistical significance ($p_{\text{boot}} < 0.05$), non-trivial effect size ($d_z \ge 0.20$), 95% MBB CIs, replicate consistency, and magnitude monotonicity ($|\epsilon|$). If GO confirmed: unlock Phase 24. If NO-GO confirmed: halt neural enhancements, freeze A0 as definitive regional benchmark, and proceed directly to Phase 26 | `[PLANNED - FORMAL DECISION GATE]` | `reproduction_audit/GATE2_DECISION_DOSSIER.md` |
| **Phase 24** | Model A1 (Lead-Aware Recursive Residual Refinement) [Track 1 CONDITIONAL] | **Strictly conditional upon Post-A0 Gate 2 (G2-R) GO**. Preserves frozen Model A0 backbone while introducing lightweight, lead-conditioned convolutional residual refinement block $\tilde{y}_k = \hat{y}_k + \mathcal{R}_\theta(\hat{y}_k, \hat{y}_{k-1}, k)$ to suppress error compounding. (Bypassed if Gate 2 confirms NO-GO) | `[PLANNED - CONDITIONAL UPON GATE 2 GO]` | Model A1 Architecture & Training Dossier |
| **Phase 25** | Ablation & Robustness Studies [Track 1 CONDITIONAL] | **Strictly conditional upon Phase 24 activation**. Lead-conditioning ablations, controlled error-damping verification, and sub-regional hydroclimatic sensitivity | `[PLANNED - CONDITIONAL UPON PHASE 24]` | Ablation & Sensitivity Report |
| **Phase 22** | Reference Comparators Track (B0, B1, B2) [Track 2 Parallel Baseline Track] | Non-neural reference benchmarks (B0 Climatology, B1 Persistence, B2 Feature-Engineered XGBoost GBDT) evaluated on identical Mindanao cases and 126 evaluation cells under matched information budget. Executes independently in parallel and feeds directly into Phase 26 comparative tables | `[PLANNED - TRACK 2 PARALLEL TRACK]` | Comparative Baseline Evaluation Report |
| **Phase 26** | Final Sealed Test Evaluation & Explainability [Synthesis Track] | Final generalization benchmark on untouched 2024–2025 Sealed Holdout Cohort ($N=202$), paired moving-block bootstrap hypothesis testing, sub-regional verification, and SHAP explainability | `[PLANNED - FINAL SEALED EVALUATION]` | **Gate 3 Final Thesis Defense Dossier** |
| **Phase 27** | Thesis Synthesis & Open Data Lake Archive | Code Freeze, GCS Lake Index, Open Access | `[PLANNED]` | Final Project Archival Registry |

---

## 3. Comprehensive Repository File Registry (The Complete Project Map)

This section catalogs every directory and file introduced for the Mindanao adaptation. Original parent codebase files (such as root notebooks `00_min_max...` through `10a_...`) are excluded to maintain absolute clarity on adaptation additions.

```
dl_dm_rzsm_subseasonal_forecast/
├── checkpoints/
│   ├── a0_pipeline_test/
│   │   ├── a0_test_epoch005.weights.npz
│   │   └── a0_test_epoch005_meta.json
│   └── a0_tiny_overfit/
├── contracts/
│   └── spatial/
│       └── spatial_grid_contract.yaml
├── parent_study_ex29/
│   ├── README.md (Original author documentation with parent archive banner)
│   ├── conda_environment_setup.yaml (Original environment setup)
│   ├── Data/ (Original author raw download scripts, masks, and EMOS)
│   ├── function/ (Original author modules: modelRzsmRelu, basicUNET, losses, etc.)
│   └── notebooks/ (All 48 original CONUS notebooks: 00_*, 01_*, 02_*, ...)
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
├── logs/
│   └── a0_tiny_overfit_execution.json
├── contracts/
│   ├── spatial/
│   │   └── spatial_grid_contract.yaml
│   └── A0/
│       ├── normalization_parameters.yaml
│       ├── normalization_parameters.json
│       ├── mindanao_a0_production_contract.yaml
│       └── A0_Mindanao_Baseline_Contract.yaml
├── manifests/
│   ├── cases_pilot_v001.csv
│   ├── cases_pilot_summary_v001.csv
│   ├── cases_production_summary.csv
│   ├── production_case_calendar.csv
│   └── splits/
│       ├── train_cases.csv
│       ├── val_cases.csv
│       ├── test_cases_sealed.csv
│       ├── cases_availability_audit.csv
│       └── split_summary.json
├── metadata/
│   └── grid_definition.yaml
├── notebooks/
│   ├── 03_mindanao_spatial_foundation_and_mask_pipeline.ipynb
│   ├── 04_mindanao_rzsm_pilot_preprocessing.ipynb
│   ├── 05_mindanao_atmospheric_and_rzsm_preprocessing.ipynb
│   ├── 06_mindanao_datacube_and_anomaly_pipeline.ipynb
│   ├── 07_mindanao_s2s_and_pilot_case_assembly.ipynb
│   ├── 08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb
│   ├── 09_mindanao_a0_vram_profiling.ipynb
│   ├── 10_mindanao_validation_atmospheric_pipeline.ipynb
│   ├── 11_mindanao_a0_production_smoke_preflight.ipynb
│   ├── 12_mindanao_a0_production_training.ipynb
│   └── 13_mindanao_recursive_degradation_diagnostic.ipynb
├── reproduction_audit/
│   ├── README.md                               <── Dossier directory and navigation portal
│   ├── OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md (This File)
│   ├── 01_parent_baseline_replication_audit.md
│   ├── 01a_parent_baseline_execution_log.md
│   ├── 02_parent_experiment_trace_audit.md
│   ├── 02a_parent_experiment_trace_log.md
│   ├── 03_parent_variable_contract_audit.md
│   ├── 04_parent_temporal_target_parity_audit.md
│   ├── 05_mindanao_geometry_envelopes_audit.md
│   ├── 06_mindanao_boundary_and_gadm_audit.md
│   ├── 07_mindanao_candidate_a_grid_freeze_audit.md
│   ├── 08_mindanao_evaluation_mask_audit.md
│   ├── 09_mindanao_spatial_foundation_dossier.md
│   ├── 10_era5_land_archive_completeness_audit.md
│   ├── 11_era5_land_pilot_rzsm_preprocessing_audit.md
│   ├── 12_era5_land_target_acceptance_audit.md
│   ├── 13_production_cube_preflight_audit.md
│   ├── 14_production_cube_compilation_and_census_audit.md
│   ├── 15_era5_atmospheric_preprocessing_audit.md
│   ├── 16_ecmwf_s2s_pilot_and_recursive_inference_audit.md
│   ├── 17_pilot_case_ladder_and_manifest_audit.md
│   ├── 18_production_case_calendar_1154_cycles_audit.md
│   ├── 19_dataset_splits_and_normalization_contract_audit.md
│   ├── 20_tf_dataset_pipeline_and_checkpoint_audit.md
│   ├── 21_vram_and_hardware_profiling_audit.md
│   ├── 22_a0_tiny_overfit_gradient_audit.md
│   ├── 23_production_smoke_preflight_audit.md
│   ├── 24_s2s_provider_availability_and_cohort_census_audit.md
│   └── 25_a0_production_training_and_parity_audit.md
├── scripts/
│   ├── 01_generate_mindanao_masks.py
│   ├── 02_download_era5_atmospheric.py
│   ├── 03_derive_era5_atmospheric_daily.py
│   ├── 04_download_all_s2s_production.py
│   ├── 05_verify_production_cube_preflight.py
│   ├── 06_build_production_case_calendar.py
│   ├── 07_generate_dataset_splits.py
│   ├── 08_derive_training_normalization.py
│   ├── 09_build_pilot_manifest_and_cases.py
│   ├── 10_verify_pilot_ladder_provenance_and_census.py
│   ├── 11_profile_a0_vram_benchmark.py
│   ├── 12_train_a0_tiny_overfit.py
│   ├── 13_train_a0_pipeline_checkpoint.py
│   ├── 14_run_a0_production_smoke_test.py
│   ├── 15_verify_validation_atmospheric_pipeline.py
│   ├── 16_train_a0_production.py
│   ├── 17_build_production_cases.py
│   ├── 18_verify_production_training_artifacts.py
│   ├── 19_run_recursive_degradation_diagnostic.py
│   └── run_download_atmospheric.sh
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── rzsm.py
│   │   ├── temporal.py
│   │   ├── compile_cube.py
│   │   ├── s2s.py
│   │   ├── case_builder.py
│   │   ├── cloud_lake.py
│   │   └── tf_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── a0_unet.py
│   └── evaluation/
│       ├── __init__.py
│       └── recursive_diagnostic.py
├── tests/
│   ├── __init__.py
│   ├── test_01_cloud_lake.py
│   ├── test_02_rzsm.py
│   ├── test_03_temporal.py
│   ├── test_04_s2s.py
│   ├── test_05_compile_cube.py
│   ├── test_06_target_reconciliation.py
│   ├── test_07_case_calendar.py
│   ├── test_08_normalization_and_splits.py
│   ├── test_09_pilot_ladder.py
│   ├── test_10_case_builder.py
│   ├── test_11_tf_dataset.py
│   ├── test_12_a0_unet.py
│   ├── test_13_production_smoke_preflight.py
│   ├── test_14_validation_atmospheric_pipeline.py
│   ├── test_15_production_training_contracts.py
│   ├── test_16_production_case_builder.py
│   └── test_17_recursive_diagnostic.py
├── logs/
│   ├── a0_tiny_overfit_execution.json
│   └── A0_gpu_benchmark.json
├── figures/
│   ├── mindanao_spatial_grid_mesh.png
│   ├── mindanao_fractional_coverage_map.png
│   ├── mindanao_evaluation_mask_map.png
│   ├── mindanao_spatial_foundation_composite.png
│   ├── mindanao_rzsm_pilot_preprocessing_composite.png
│   ├── mindanao_era5_atmospheric_pilot_verification.png
│   ├── mindanao_production_cube_verification_composite.png
│   ├── mindanao_s2s_dynamic_predictor_composite.png
│   └── mindanao_s2s_pilot_case_and_recursive_inference.png
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
    │   └── mindanao_fraction_025.qml / mindanao_eval_mask_025.qml
    ├── atmospheric/
    │   └── pilot/
    │       └── era5_atmospheric_pilot_2015_01.nc
    ├── cases/
    │   └── pilot/
    │       └── CASE_2015*.npz (8 serialized pilot archives)
    ├── rzsm/
    │   ├── pilot/
    │   │   ├── era5_land_rzsm_pilot_2014_2015.nc
    │   │   └── README.md
    │   └── production/
    │       └── era5_land_rzsm_production_2015_2025.nc
    └── s2s/
        └── pilot/
            └── s2s_pilot_reforecast_w1_w2.nc
```

---

### 3.1 `contracts/` (Spatial, Mathematical & Governance Specifications)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml) | Spatial grid specification contract | Locks Candidate A coordinates (`lat`: 11.75 to 4.00, `lon`: 116.00 to 127.75, shape $32 \times 48$), resolution ($0.25^\circ$), authoritative boundary area ($99,948.76\text{ km}^2$), evaluation threshold ($f \ge 0.50$), and active cell count (126 cells: full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$). | `[FROZEN]` |
| [`VERIFICATION_STATUS.yaml`](../contracts/A0/VERIFICATION_STATUS.yaml) | Authoritative Milestone Verification Status Matrix | Single machine-readable source of truth for milestone statuses (21A–21K) and strict certification rules (`mock_counts_as_pass: false`, `estimated_counts_as_pass: false`, `sealed_test_access_allowed: false`). | `[ACTIVE / AUTHORITATIVE]` |
| [`metric_evaluation_contract.yaml`](../contracts/A0/metric_evaluation_contract.yaml) | Pre-Training Metric Evaluation Contract | Freezes mathematical definitions and aggregation rules for ACC (spatial anomaly correlation over 126 active cells), CRPSS (`CRPSS_B0` vs climatology; `CRPSS_A0` vs Lead-A0 baseline), and Brier Drought Score (2015–2021 training 20th percentile). | `[FROZEN]` |
| [`mindanao_a0_production_contract.yaml`](../contracts/A0/mindanao_a0_production_contract.yaml) | Pre-Training Production Model Contract | Freezes production hyperparameters (Adam, $\text{lr}=10^{-4}$, seeds $[42, 123, 456]$, nominal batch size $B=11$ amended to production $B=33$, loss weights $[0.2, 0.3, 0.5]$), per-lead parameter expectations, and reporting metrics. | `[FROZEN]` |
| [`normalization_parameters.yaml`](../contracts/A0/normalization_parameters.yaml) | Training-Only Active Domain Normalization Contract | Freezes min/max normalization parameters derived strictly from the 2015–2021 training partition across 126 active evaluation cells. Actively bound in `case_builder.py`. | `[FROZEN]` |
| [`A0_Mindanao_Baseline_Contract.yaml`](../contracts/A0/A0_Mindanao_Baseline_Contract.yaml) | Authoritative Post-Training Model A0 Baseline Contract | Machine-readable single source of truth for Model A0: seals Phase 21, locks 12 checkpoints, records 3-seed replicate performance summary, documents $B=33$ amendment, contextual parent reference comparison, and 2024–2025 sealed test quarantine. | `[CERTIFIED BASELINE]` |
| [`PHASE_23_DIAGNOSTIC_CONTRACT.yaml`](../contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml) | Phase 23 Pre-Analysis Specification & Diagnostic Contract | Pre-registered specification freezing Protocols 1–4, exact Oracle replacement table, deterministic member-wise $[0, 1]$ perturbation clipping rule, primary Moving-Block Bootstrap (MBB, $L=4, B=10000, p_{\text{boot}}$), supplementary Wilcoxon test, Holm-Bonferroni correction, and Post-A0 Gate 2 (G2-R) decision criteria. | `[PRE-ANALYSIS FROZEN]` |


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
| [`06_mindanao_datacube_and_anomaly_pipeline.ipynb`](../notebooks/06_mindanao_datacube_and_anomaly_pipeline.ipynb) | Production cube & anomaly pipeline | 16 cells. Compiles full 11-year ($2015\text{--}2025$) RZSM cube, fits locked 2015–2021 seasonal climatology, derives standardized anomalies, verifies census (506,268 points, zero NaNs/Infs), and exports 4-panel verification composite. | `[PASS]` |
| [`07_mindanao_s2s_and_pilot_case_assembly.ipynb`](../notebooks/07_mindanao_s2s_and_pilot_case_assembly.ipynb) | S2S Ingestion, ecCodes Equivalence Gate & UNET_RZSM Recursive Inference | 21 cells. Ingests clean 2015-01-16 S2S cycle, executes ecCodes side-by-side equivalence gate, harmonizes with zero step-0 fallbacks, builds multi-lead A0 case hierarchy, executes UNET_RZSM 4-lead recursive cascade, demonstrates downstream perturbation sensitivity, audits 126 active cells (0 NaNs/Infs), and generates dual 2x2 publication dashboards (`mindanao_s2s_dynamic_predictor_composite.png` and `mindanao_s2s_pilot_case_and_recursive_inference.png`). | `[PASS / VERIFIED]` |
| [`08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb`](../notebooks/08_mindanao_a0_tf_pipeline_and_checkpoint.ipynb) | Surrogate TensorFlow Pipeline, 11-Member Batching & Checkpoint Pipeline | 18 cells. Streams 8 pilot NPZ cases, verifies 11-member ensemble batching invariance, multi-head dictionary loss, optimizer step execution, and bit-for-bit checkpoint restoration using 2-layer Conv2D surrogate. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`09_mindanao_a0_vram_profiling.ipynb`](../notebooks/09_mindanao_a0_vram_profiling.ipynb) | Genuine Model A0 (`UNET_RZSM`) Hardware Profiling & VRAM Feasibility Benchmark | 17 cells. Physical GPU verification of the Six Technical Pillars on Google Colab Tesla T4 GPU (15,360 MB): genuine 1.63M-parameter instantiation, multi-lead forward pass with Candidate A ocean mask, real-model backward pass ($\|\Delta w\| = 3.95 \times 10^{-3}$), 4-lead recursive cascade, VRAM memory ladder ($B \in \{11, 22, 33, 44, 66\}$), and production contract freeze ($0.00\text{e}+00$ parity). | `[PASS / CERTIFIED_ON_GPU: 2026-09-17]` |
| [`10_mindanao_validation_atmospheric_pipeline.ipynb`](../notebooks/10_mindanao_validation_atmospheric_pipeline.ipynb) | Validation Atmospheric Pipeline Execution & Dataset Certification (2022–2023) | 21 cells. Execution wrapper and audit record orchestrating 2022–2023 atmospheric mirroring via ARCO-ERA5, batch 5-channel derivation, local file census, authoritative verification via `scripts/15_verify_validation_atmospheric_pipeline.py --mode live`, three-way normalization audit review, spatial sanity inspection, telemetry archival, and Executive Findings & Scientific Certification Summary. | `[PASS / CERTIFIED: 2026-09-16]` |
| [`11_mindanao_a0_production_smoke_preflight.ipynb`](../notebooks/11_mindanao_a0_production_smoke_preflight.ipynb) | Pre-Training GPU Smoke Preflight — End-to-End Verification of Model A0 Training Pipeline | 14 cells. Validates 5 core production stages on physical NVIDIA Tesla T4 GPU: Stage A (4-lead real backward pass on `CASE_20150115_W01.npz`, finite gradients $\|\nabla_\theta\| > 0$, $\|\Delta w\| = 3.95 \times 10^{-3}$), Stage B (recursive autoregressive cascade $W_1 \to \hat{y}_1 \dots \to W_4$, exact indices $W_2=11, W_3=[3,4], W_4=[3,4,5]$, permutation tamper detection delta 0.2135), Stage C (production multi-head deep supervision loss with downstream 126-cell masking, domain discrepancy 0.5646 / 0.6793), Stage D (checkpoint parity scoping, model-weight parity $0.00\text{e}+00$, verified step-2 trajectory equality: loss delta $0.00\text{e}+00$, weight delta $< 10^{-6}$), Stage E (authoritative CLI engine execution with exit code 0, zero fallbacks, and telemetry synchronization), and Executive Findings & Scientific Certification Summary. | `[PASS / CERTIFIED_ON_GPU: 2026-09-17]` |
| [`12_mindanao_a0_production_training.ipynb`](../notebooks/12_mindanao_a0_production_training.ipynb) | Model A0 Baseline Production Training — Full Three-Seed Training across all 4 Subseasonal Leads (W1–W4) | 22 cells. Interactive Google Colab execution wrapper and audit record orchestrating full three-seed (`[42, 123, 456]`), 4-lead ($W_1 \to W_4$) recursive production training. Configured for physical NVIDIA Tesla T4 GPU with GCS lake synchronization, training curve diagnostics, bit-for-bit checkpoint restore verification, multi-seed performance synthesis, automated cloud lake sync (`gs://rise-unet-rzsm/checkpoints/A0/`), and Executive Findings & Scientific Certification Summary. | `[PASS / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]` |
| [`13_mindanao_recursive_degradation_diagnostic.ipynb`](../notebooks/13_mindanao_recursive_degradation_diagnostic.ipynb) | Phase 23 Recursive Degradation Diagnostic & Post-A0 Gate 2 (G2-R) Decision | 20 cells. Multi-protocol evaluation (Protocols 1, 2, 4) across 194 validation cases, primary Moving-Block Bootstrap significance ($p_{\text{boot}}$), supplementary Wilcoxon test, Holm-Bonferroni correction, Post-A0 Gate 2 (G2-R) GO / NO-GO evaluation, and Executive Findings & Scientific Certification Summary. | `[PRE-ANALYSIS FROZEN / READY FOR GPU EXECUTION]` |

---

### 3.5 `reproduction_audit/` (Authoritative Audit Dossiers & Decision Logs)

| File Link | Step / Focus | Scope & Empirical Content | Verdict / Decision |
| :--- | :--- | :--- | :---: |
| [`01_parent_baseline_replication_audit.md`](01_parent_baseline_replication_audit.md) | Gate 1A | Parent code reproduction, environment lock, and baseline execution verification. | `[VERIFIED]` |
| [`01a_parent_baseline_execution_log.md`](01a_parent_baseline_execution_log.md) | Gate 1A | Complete terminal execution logs and runtime traces for parent baseline. | `[PASS]` |
| [`02_parent_experiment_trace_audit.md`](02_parent_experiment_trace_audit.md) | Gate 1B | Parent EX29 experiment audit: recursive loop mechanics, input/target window alignment. | `[VERIFIED]` |
| [`02a_parent_experiment_trace_log.md`](02a_parent_experiment_trace_log.md) | Gate 1B | Terminal execution logs verifying EX29 model convergence and evaluation outputs. | `[PASS]` |
| [`06_mindanao_boundary_and_gadm_audit.md`](06_mindanao_boundary_and_gadm_audit.md) | Step 21A.2 | Administrative boundary extraction from PSA shapefile; WGS84 geodesic area quadrature ($99,948.76\text{ km}^2$). | `[ACCEPTED]` |
| [`03_parent_variable_contract_audit.md`](03_parent_variable_contract_audit.md) | Step 21A.3 | Variable mapping, lag schedule ($[-1, -7, -14]$), and channel counts ($W_1=11, W_2=12, W_3=5, W_4=6$). | `[VERIFIED]` |
| [`05_mindanao_geometry_envelopes_audit.md`](05_mindanao_geometry_envelopes_audit.md) | Step 21A.4 | Comparative geometric analysis of Candidate A ($32 \times 48$) vs Candidate B envelopes. | `[ACCEPTED]` |
| [`07_mindanao_candidate_a_grid_freeze_audit.md`](07_mindanao_candidate_a_grid_freeze_audit.md) | Step 21A.6 | Formal freeze of Candidate A regular $0.25^\circ$ mesh with 16-divisible dimensions for UNet pooling. | `[ACCEPTED]` |
| [`08_mindanao_evaluation_mask_audit.md`](08_mindanao_evaluation_mask_audit.md) | Step 21A.7B | Ellipsoidal fractional land raster and binary evaluation mask (126 active cells: full-cell footprint $96,085.57\text{ km}^2$, boundary-intersection area $86,418.83\text{ km}^2$ or $86.46\%$). | `[ACCEPTED]` |
| [`12_era5_land_target_acceptance_audit.md`](12_era5_land_target_acceptance_audit.md) | Step 21A.9 & 21B.2 | Scientific defense of ERA5-Land vs GLEAM, hourly source-of-record, and RZSM physics. | `[ACCEPTED]` |
| [`09_mindanao_spatial_foundation_dossier.md`](09_mindanao_spatial_foundation_dossier.md) | Sub-Phase 21A Synthesis | Master synthesis dossier certifying area conservation ($0.0000\%$ quadrature error) and spatial foundation freeze. | `[PASS]` |
| [`10_era5_land_archive_completeness_audit.md`](10_era5_land_archive_completeness_audit.md) | Step 21B.1 | Census of 264 synchronized ERA5-Land NetCDF files covering 2015–2025 nominal archive (3.67 GiB in GCS). | `[PASS]` |
| [`11_era5_land_pilot_rzsm_preprocessing_audit.md`](11_era5_land_pilot_rzsm_preprocessing_audit.md) | Step 21B.3–4 | Pilot RZSM calculation across 20-day antecedent window; zero NaNs across $20 \times 126 = 2,520$ samples. | `[PASS]` |
| [`15_era5_atmospheric_preprocessing_audit.md`](15_era5_atmospheric_preprocessing_audit.md) | Step 21C.1–2 | Pilot extraction of 5 atmospheric channels for Jan 2015 via Bolton (1980); zero NaNs across 3,906 evaluation points. | `[PASS / VERIFIED / ACCEPTED]` |
| [`04_parent_temporal_target_parity_audit.md`](04_parent_temporal_target_parity_audit.md) | Step 21D.3 | Target reconciliation ($L=[6,13,20,27]$), trailing rolling mean, 3-month season climatology, domain-wide normalization, and 24/24 unit tests. | `[VERIFIED]` / `[ACCEPTED]` |
| [`13_production_cube_preflight_audit.md`](13_production_cube_preflight_audit.md) | Step 21D.4-PREFLIGHT | Formal preflight verification report: census of 265 NetCDFs (96,912 hours, 4,038 days), zero NaNs/Infs, spatial coordinate contract lock. | `[PASS]` |
| [`14_production_cube_compilation_and_census_audit.md`](14_production_cube_compilation_and_census_audit.md) | Step 21D.4 | Formal production cube compilation audit: multi-year ERA5-Land ingestion, 506,268 finite evaluations, zero NaNs/Infs, dual GCS synchronization. | `[PASS / VERIFIED / ACCEPTED]` |
| [`16_ecmwf_s2s_pilot_and_recursive_inference_audit.md`](16_ecmwf_s2s_pilot_and_recursive_inference_audit.md) | Step 21F.3 | Formal pilot case assembly and recursive inference audit: real S2S integration, perturbation sensitivity, 4-lead cascade. | `[PASS / VERIFIED]` |
| [`17_pilot_case_ladder_and_manifest_audit.md`](17_pilot_case_ladder_and_manifest_audit.md) | Step 21G | 8-case pilot ladder manifest and census audit: 88 member rows, 8 summary rows, 314,496 finite feature values, 0 NaNs/Infs. | `[PASS / VERIFIED]` |
| [`20_tf_dataset_pipeline_and_checkpoint_audit.md`](20_tf_dataset_pipeline_and_checkpoint_audit.md) | Step 21H | Surrogate TensorFlow pipeline audit: 11-member ensemble grouping semantics, CRPS loss mathematical reconciliation, 5-epoch stability, bit-for-bit checkpoint restore for 2-layer surrogate. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`22_a0_tiny_overfit_gradient_audit.md`](22_a0_tiny_overfit_gradient_audit.md) | Step 21I | Surrogate tiny-data optimization test: 8 cases, 40 epochs, 320 updates on 2-layer Conv2D surrogate model; 99% error reduction is an unnormalized predictor scale artifact, not A0 capacity. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`21_vram_and_hardware_profiling_audit.md`](21_vram_and_hardware_profiling_audit.md) | Sub-Phase 21J | Hardware profiling & VRAM feasibility audit on physical Tesla T4 GPU; all 6 technical pillars certified pass under strict fail-closed criteria with cloud lake parity. | `[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]` |
| [`18_production_case_calendar_1154_cycles_audit.md`](18_production_case_calendar_1154_cycles_audit.md) | Step 21K.1 | Usable production case calendar audit: 1,154 operational cycles, 4-way data intersection, 735 Train / 210 Val / 209 Test partitioning, dual GCS parity. | `[PASS / VERIFIED / ACCEPTED]` |
| [`19_dataset_splits_and_normalization_contract_audit.md`](19_dataset_splits_and_normalization_contract_audit.md) | Step 21K.2 | Dataset split partitioning manifest and training-only normalization parameter audit: 735 Train / 210 Val / 209 Sealed Test, four-part scope, production contract freeze, 83 unit tests passing, dual GCS parity. | `[PASS / VERIFIED / ACCEPTED]` |
| [`23_production_smoke_preflight_audit.md`](23_production_smoke_preflight_audit.md) | Step 21K.3-pre | Pre-Production Gate 3 production smoke preflight audit on physical Tesla T4 GPU: 5 stages (4-lead genuine backward pass on real data, recursive cascade on model inferences, downstream 126-cell masking, trajectory roundtrip parity $< 5\times 10^{-6}$, and authoritative fail-closed engine execution). Step 21K.3 production training AUTHORIZED. | `[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]` |
| [`24_s2s_provider_availability_and_cohort_census_audit.md`](24_s2s_provider_availability_and_cohort_census_audit.md) | Step 21K.3-census | S2S provider availability audit & cohort census reconciliation: mathematical proof of 74 provider archive gaps (58 Train + 16 Val, `MarsNoDataError`), 677 Train & 194 Val active cases, fail-closed census assertion, and dual GCS sync. | `[PASS / VERIFIED / ACCEPTED: 2026-09-17]` |
| [`25_a0_production_training_and_parity_audit.md`](25_a0_production_training_and_parity_audit.md) | Step 21K.3–5 | Model A0 production training audit & baseline certification: 12 training runs completed, 3-seed replicate performance summary, 871 usable cohort accounting, Parent EX29 Reference Comparison, and sealed test quarantine. Phase 21 officially sealed. | `[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]` |
| [`OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md) | Governance | Master operational execution matrix, living directory registry, and file map (This Document). | `[APPROVED]` |

---

### 3.6 `scripts/` (Operational Processing & Automation)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`09_build_pilot_manifest_and_cases.py`](../scripts/09_build_pilot_manifest_and_cases.py) | Pilot Case Hierarchy Assembler | Batch compiles the 8 pilot cases (`CASE_20150115_W01.npz` through `CASE_20150304_W08.npz`) and writes 88-row member and 8-row summary manifests. | `[PASS / VERIFIED]` |
| [`06_build_production_case_calendar.py`](../scripts/06_build_production_case_calendar.py) | Production Case Calendar Generator | Generates the 1,154 operational forecast issuance cycles (2015–2025), computes deterministic lags/targets, evaluates 4-way data intersection, and exports `manifests/production_case_calendar.csv`. | `[PASS / VERIFIED / ACCEPTED]` |
| [`08_derive_training_normalization.py`](../scripts/08_derive_training_normalization.py) | Training Normalization Engine | Derives domain-wide active scalar normalization bounds strictly over 2015–2021 training partition across 126 active cells and exports `contracts/A0/normalization_parameters.yaml`. | `[PASS / VERIFIED / ACCEPTED]` |
| [`07_generate_dataset_splits.py`](../scripts/07_generate_dataset_splits.py) | Dataset Splits Generator | Partitions 1,154 production cycles into TRAIN (735), VAL (210), and SEALED_TEST (209) manifests with SHA-256 digests and leakage verification. | `[PASS / VERIFIED / ACCEPTED]` |
| [`04_download_all_s2s_production.py`](../scripts/04_download_all_s2s_production.py) | Production S2S Downloader | High-efficiency S2S downloader for all 385 cycles (2015–2025) featuring startup GCS caching, leap-year safety, and scratch cleanup. | `[IN PROGRESS]` |
| [`02_download_era5_atmospheric.py`](../scripts/02_download_era5_atmospheric.py) | Copernicus CDS-Beta API Downloader | Resilient multi-threaded downloader retrieving hourly ERA5 atmospheric pressure levels in monthly chunks with automatic retry and GCS synchronization. | `[IN PROGRESS]` (168/264 files secured for training fold Jan 2015–Dec 2021; 2022–2025 mirroring active) |
| [`01_generate_mindanao_masks.py`](../scripts/01_generate_mindanao_masks.py) | Geodesic Mask Generator | Python script computing ellipsoidal polygon intersections on WGS84 to produce Candidate A grid, fractional land raster, and 126-cell binary evaluation mask. | `[PASS]` |
| [`11_profile_a0_vram_benchmark.py`](../scripts/11_profile_a0_vram_benchmark.py) | Standalone A0 VRAM & Hardware Profiler | Executes the Six Technical Pillars for genuine `UNET_RZSM` (1.63M parameters), multi-lead forward pass, backward pass, 4-lead cascade, VRAM ladder ($B \in \{11, 22, 33, 44, 66\}$), and real model-weight checkpoint roundtrip under fail-closed certification (`--mode certify`). | `[PASS / VERIFIED / ACCEPTED: 2026-09-17: CERTIFIED_ON_GPU]` |
| [`run_download_atmospheric.sh`](../scripts/run_download_atmospheric.sh) | Shell Execution Wrapper | Headless background execution wrapper for atmospheric retrieval on Unix/WSL environments. | `[READY]` |
| [`12_train_a0_tiny_overfit.py`](../scripts/12_train_a0_tiny_overfit.py) | Surrogate Pipeline Smoke Test Script | 40-epoch surrogate diagnostic verifying data streaming, 11-member batching, target broadcasting, and checkpoint restore parity across 320 updates. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`13_train_a0_pipeline_checkpoint.py`](../scripts/13_train_a0_pipeline_checkpoint.py) | Surrogate TensorFlow Pipeline Diagnostic | Diagnostic script validating multi-worker streaming, 11-member batch invariant, and checkpoint serialization for 2-layer surrogate. | `[PASS / VERIFIED FOR SURROGATE]` |
| [`03_derive_era5_atmospheric_daily.py`](../scripts/03_derive_era5_atmospheric_daily.py) | Atmospheric Derivation Engine | Standalone and batch script computing Bolton (1980) specific humidity, daily temperature extremes ($T_{\max}, \Delta T$), and geopotential height ($Z_{200}/g_0$). Supports both January 2015 pilot and multi-year batch derivation (2022–2023) with GCS sync. | `[PASS]` |
| [`10_verify_pilot_ladder_provenance_and_census.py`](../scripts/10_verify_pilot_ladder_provenance_and_census.py) | Pilot Census & Provenance Engine | Independent 16-point audit verifying all 8 pilot cases, 88 member realizations, zero NaNs across 314,496 points, and GCS lake parity. | `[PASS / VERIFIED]` |
| [`05_verify_production_cube_preflight.py`](../scripts/05_verify_production_cube_preflight.py) | Production Cube Preflight Engine | Preflight engine verifying complete date coverage across all 4,038 days, zero NaNs/Infs, spatial coordinate contract lock, and dual GCS parity. | `[PASS / VERIFIED / ACCEPTED]` |
| [`14_run_a0_production_smoke_test.py`](../scripts/14_run_a0_production_smoke_test.py) | Model A0 Production Training Smoke Test | End-to-end preflight verifying 4-lead genuine backward pass & parameter updates ($C \in \{11, 12, 5, 6\}$), 3 deep supervision heads, evaluation-domain masking, recursive channel semantics and tamper detection, and model-weight vs full training-state checkpointing. | `[PASS / CERTIFIED PREFLIGHT]` |
| [`15_verify_validation_atmospheric_pipeline.py`](../scripts/15_verify_validation_atmospheric_pipeline.py) | Validation Atmospheric Pipeline Verification Engine | Authoritative certification engine verifying complete continuous 730-day calendar coverage (2022–2023, 0 missing/duplicate days), fail-closed 24-file monthly whitelist, 210 validation cycles, 5-channel presence and ordering (`pwat, spfh, tmax, diff_temp, hgt_pres`), Candidate A grid conformity ($32 \times 48$), 126 active cell finiteness, three-way normalization audit (Checks A, B, C), production `CaseBuilder` ingestion into `CaseTensorHierarchy` ($[11, 12, 5, 6]$ channels, normalization, zero ocean buffer), and telemetry JSON export (`logs/gate1_validation_atmospheric_execution.json`). Supports `--mode live` and `--mode mock`. | `[PASS / VERIFIED]` |
| [`16_train_a0_production.py`](../scripts/16_train_a0_production.py) | Step 21K.3 Full Three-Seed Model A0 Production Training Engine | Standalone training engine orchestrating the multi-seed (`[42, 123, 456]`), multi-lead ($W_1 \to W_4$) recursive cascade. Enforces genuine `UNET_RZSM` architecture (1,627,139 / 1,630,307 / 1,608,131 / 1,611,299 params), spatial CRPS loss with deep-supervision weights `[0.2, 0.3, 0.5]`, downstream 126-cell land masking, Adam optimizer ($\eta=10^{-4}$), ReduceLROnPlateau, early stopping (patience 8), minimum validation CRPS checkpoint selection, and automated GCS lake synchronization (`gs://rise-unet-rzsm/checkpoints/A0/`). | `[AUTHORIZED / PRODUCTION READY]` |
| [`17_build_production_cases.py`](../scripts/17_build_production_cases.py) | Step 21K Production Case Assembly Engine | Standalone and batch pipeline assembling, normalizing, validating, and serializing all production forecast case tensors (`processed/cases/production/CASE_*.npz`) across Train (2015–2021) and Validation (2022–2023) splits. Enforces the frozen training normalization contract (`contracts/A0/normalization_parameters.yaml`), strict $[11, 12, 5, 6]$ multi-lead channel topology, 126-cell majority-land evaluation masking, zero ocean buffer filling, on-demand cloud lake synchronization, and SHA-256 manifest registration (`manifests/cases_production_summary.csv`). | `[PASS / VERIFIED]` |
| [`18_verify_production_training_artifacts.py`](../scripts/18_verify_production_training_artifacts.py) | Model A0 Production Training Artifact Verifier | Verifies 12 model checkpoints, dual GCS cloud lake synchronization, loss logs, and figure checksums across local repo and cloud lake. | `[PASS / VERIFIED]` |
| [`19_run_recursive_degradation_diagnostic.py`](../scripts/19_run_recursive_degradation_diagnostic.py) | Phase 23 Recursive Degradation Diagnostic CLI | Authoritative execution CLI for Phase 23: evaluates Protocols 1, 2, 4 across Seeds 42, 123, 456, enforces $[0, 1]$ clipping and member identity preservation, computes primary MBB bootstrap inference ($p_{\text{boot}}$), supplementary Wilcoxon test, Holm-Bonferroni correction, and Post-A0 Gate 2 (G2-R) decision. Supports `--mode certify` and `--mode execute`. | `[PASS / PRE-ANALYSIS FROZEN]` |

---

### 3.7 `src/` (Production Python Package)

| File Link | Primary Purpose | Technical Description | Status / Outputs |
| :--- | :--- | :--- | :---: |
| [`src/__init__.py`](../src/__init__.py) | Package Root | Top-level package initializer. | `[ACTIVE]` |
| [`src/data/__init__.py`](../src/data/__init__.py) | Data Package Exports | Exports `compute_depth_weighted_rzsm`, `remap_era5_land_to_candidate_a`, `compute_trailing_rolling_mean`, `compute_training_climatology`, `compute_seasonal_anomalies`, `fit_min_max_bounds`, `standardize_with_training_bounds`, `compile_production_rzsm_pipeline`, `verify_production_cube_census`, `FastLandAwareRemapper`, `process_era5_land_monthly_pair`, `process_era5_land_antecedent_file`, `compile_full_11yr_rzsm_cube`, `create_a0_dataset_from_manifest`, `save_a0_checkpoint`, `restore_a0_checkpoint`, and `ProductionCubeConfig`. | `[ACTIVE]` |
| [`src/data/rzsm.py`](../src/data/rzsm.py) | RZSM Calculation & Remapping | Implements depth-weighted RZSM formula ($0.07\cdot\text{SM}_1 + 0.21\cdot\text{SM}_2 + 0.72\cdot\text{SM}_3$). Remapping order: 1) Isolate finite land points on native ERA5-Land ($0.10^\circ$), 2) Bilinear interpolation over active evaluation cells ($M_{i,j}=1$), 3) Nearest-neighbor extrapolation fallback triggered specifically for unassigned coastal boundary cells to prevent coastline clipping, 4) Binary evaluation mask application, and 5) Zero-filling of 1,410 inactive computational cells. | `[PASS]` / `[ACCEPTED]` |
| [`src/data/temporal.py`](../src/data/temporal.py) | Temporal Preprocessing | Implements 7-day backward trailing rolling mean (`center=False`), locked 3-month seasonal climatology (DJF, MAM, JJA, SON) fitted on training years $\le 2021$, seasonal anomalies, and domain-wide active scalar normalization. | `[PASS]` / `[VERIFIED]` |
| [`src/data/compile_cube.py`](../src/data/compile_cube.py) | Production Cube Engine | Production compilation engine with vectorized multi-day spatial remapper (`FastLandAwareRemapper`), monthly pair ingestion (`process_era5_land_monthly_pair`), antecedent support processor (`process_era5_land_antecedent_file`), and end-to-end multi-year compilation (`compile_full_11yr_rzsm_cube`). Features automated date uniqueness and canonical endpoint validation (`2014-12-12` to `2025-12-31`), orchestrates temporal transformations on full 4,038-day archive series, slices nominal 4,018-day period (506,268 evaluation cell-days), and executes automated census certification (zero NaNs/Infs). | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |
| [`src/data/s2s.py`](../src/data/s2s.py) | ECMWF S2S Harmonization Engine | Pure-Python GRIB2 Section 7 Template 0 decoder and harmonizer. Ingests verified EX29 dynamic triplet (`t2m, d2m, tcw`), performs 6-hourly step aggregation to Week 1 ($0\text{--}168\text{ h}$) and Week 2 ($168\text{--}336\text{ h}$) leads, executes bilinear remapping with nearest-boundary extrapolation to Candidate A 0.25° grid, structures all 11 ensemble members ($M=11$), and applies evaluation masking. | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |
| [`src/data/case_builder.py`](../src/data/case_builder.py) | Multi-Lead Tensor Hierarchy Assembler | Production case assembler constructing the multi-lead tensor hierarchy $[M=11, H=32, W=48, C_k]$ across leads $W_1..W_4$ adhering strictly to Kyle Lesinger's channel counts ($[11, 12, 5, 6]$). Implements recursive prior-lead prediction concatenation and ground truth target slicing with zero future leakage. | `[PASS]` / `[VERIFIED]` / `[ACCEPTED]` |
| [`src/data/tf_dataset.py`](../src/data/tf_dataset.py) | TensorFlow Streaming & Batching Engine | Production streaming pipeline enforcing 11-member ensemble batching invariance ($B \in 11\mathbb{Z}^+$), case-level shuffling, target broadcasting, multi-worker prefetching, and checkpoint persistence. | `[PASS / VERIFIED]` |
| [`src/models/__init__.py`](../src/models/__init__.py) | Model Package Root | Exports `build_a0_unet` and Model A0 architecture specifications. | `[ACTIVE]` |
| [`src/models/a0_unet.py`](../src/models/a0_unet.py) | Genuine Model A0 (`UNET_RZSM`) Architecture Factory | Authoritative factory instantiating the authentic 1,630,307-parameter nested U-Net (`function/modelRzsmRelu.py`) with 298 weight tensors, Inception blocks, SE attention, multiscale decoders, multi-head deep supervision outputs, and channel schedule ($W_1=11, W_2=12, W_3=5, W_4=6$). | `[PASS / VERIFIED]` |
| [`src/evaluation/__init__.py`](../src/evaluation/__init__.py) | Evaluation Package Root | Exports Phase 23 diagnostic and statistical inference modules. | `[ACTIVE]` |
| [`src/evaluation/recursive_diagnostic.py`](../src/evaluation/recursive_diagnostic.py) | Recursive Degradation Diagnostic Engine | Core analytical library implementing Oracle counterfactual prior assembly, deterministic member-wise boundary clipping ($y' = \operatorname{clip}(y + \epsilon, 0, 1)$), case-level metric evaluation, paired differences, moving-block bootstrap (MBB with $p_{\text{boot}}$), paired Wilcoxon test, Holm-Bonferroni correction, and Post-A0 Gate 2 (G2-R) evaluator. | `[PASS / PRE-ANALYSIS FROZEN]` |

---

### 3.8 `tests/` (Automated Unit Test Suite)

All 114 automated unit tests are executed with `python -m unittest discover -s tests -p "test_*.py"`. **108 tests passed, 6 skipped, 0 failed, 0 errors:**

> [!NOTE]
> All unit tests conform to the strict zero-failure baseline. Skipped tests represent deliberate runtime environment constraints (1 offline GCS network test, 3 TF-gated graph preflight tests, 2 hardware-specific GPU benchmarks), preserving fail-closed test architecture.

| File Link | Test Count | Key Test Assertions | Execution Time / Status |
| :--- | :---: | :--- | :---: |
| [`test_17_recursive_diagnostic.py`](../tests/test_17_recursive_diagnostic.py) | 11 | 1. Oracle counterfactual prior assembly and tensor shapes $[11, 12, 5, 6]$.<br>2. Integration with `prepare_case_lead_tensors`.<br>3. Boundary clipping $y' = \operatorname{clip}(y + \epsilon, 0, 1)$ and active cell census.<br>4. Case-level metric calculation (MAE, RMSE, ACC, exact CRPS, proxy CRPS).<br>5. Paired difference calculations and Cohen's $d_z$ effect size.<br>6. Moving-block bootstrap (MBB) point estimates, 95% CIs, and $p_{\text{boot}}$.<br>7. Paired Wilcoxon signed-rank test behavior.<br>8. Step-down Holm-Bonferroni multiple testing adjustment.<br>9. Post-A0 Gate 2 (G2-R) GO / NO-GO rule evaluator.<br>10. Cross-contract reconciliation between Phase 23 Contract and Baseline Contract A0.<br>11. Monotonicity evaluation on perturbation magnitude $|\epsilon|$ separately by sign ($+0.05 \to +0.10$ and $-0.05 \to -0.10$). | 0.17s / `[PASS]` |
| [`test_16_production_case_builder.py`](../tests/test_16_production_case_builder.py) | 1 | 1. Production case tensor hierarchy shape assertions ($[11, 12, 5, 6]$).<br>2. Active evaluation domain masking (126 cells) and ocean zero-filling.<br>3. Frozen normalization parameter bounds in $[0.0, 1.0]$.<br>4. Target dates offsets ($+6\text{d}, +13\text{d}, +20\text{d}, +27\text{d}$). | 0.53s / `[PASS]` |
| [`test_01_cloud_lake.py`](../tests/test_01_cloud_lake.py) | 1 | 1. Local codebase artifact resolution with zero network dependency.<br>2. Formatted GCS URI construction targeting authoritative lake `gs://rise-unet-rzsm/`.<br>3. Informative error handling when cloud artifacts are unavailable. | 0.01s / `[PASS]` |
| [`test_08_normalization_and_splits.py`](../tests/test_08_normalization_and_splits.py) | 8 | 1. Split disjointness (zero date overlap across TRAIN, VAL, TEST).<br>2. Split completeness (735 Train + 210 Val + 209 Test = 1,154 total).<br>3. Strict chronological ordering without time inversion.<br>4. Sealed test quarantine integrity (202 queued + 7 out-of-bounds).<br>5. Normalization YAML contract validity and $[0, 1]$ bounds.<br>6. Production training contract schema validity.<br>7. Executable normalization tensor-builder contract test proving active consumption, $[0.0, 1.0]$ bounds, and $0.0$ ocean buffer. | 0.25s / `[PASS]` / `[VERIFIED]` |
| [`test_07_case_calendar.py`](../tests/test_07_case_calendar.py) | 4 | 1. Exactly 1,154 operational cycles indexed monotonically.<br>2. Strict partition splits (735 Train, 210 Val, 209 Sealed Test).<br>3. Deterministic lag arithmetic ($-1\text{d}, -7\text{d}, -14\text{d}$) and target offsets ($+6\text{d}, +13\text{d}, +20\text{d}, +27\text{d}$).<br>4. Isolation of exactly 7 boundary-clipped cycles in Dec 2025. | 0.04s / `[PASS]` / `[VERIFIED]` |
| [`test_05_compile_cube.py`](../tests/test_05_compile_cube.py) | 5 | 1. End-to-end multi-year pipeline execution and census certification.<br>2. Out-of-sample temporal leakage isolation ($\ge 2022$ cannot alter training bounds).<br>3. `FastLandAwareRemapper` exact numerical agreement on tested data (observed maximum absolute difference = 0.0; test tolerance = 1e-6) with reference remapper.<br>4. 2014 antecedent support file real data processing (20 days, finite on 126 cells, 0 on 1410 buffer).<br>5. Partial archive compilation pipeline orchestration. | 4.8s / `[PASS]` |
| [`test_02_rzsm.py`](../tests/test_02_rzsm.py) | 10 | 1. Layer weights sum to $1.0$.<br>2. Constant field preserves soil moisture.<br>3. Hand-computable analytical solution ($2.65$).<br>4. Non-unit weight exception.<br>5. Land mask application and ocean zero-filling.<br>6. Min-max scaling to $[0, 1]$.<br>7. Backward rolling mean mechanics.<br>8. Antecedent lag extraction ($[-1, -7, -14]$).<br>9. Real pilot NetCDF physical range $[0.10, 0.60]\,\text{m}^3/\text{m}^3$.<br>10. Land-aware bilinear remapping with zero NaNs across evaluation cells. | 1.1s / `[PASS]` |
| [`test_06_target_reconciliation.py`](../tests/test_06_target_reconciliation.py) | 3 | 1. Hand-calculated arithmetic verification.<br>2. Exact parity with parent formula $L = (\text{lead} \times 7) - 1$ on anchor dates (`2015-01-15`, `2015-06-01`, `2016-02-29`, `2018-08-15`, `2020-12-01`).<br>3. Mathematical proof of contiguous, non-overlapping 28-day partition across $W_1$–$W_4$. | 0.8s / `[PASS]` / `[VERIFIED]` |
| [`test_03_temporal.py`](../tests/test_03_temporal.py) | 9 | 1. Rolling mean future invariance (zero leakage from $t+1$).<br>2. Trailing window arithmetic values.<br>3. Climatology strictly ingests years $\le 2021$.<br>4. 3-month seasonal climatology conforms to DJF, MAM, JJA, SON.<br>5. Training period mean anomaly equals zero ($\pm 10^{-6}$).<br>6. Domain-wide active scalar normalization bounds.<br>7. Exact lag and lead date extraction.<br>8. Symmetric continuous series extraction for targets and antecedents.<br>9. Integration test on real 31-day pilot NetCDF data. | 2.0s / `[PASS]` / `[VERIFIED]` |
| [`test_04_s2s.py`](../tests/test_04_s2s.py) | 7 | 1. Pure-Python GRIB2 Section 7 Template 0 constant field unpacking.<br>2. Known synthetic bitstream decoding.<br>3. Bilinear remapping geometry from $(5, 8)$ to Candidate A $(32, 48)$ with zero NaNs.<br>4. Monotonic temperature gradient preservation.<br>5. Pilot cycle harmonization with explicit fallback opting and diagnostic flag.<br>6. Production mode hard-fail enforcement rejecting missing forecast steps.<br>7. Clean ECDS production GRIB harmonization with zero fallbacks, separate hdate (`2015-01-16`) and model date (`2020-01-16`). | 2.2s / `[PASS]` / `[VERIFIED]` |
| [`test_10_case_builder.py`](../tests/test_10_case_builder.py) | 5 | 1. Single-case tensor hierarchy shapes $[11, 12, 5, 6]$ across $M=11$ members.<br>2. Zero-filling of non-evaluation ocean cells across inputs and targets.<br>3. Incomplete antecedent window exception enforcement.<br>4. Real pilot Jan 15, 2015 case integration test.<br>5. Recursive normalization invariant test proving prior model predictions are not double-normalized. | 1.8s / `[PASS]` / `[VERIFIED]` |
| [`test_09_pilot_ladder.py`](../tests/test_09_pilot_ladder.py) | 4 | 1. Manifest structure and column integrity.<br>2. Temporal ordering and 7-day interval consistency.<br>3. Member indexing $[0..10]$ and case grouping invariance.<br>4. Missing file and corrupted case exception handling. | 1.2s / `[PASS]` / `[VERIFIED]` |
| [`test_11_tf_dataset.py`](../tests/test_11_tf_dataset.py) | 16 | 1. Batch size exact multiple of 11 enforcement.<br>2. Member realization order preservation.<br>3. Case-level shuffle integrity (members stay clustered).<br>4. Target broadcasting invariance ($0.00 \times 10^0$).<br>5. Multi-head output dictionary matching UNET_RZSM heads.<br>6. Checkpoint save, restore, and state parity. | 1.5s / `[PASS]` / `[VERIFIED]` |
| [`test_12_a0_unet.py`](../tests/test_12_a0_unet.py) | 6 | 1. Grid dimensions & Candidate A geometry guard (divisibility by 16).<br>2. Multi-lead channel schedule $[11, 12, 5, 6]$.<br>3. Invalid lead rejection.<br>4. Expected per-lead parameter counts ($W_1=1,627,139; W_2=1,630,307; W_3=1,608,131; W_4=1,611,299$).<br>5. Conditional TensorFlow-gated genuine architecture instantiation, 298 weight tensors, 3 deep supervision heads.<br>6. Multi-lead input shapes, forward pass execution, and output finiteness. | 0.9s / `[PASS]` / `[VERIFIED]` |
| [`test_13_production_smoke_preflight.py`](../tests/test_13_production_smoke_preflight.py) | 4 | 1. Recursive channel semantics and exact index placement across Leads 2, 3, 4.<br>2. Recursive channel tamper and order scrambling detection.<br>3. Full training-state checkpoint serialization and restoration (model weights, optimizer weights, epoch, step).<br>4. 4-Lead GradientTape backpropagation and parameter update with evaluation domain masking. | 0.05s / `[PASS]` |
| [`test_14_validation_atmospheric_pipeline.py`](../tests/test_14_validation_atmospheric_pipeline.py) | 9 | 1. Validation manifest census (210 cases: 105 in 2022, 105 in 2023).<br>2. Evaluation mask geometry and active cell count.<br>3. Clean pipeline flow across all 210 validation dates.<br>4. Missing variable hard-failure detection.<br>5. Active cell NaN/Inf injection detection.<br>6. Missing date detection.<br>7. Three-way normalization diagnostics (finite math, unclipped range & excursions, and contracted post-clipping).<br>8. Genuine production `CaseBuilder` ingestion into `CaseTensorHierarchy` ($[11, 12, 5, 6]$ channels, normalization, zero ocean buffer).<br>9. Complete continuous 730-day daily calendar check and duplicate/missing date fail-closed detection. | 3.5s / `[PASS]` |
| [`test_15_production_training_contracts.py`](../tests/test_15_production_training_contracts.py) | 3 | 1. Expected per-lead parameter counts ($W_1: 1,627,139; W_2: 1,630,307; W_3: 1,608,131; W_4: 1,611,299$) and channel schedule.<br>2. Graph-differentiable `crps2d_tf` loss execution under GradientTape with active land masking.<br>3. Ocean buffer perturbation invariance test ($0.00 \times 10^0$ sensitivity to non-evaluation cells). | 0.05s / `[PASS]` |



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
| [`figures/mindanao_s2s_dynamic_predictor_composite.png`](../figures/mindanao_s2s_dynamic_predictor_composite.png) | Canvas 1: 4-panel ECMWF S2S multi-scale dynamic predictor dashboard resolving S2S abstractness via discrete sampling nodes ($N=40$), vector GADM coastline overlay, continuous bilinear remapping ($32 \times 48$), 11-member ensemble spread ($\sigma_{\text{ENS}}$), and dynamic moisture shift ($\Delta_{W2-W1}$) | 300 DPI PNG |
| [`figures/mindanao_s2s_pilot_case_and_recursive_inference.png`](../figures/mindanao_s2s_pilot_case_and_recursive_inference.png) | Canvas 2: 4-panel Model A0 case assembly & recursive sensitivity dashboard: antecedent RZSM memory lag -1d over 126 evaluation cells, ERA5 precipitable water ($PWAT$ at $t_0$), observed ground truth target $Y_{W1}$, and downstream perturbation sensitivity response map $\Delta_{W2}$ in sequential `inferno` colormap | 300 DPI PNG |
| [`figures/gate1_validation_atmospheric_spatial_sanity.png`](../figures/gate1_validation_atmospheric_spatial_sanity.png) | Pre-Production Gate 1 atmospheric spatial sanity verification composite (5 variables across Mindanao domain) | 300 DPI PNG |
| [`figures/a0_production_seed42_training_curves.png`](../figures/a0_production_seed42_training_curves.png) | Model A0 Seed 42 four-lead ($W_1 \to W_4$) production training convergence curves and learning rate schedule | 300 DPI PNG (258 KB) |
| [`figures/a0_production_seed123_training_curves.png`](../figures/a0_production_seed123_training_curves.png) | Model A0 Seed 123 four-lead ($W_1 \to W_4$) production training convergence curves and learning rate schedule | 300 DPI PNG (433 KB) |
| [`figures/a0_production_spatial_forecast_evaluation.png`](../figures/a0_production_spatial_forecast_evaluation.png) | 4-panel Model A0 spatial forecast evaluation composite: Ground Truth $Y_{W1}$, Ensemble Mean $\hat{Y}_{W1}$, Error Residuals $(\hat{Y}-Y)$, and 11-member Ensemble Spread ($\sigma_{\text{ENS}}$) | 300 DPI PNG (1.14 MB) |
| [`figures/a0_production_benchmark_parity_comparison.png`](../figures/a0_production_benchmark_parity_comparison.png) | Three-seed Model A0 training replicates performance summary vs. Parent EX29 baseline parity verification with $\pm 1\sigma$ error bars and confidence envelopes across Leads $W_1 \to W_4$ | 300 DPI PNG (753 KB) |

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
| [`processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc`](../processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc) | Full 11-year ($2015\text{--}2025$) production RZSM data cube on Candidate A grid with locked climatology and anomalies | NetCDF4 CF-1.8 ($8.40\text{ MiB}$) |
| [`processed/rzsm/pilot/README.md`](../processed/rzsm/pilot/README.md) | Technical data dictionary documenting pilot RZSM coordinate arrays, variable names, and units | Markdown Document |
| [`processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc`](../processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc) | Harmonized ECMWF S2S reforecast pilot dataset across 11 members, Leads 1 & 2, and EX29 triplet (`t2m, d2m, tcw`) on Candidate A grid | NetCDF4 CF-1.8 ($0.42\text{ MiB}$) |

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

