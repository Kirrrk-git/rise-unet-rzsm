<!-- markdownlint-disable -->
# Audit Dossier 25: Model A0 Production Training, Multi-Seed Performance Summary & Baseline Certification

**Project**: Enhanced RISE-UNet for Subseasonal Root-Zone Soil Moisture (RZSM) Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Branch**: `mindanao-adaptation`  
**Milestone**: Sub-Phase 21K — Step 21K.3 (Full Three-Seed Production Training), Step 21K.4 (Multi-Seed Validation Evaluation), and Step 21K.5 (A0 Baseline Contract Freeze)  
**Hardware Environment**: NVIDIA Tesla T4 GPU (15,360 MB VRAM), Google Colab, CUDA 12.5.1, TensorFlow 2.20.0  
**Cloud Storage Lake**: `gs://rise-unet-rzsm/`  
**Audit Date**: 2026-09-17  
**Scientific Certification Status**: `[PASS / ACCEPTED: CERTIFIED_ON_GPU: 2026-09-17]`  


---

## 1. Executive Summary & Milestone Certification

This audit dossier establishes the formal scientific, programmatic, and operational certification of the **Mindanao Model A0 Baseline Production Training** across all four subseasonal forecast horizons ($W_1 \to W_4$, Days 1–28) and three independent training replicates (Seeds `42`, `123`, `456`).

Following the formal GPU certification of all three pre-production clearance gates (Gate 1 Validation Atmospheric Pipeline, Gate 2 Hardware Profiling & VRAM Feasibility, and Gate 3 Production Training Smoke Preflight), the production training pipeline executed cleanly end-to-end on an authoritative physical NVIDIA Tesla T4 GPU in Google Colab. All 12 model training runs completed, selected minimum-validation checkpoints were verified, and full dual-synchronization parity between the local repository and the Google Cloud Storage bucket (`gs://rise-unet-rzsm/`) was confirmed byte-for-byte.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 21K PRODUCTION STATUS                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Pre-Production Gate 1 (Validation Atmospheric Pipeline) : [PASS / VERIFIED]          │
│  Pre-Production Gate 2 (Hardware Profiling & VRAM Feasibility) : [PASS / CERTIFIED_ON_GPU] │
│  Pre-Production Gate 3 (Production Training Smoke Preflight): [PASS / CERTIFIED_ON_GPU]│
│  Step 21K.1 (Usable S2S Case Calendar 1,154 Cycles)     : [PASS / VERIFIED]           │
│  Step 21K.2 (Dataset Splits & Frozen Normalization Contract): [PASS / VERIFIED]        │
│  Step 21K.3 (Full 3-Seed Model A0 Production Training)  : [PASS / CERTIFIED_ON_GPU]    │
│  Step 21K.4 (Three-Seed Validation Evaluation)          : [PASS / VERIFIED]           │
│  Step 21K.5 (Freeze A0 Baseline Contract & Dossier 25)  : [PASS / VERIFIED]           │
│  Final Generalization Evaluation (Sealed Holdout N=202) : [QUARANTINED FOR PHASE 26]   │
│                                                                                        │
│  OVERALL VERDICT: PASS & CERTIFIED — PHASE 21 SEALED; PHASE 23 UNLOCKED                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pre-Production Clearance Gates Audit Trail

Before production training authorization was granted on 2026-09-17, the pipeline cleared three mandatory, fail-closed pre-production gates:

1. **Pre-Production Gate 1: Validation Atmospheric Pipeline Preflight Certification (2026-09-16)**
   * **Scope**: Verified 2022–2023 atmospheric mirror across 730 continuous days (zero missing or duplicate days), 24 monthly NetCDF files, 210 validation cycles, 5 surface channels (`spfh`, `tmax`, `diff_temp`, `pwat`, `hgt_pres`), and 126 active evaluation cells.
   * **Diagnostics**: Three-way normalization checks showed unclipped bounds $[0.0220, 0.9879]$ with 0 bound excursions. Real production `CaseBuilder` ingestion produced exact multi-lead tensor hierarchies.
   * **Audit Reference**: [`reproduction_audit/14_production_cube_compilation_and_census_audit.md`](14_production_cube_compilation_and_census_audit.md); Telemetry in `logs/gate1_validation_atmospheric_execution.json`.

2. **Pre-Production Gate 2: Genuine Model A0 (`UNET_RZSM`) Hardware Profiling & VRAM Feasibility (2026-09-17)**
   * **Scope**: Certified all six technical pillars on physical NVIDIA Tesla T4 GPU:
     - *Pillar 21J.1*: Genuine `UNET_RZSM` architecture instantiation (1,627,139 $W_1$ / 1,630,307 $W_2$ parameters, 298 weight tensors, 3 deep-supervision heads).
     - *Pillar 21J.2*: Forward pass across Leads $W_1 \to W_4$ with $0.00 \times 10^0$ ocean buffer leakage.
     - *Pillar 21J.3*: Finite gradients across all 298 tensors under representative multi-head loss ($\|\Delta w\| = 3.95 \times 10^{-3}$).
     - *Pillar 21J.4*: 4-lead recursive cascade with active perturbation propagation.
     - *Pillar 21J.5*: VRAM ladder $B \in \{11, 22, 33, 44, 66\}$ from $2.08\text{ GB}$ to peak $10.57\text{ GB}$ ($4.79\text{ GB}$ headroom on T4 with zero OOM faults).
     - *Pillar 21J.6*: Checkpoint serialization parity with $0.00 \times 10^0$ maximum weight discrepancy.
   * **Audit Reference**: [`reproduction_audit/21_vram_and_hardware_profiling_audit.md`](21_vram_and_hardware_profiling_audit.md); Telemetry in `logs/A0_gpu_benchmark.json`.

3. **Pre-Production Gate 3: Genuine Production-Path Model A0 Training Smoke Preflight (2026-09-17)**
   * **Scope**: Executed 5-stage preflight on real pilot case (`CASE_20150115_W01.npz`) normalized via frozen contract:
     - *Stage A*: Genuine 4-lead backward pass with finite gradients and non-zero weight updates ($\|\Delta w\| \approx 0.11 - 0.12$).
     - *Stage B*: Recursive autoregressive cascade ($W_1 \to \hat{y}_1 \to W_2 \to \hat{y}_2 \to W_3 \to \hat{y}_3 \to W_4$) on actual model inferences, verifying exact recursive channel placement and permutation tamper rejection ($\Delta_{\max} = 0.2135$).
     - *Stage C*: Downstream 126-cell land masking proof decoupling active evaluation error from zero-filled ocean (mean unmasked MAE = 0.3201 vs. masked MAE = 0.8847).
     - *Stage D*: Exact checkpoint restoration parity ($0.00 \times 10^0$) and full training-state step-2 trajectory roundtrip (loss delta = $0.00 \times 10^0$, weight delta = $6.70 \times 10^{-7} < 5 \times 10^{-6}$).
     - *Stage E*: Automated engine execution (`scripts/14_run_a0_production_smoke_test.py --mode certify`) with exit code 0.
   * **Audit Reference**: [`reproduction_audit/23_production_smoke_preflight_audit.md`](23_production_smoke_preflight_audit.md); Telemetry in `logs/a0_production_smoke_test.json`.

---

## 3. Authoritative Cohort Accounting & Conservation Proof

To ensure complete experimental transparency, the effective training and validation cohort is rigorously distinguished from the nominal scheduled calendar:

* **Nominal Scheduled Calendar**:
  - Training Partition (2015–2021): 7 years $\times$ 105 bi-weekly cycles = **735 scheduled cases**
  - Validation Partition (2022–2023): 2 years $\times$ 105 bi-weekly cycles = **210 scheduled cases**
  - Grand Total Scheduled (Train + Val): **945 scheduled cycles**
* **ECMWF Provider MARS Unavailable Exceptions**:
  - Exhaustive querying of ECMWF MARS archive via CDS API returned `MarsNoDataError` on 74 specific candidate slots:
    - Training (2015–2021): 58 cycles (56 late-year bi-weekly skips where ECMWF did not issue reforecasts + 2 leap days `2016-02-29` and `2020-02-29`).
    - Validation (2022–2023): 16 cycles (late-year bi-weekly skips).
  - These cycles do not physically exist in ECMWF's operational archives and represent true provider-level boundaries.
* **Effective Empirically Usable Cohort**:
  - Effective Training Cases: $735 - 58 = \mathbf{677\text{ usable cases}}$ ($92.1\%$ of nominal)
  - Effective Validation Cases: $210 - 16 = \mathbf{194\text{ usable cases}}$ ($92.4\%$ of nominal)
  - Grand Total Effective Usable (Train + Val): $\mathbf{871\text{ cases}}$
* **Exact Conservation Identity**:
  $$945\text{ scheduled} \equiv 871\text{ usable cases} + 74\text{ provider exceptions} + 0\text{ unexplained gaps} \quad (\Delta = 0)$$
* **Ledger Record**: Documented in [`manifests/splits/cases_availability_audit.csv`](../manifests/splits/cases_availability_audit.csv) and certified in [`reproduction_audit/24_s2s_provider_availability_and_cohort_census_audit.md`](24_s2s_provider_availability_and_cohort_census_audit.md).

---

## 4. Production Training Execution & Checkpoints Manifest

* **Training Engine**: [`scripts/16_train_a0_production.py`](../scripts/16_train_a0_production.py)
* **Orchestration Notebook**: [`notebooks/12_mindanao_a0_production_training.ipynb`](../notebooks/12_mindanao_a0_production_training.ipynb)
* **Execution Parameters & Formal Batch Size Amendment**:
  - Optimizer: Adam ($\eta = 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$)
  - **Batch Size Reconciliation Chain**:
    - *Nominal Pre-Production Freeze*: $B = 11$ ($1\text{ case} \times 11\text{ members}$, initial single-case minimum configuration)
    - *Production Execution Amendment*: $B = 33$ ($3\text{ cases} \times 11\text{ members}$)
    - *Amendment Reason*: Improved production throughput and provided a larger three-case mini-batch for more stable batch statistics while remaining within the certified T4 memory envelope (peak 5.6 GB VRAM vs. 15.4 GB capacity).

    - *Amendment Date & Commits*: 2026-09-17; engine commit `62500f8` and execution commit `e2b35d9`.
    - *Effect on Scientific Interpretation*: Documented as an authoritative production-training hyperparameter. Ingesting 3 cases (33 members) provides sample diversity for BatchNorm statistics across 3 distinct meteorological events per gradient step while strictly preserving 11-member ensemble realization ordering and loss broadcasting invariance.
  - Max Epochs: 40
  - Learning Rate Schedule: `ReduceLROnPlateau` (factor = 0.5, patience = 3, min_lr = $10^{-6}$)
  - Early Stopping: patience = 8 epochs, restoring best minimum-validation CRPS weights
  - Loss Objective: Spatial CRPS Proxy $\mathcal{L} = \text{MAE}_{\text{eval}} - 0.08 \cdot \bar{\sigma}_{\text{spatial}}$

* **Checkpoints Verification (12 Models)**:
  All 12 model training runs completed cleanly and generated verified `best_model.weights.h5` artifacts across Seeds `42`, `123`, and `456` and Leads $W_1 \to W_4$. Checkpoints are stored in GCS (`gs://rise-unet-rzsm/checkpoints/A0/`) and mirrored locally:

| Seed | Lead | Checkpoint Path | Status | Best Val Metric |
| :---: | :---: | :--- | :---: | :---: |
| **42** | $W_1$ | `checkpoints/A0/seed_42/lead_1/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **42** | $W_2$ | `checkpoints/A0/seed_42/lead_2/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **42** | $W_3$ | `checkpoints/A0/seed_42/lead_3/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **42** | $W_4$ | `checkpoints/A0/seed_42/lead_4/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **123** | $W_1$ | `checkpoints/A0/seed_123/lead_1/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **123** | $W_2$ | `checkpoints/A0/seed_123/lead_2/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **123** | $W_3$ | `checkpoints/A0/seed_123/lead_3/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **123** | $W_4$ | `checkpoints/A0/seed_123/lead_4/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **456** | $W_1$ | `checkpoints/A0/seed_456/lead_1/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **456** | $W_2$ | `checkpoints/A0/seed_456/lead_2/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **456** | $W_3$ | `checkpoints/A0/seed_456/lead_3/best_model.weights.h5` | `[PASS]` | Verified in GCS |
| **456** | $W_4$ | `checkpoints/A0/seed_456/lead_4/best_model.weights.h5` | `[PASS]` | Verified in GCS |

---

## 5. Three-Seed Performance Summary & Parent Reference Comparison

### 5.1 Replicate Variability vs. Meteorological Ensemble Distinction
To preserve strict methodological rigor in thesis reporting:
1. **Three Training Seeds (`42`, `123`, `456`)**: Represent **independent training replicates** measuring optimization stability and weight initialization sensitivity. Reporting their mean $\pm 1\sigma$ standard deviation constitutes a **multi-seed variability / robustness analysis**, not an operational forecast ensemble.
2. **$M=11$ Realizations**: The exact analytical CRPS (Hersbach, 2000) is evaluated across the **11 ECMWF S2S dynamic meteorological forecast realizations** per case, which is a genuine physical weather ensemble.

### 5.2 Three-Seed Performance Summary Table (194 Validation Cases)

| Lead Horizon | Forecast Range | Spatial CRPS Proxy | Exact Ensemble CRPS (Hersbach 2000) | Validation MAE | Validation RMSE | Validation ACC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$W_1$** | Days 1–7 | $0.0339 \pm 0.0007$ | $\mathbf{0.0330 \pm 0.0008}$ | $0.0341 \pm 0.0007$ | $\mathbf{0.0505 \pm 0.0009}$ | $\mathbf{0.8544 \pm 0.0047}$ |
| **$W_2$** | Days 8–14 | $0.0525 \pm 0.0016$ | $\mathbf{0.0506 \pm 0.0020}$ | $0.0528 \pm 0.0016$ | $\mathbf{0.0714 \pm 0.0015}$ | $\mathbf{0.6824 \pm 0.0096}$ |
| **$W_3$** | Days 15–21 | $0.0592 \pm 0.0008$ | $\mathbf{0.0582 \pm 0.0011}$ | $0.0593 \pm 0.0007$ | $\mathbf{0.0800 \pm 0.0006}$ | $\mathbf{0.5629 \pm 0.0067}$ |
| **$W_4$** | Days 22–28 | $0.0625 \pm 0.0002$ | $\mathbf{0.0613 \pm 0.0003}$ | $0.0627 \pm 0.0002$ | $\mathbf{0.0844 \pm 0.0005}$ | $\mathbf{0.4948 \pm 0.0095}$ |

### 5.3 Parent EX29 Contextual Reference Comparison — CONTEXTUAL / REPORTED REFERENCE

> **Certification Taxonomy Distinction**: Under the repository governance standard, `[VERIFIED]` is strictly reserved for direct mathematical and programmatic parity with the parent EX29 source code and execution architecture. Empirical performance comparisons against reported CONUS literature numbers are classified as **Contextual / Reported References**, reflecting distinct regional, geographic, and climatological regimes.

Comparing Mindanao Model A0 against Kyle Lesinger's reported EX29 baseline (*Lesinger & Tian 2025, Nature Communications*):


| Lead Horizon | Mindanao Model A0 ACC | Reported Parent EX29 ACC | Mindanao Model A0 RMSE | Reported Parent EX29 RMSE | Reference Context Analysis |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$W_1$ (Days 1–7)** | $\mathbf{0.8544 \pm 0.0047}$ | $\sim 0.82$ | $\mathbf{0.0505 \pm 0.0009}$ | $\sim 0.050$ | Higher ACC; slightly higher RMSE |
| **$W_2$ (Days 8–14)** | $\mathbf{0.6824 \pm 0.0096}$ | $\sim 0.65$ | $\mathbf{0.0714 \pm 0.0015}$ | $\sim 0.070$ | Higher ACC; slightly higher RMSE |
| **$W_3$ (Days 15–21)** | $\mathbf{0.5629 \pm 0.0067}$ | $\sim 0.55$ | $\mathbf{0.0800 \pm 0.0006}$ | $\sim 0.080$ | Higher ACC; approximately equal RMSE |
| **$W_4$ (Days 22–28)** | $\mathbf{0.4948 \pm 0.0095}$ | $\sim 0.47$ | $\mathbf{0.0844 \pm 0.0005}$ | $\sim 0.090$ | Higher ACC; lower RMSE |

#### Formal Scientific Characterization:
> **"Model A0 achieved performance broadly comparable to the reported EX29 benchmark, with higher ACC across all four horizons and mixed RMSE differences."**
> 
> *Contextual Qualification*: The parent study evaluated contiguous United States (CONUS) temperate regimes using GLEAM v3.8a RZSM, whereas Mindanao represents a tropical maritime monsoon setting utilizing ERA5-Land RZSM. While exact numerical equivalence is not expected across distinct climates, the results demonstrate that the adapted RISE-UNet architecture successfully captures subseasonal soil moisture memory and dynamic atmospheric forcing over Mindanao, validating the regional adaptation framework.

---

## 6. Strict Quarantine of the Sealed Test Cohort

* **Quarantine Policy**: The 2024–2025 holdout partition ($N=209$ scheduled cases, **$202$ usable evaluation denominator cases**, 7 boundary OOB cases) remains **strictly sealed and untouched**.
* **Zero Contamination Assertion**:
  - No model training updates, learning rate adjustments, early-stopping decisions, or hyperparameter tuning touched the 2024–2025 cohort.
  - Normalization bounds (`contracts/A0/normalization_parameters.yaml`) remain strictly locked to 2015–2021 training cases.
  - Final thesis generalization assertions and paired moving-block bootstrap hypothesis tests are formally reserved for **Phase 26** after post-A0 diagnostics and potential refinement gates are concluded.

---

## 7. Dual Artifact & Cloud Lake Synchronization Audit

All visual and numerical artifacts generated during the production training phase were verified byte-for-byte across the local repository and Google Cloud Storage:

```text
Local Codebase                                                        GCS Cloud Lake (gs://rise-unet-rzsm/)
├── figures/                                                          ├── figures/
│   ├── a0_production_benchmark_parity_comparison.png (752,596 B)     │   └── a0_production_benchmark_parity_comparison.png (752,596 B) [MATCH]
│   ├── a0_production_spatial_forecast_evaluation.png (1,137,382 B)   │   └── a0_production_spatial_forecast_evaluation.png (1,137,382 B) [MATCH]
│   ├── a0_production_seed123_training_curves.png     (432,696 B)     │   └── a0_production_seed123_training_curves.png     (432,696 B) [MATCH]
│   └── a0_production_seed42_training_curves.png      (258,240 B)     │   └── a0_production_seed42_training_curves.png      (258,240 B) [MATCH]
├── logs/                                                             ├── logs/
│   └── a0_production_3seed_summary.json              (6,266 B)       │   └── a0_production_3seed_summary.json              (6,266 B)   [MATCH]
└── checkpoints/A0/                                                   └── checkpoints/A0/
    └── [12 model weights across Seeds 42, 123, 456; Leads 1-4]        └── [12 model weights across Seeds 42, 123, 456; Leads 1-4] [MATCH]
```

---

## 8. Clearance for Phase 23 (Recursive Degradation Diagnostic)

With Step 21K.3, 21K.4, and 21K.5 formally executed, verified, and sealed:
1. **Phase 21 is officially certified and closed**.
2. **Model A0 is frozen as the authoritative regional baseline**.
3. **Phase 23 (Recursive Degradation Diagnostic) is formally UNLOCKED**.

Phase 23 will investigate error accumulation across the recursive chain ($W_1 \to W_2 \to W_3 \to W_4$) on the training/validation material, directly preparing for the **Post-A0 Gate 2 (G2-R): Recursive Refinement Decision Gate** (formally disambiguated from Pre-Production Gate 2 Hardware Profiling).
