<!-- markdownlint-disable -->
# Mindanao RISE-UNet Adaptation Master Plan (v3.0 — Revised Thesis SCPW–MRGR Roadmap)

```text
Document Authority : Master Plan (Governance Tier 2: Contracts -> Master Plan -> Artifact Registry)
Last Updated       : 2026-11-19
Plan Version       : v3.0.0-thesis-revised (Harmonized to Updated Thesis 1 Proposal & Audit)
Primary Thesis Ref : Improving DL-DM RISE-UNet: Seasonal Conditioning and Reliability-Gated
                     Recursive Refinement for Probabilistic Subseasonal Root-Zone
                     Soil-Moisture Forecasting (Outline Defense Revision, Oct 12 2026)
Parent Baseline    : Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3
Active Branch      : mindanao-adaptation
Contracts Locked   : contracts/A0/thesis_revised_baseline_and_data_contract.yaml (v3.0.0)
                     contracts/A1/mrgr_model_architecture_contract.yaml (v3.0.0)
                     contracts/spatial/spatial_grid_contract.yaml (FROZEN)
Primary Storage    : Consolidated Kaggle Lake (`kirkishere/rise-unet-rzsm`, Private, 3,828 files / 6.45 GB)
                     + Private Code/Docs (`kirkishere/rise-unet-rzsm-docs`)
                     + Legacy Migration Backup (`aaronjalapon02/mindanao-gcs-migration-index` v2)
                     [GCS gs://rise-unet-rzsm/ read/write BLOCKED — HTTP 403 Billing Disabled]
```

---

## 0. Current Scientific & Operational Status Disclosure

> [!IMPORTANT]
> **Thesis 1 Stage & Legacy Baseline Posture**:
> 1. Per the updated 24-page thesis documentation and the **12 October 2026 outline-defense minutes (Appendix D)**, this project is at the **Thesis 1 (Outline / Proposal) stage**. Zero final empirical model results or accuracy improvements are claimed yet.
> 2. Previously executed **Legacy Phase 21–23** runs (`2026-09-17`) remain preserved in contracts/logs as historical engineering & preliminary diagnostic records, **but are methodologically superseded** for all thesis-reported baseline comparisons because an exhaustive audit (`reproduction_audit/26_updated_thesis_alignment_and_comprehensive_audit.md`) verified five critical discrepancies against the updated thesis protocol and parent `EX29` study:
>    - `src/models/a0_unet.py` imported `modelRzsmRelu.py` (`1.63M` params, DepthwiseConv2D, `[32,32,64,128]`) instead of authentic parent `modelRzsmReluExtraConv.py` (`10.24M` params, standard `Conv2D`, `[32,64,128,256]`).
>    - `scripts/08_derive_training_normalization.py` hardcoded `s2s_tcw = [0.5, 15.0]`, clipping **100%** of active Mindanao ECDS `tcw` (`27.66–53.77 kg/m²`) to constant `1.0000` (`ch10 std = 0.0`).
>    - ERA5 atmospheric and ECMWF S2S predictors used un-anomalized single-day/raw values with hardcoded bounds instead of 7-day trailing summaries (`lag=-1d` for ERA5 obs), DJF/MAM/JJA/SON seasonal anomalies, and training-only empirical min-max normalization.
>    - `26` derived monthly atmospheric files (`2015_02`, `2015_03`, `2024_01..2025_12`) and late-Dec 2014 ERA5 atmospheric warmup files are absent from GCS/local caches.
>    - Legacy year splits (`677/194/202`, unpurged), `B=33`, `D=1` (`M=11`), `L=4` MBB, and `<= tau` dry-event inequalities differ from the updated thesis protocol (`851/100/108` purged 80/10/10 split, `B=66`, `D=10` / `M=110`, `28d/56d` MBB, strict `< tau`).
> 3. **Storage Infrastructure**: Google Cloud Storage billing on project `mindanao-drought-aaron-jalapon` is disabled (`HTTP 403` on object reads/writes). All `7,789` GCS objects (`45,788,363,491` bytes across 4 buckets, including all `6,795` objects / `31,092,944,230` B in `gs://rise-unet-rzsm/`) are preserved in `aaronjalapon02`'s Kaggle backup, and **all `6,795` `rise-unet-rzsm` objects were read back and verified byte-for-byte (`COMPLETE_ALL_6795_VERIFIED`, `logs/kaggle_migration_verification_report.json`) inside `kirkishere/rise-unet-rzsm-verify-and-consolidate`**, consolidating the `3,826` active files (`6,448,640,613` B, including all 12 canonical legacy `A0` weights) into `7` topic `ZIP_STORED` bundles alongside Private `kirkishere/rise-unet-rzsm-docs`.

---

## 1. Authoritative Architecture & Methodology Summary (Updated Thesis v3.0)

### 1.1 Study Domain & Case Inventory (`[FROZEN]` & `[VERIFIED]`)
- **Grid**: Candidate A regular $0.25^\circ$ mesh over Mindanao ($11.75^\circ\text{N} \to 4.00^\circ\text{N}$, $116.00^\circ\text{E} \to 127.75^\circ\text{E}$, grid shape $H=32, W=48 = 1,536$ cells).
- **Evaluation Mask**: `126` active majority-coverage land cells ($f \ge 0.50$, covering $86,418.83\text{ km}^2$ or $86.46\%$ of Mindanao's $99,948.76\text{ km}^2$ PSA boundary); `1,410` ocean/buffer cells zero-filled.
- **Usable Case Census**: `1,154` CY48R1 operational issue cycles (`2015–2025`) $\to$ `7` `TARGET_OUT_OF_BOUNDS` (`Dec 2025`) + `74` `MarsNoDataError` provider gaps $\to$ **`1,073` usable issues** (`11,803` physical member-samples per lead before boundary purging).

### 1.2 Chronological 80/10/10 Partitioning with Target-Window Boundary Purging (Sec 2.3)
- **Raw Chronological Cutoffs**: First `floor(0.80 * 1073) = 858` issues Train / next `107` Val (`858:965`) / last `108` Sealed Test (`965:1073`).
- **Boundary-Purging Rule**: Exclude any Train/Val issue whose 28-day target window `[t0, t0+27d]` overlaps the earliest issue date of the following partition (`w4_end >= next_split.min_issue_date`):
  - **Training (Retained)**: **`851` issues** (`9,361` member-samples/lead), `2015-01-01` .. `2023-09-23` (max W4 end `2023-10-20`; `7` trailing issues `2023-09-26..2023-10-17` purged).
  - **Validation (Retained)**: **`100` issues** (`1,100` member-samples/lead), `2023-10-21` .. `2024-10-28` (max W4 end `2024-11-24`; `7` trailing issues `2024-10-31..2024-11-21` purged).
  - **Sealed Final Test**: **`108` issues** (`1,188` member-samples/lead), `2024-11-25` .. `2025-12-04` (max W4 end `2025-12-31`).

### 1.3 Predictor & Target Pipeline (Sec 2.1–2.2)
- **Targets & Antecedent RZSM**: ERA5-Land $0.10^\circ$ layers (`0.07*SM1 + 0.21*SM2 + 0.72*SM3`), land-aware Delaunay/coastal-NN remapped to $0.25^\circ$, `7`-day trailing rolling mean (`center=False`), training-only seasonal climatology (`DJF/MAM/JJA/SON`) anomaly subtraction, and training-only active-cell min-max normalization. Lags `[-1, -7, -14]` days; Targets `[+6, +13, +20, +27]` days.
- **ERA5 Atmospheric Predictors (`pwat, spfh, tmax, diff_temp, z200`)**: Daily values derived on $0.25^\circ$, aggregated into **7-day trailing historical summaries** up to `lag = -1d` (`t0 - 1d`, zero `t0` lookahead), transformed into training-fitted seasonal anomalies and training-only min-max scaled. *(Requires 2014-12 warmup or documented exclusion check, plus deriving 26 missing months).*
- **ECMWF S2S Predictors (`t2m, d2m, tcw`, 11 members)**: Steps `0–144h` (W1) and `168–312h` (W2) bilinearly remapped from $1.5^\circ \to 0.25^\circ$, converted to seasonal anomalies and empirical training-only min-max scaled (**fixing `ch10` `tcw` saturation**).
- **Stage Channel Tensor Topology (`[M=11, H=32, W=48, C_h]`)**:
  - **W1 (`11 ch`)**: `[3 RZSM lags, 5 ERA5 obs, 3 W1 S2S]` (nonrecursive `x_1 = 11 ch`).
  - **W2 (`12 ch`)**: `[3 RZSM lags, 5 ERA5 obs, 3 W2 S2S, r*_{1->2}]` (nonrecursive `x_2 = 11 ch` + `1` handoff).
  - **W3 (`5 ch`)**: `[3 RZSM lags, r*_{1->3}, r*_{2->3}]` (nonrecursive `x_3 = 3 ch` + `2` handoffs).
  - **W4 (`6 ch`)**: `[3 RZSM lags, r*_{1->4}, r*_{2->4}, r*_{3->4}]` (nonrecursive `x_4 = 3 ch` + `3` handoffs).

### 1.4 Authentic Stage Backbone & Joint SCPW–MRGR Enhancement (Sec 2.4)
- **Authentic Backbone (`modelRzsmReluExtraConv.py`)**: `[32, 64, 128, 256]` schedule, standard `Conv2D` throughout inception + conv composites, `SqueezeAndExcite2D(ratio=0.25)` in both inception & conv blocks, dropout `0.10` initial / `0.25` later (`SpatialDropout2D` + `Dropout`, active via `training=True`), 3 deep-supervision ReLU heads (`loss_weights = [0.2, 0.3, 0.5]`, `CRPS_exp` masked surrogate with `std` denom `11`). Verified parameter counts at $32 \times 48$: **`10,240,723` (W1), `10,243,539` (W2), `10,223,827` (W3), `10,226,643` (W4)**.
- **SCPW Module (`+15,024` ch11 / `+5,808` ch3 per stage)**:
  - Midpoint date `d_h(t) = t + 7(h-1) + 3`, descriptor `s_h(t) = [sin(2*pi*(n-1)/Y), cos(2*pi*(n-1)/Y)]`.
  - Inserted strictly between initial `128`-ch inception output `u_h` and following `64`-ch conv composite:
    `z_h = ReLU(Conv3x3_128(x_h))`, `w_h = 1 + 0.25 * tanh(MLP_16_128(s_h))` (MLP output zero-initialized), `u_h^S = u_h + (w_h - 1) * z_h`.
- **MRGR Module (`6` directed handoffs, `+5,250` W1->W2 & `+4,098` each W3/W4 = `25,740` params total)**:
  - Each handoff `(j -> h)` receives original stored source state `r_j` (`m^3/m^3`) and raw normalized consumer context `x_h`.
  - Parallel `RiskNet` (`3x3_8` on `x_h` & `r_j` -> concat `24ch` with `|e_x - e_r|` -> `3x3_8` -> `1x1` Softplus `q_hat >= 0`, no BN/dropout) trained first via active-cell Huber loss (`delta=1.0`) against `q_{j->h} = |r_{j,OOF} - y_j| / (c_{j->h} + 1e-6)` where `c_{j->h} = max(P95(|r_{j,OOF}-y_j|), 1e-6 m^3/m^3)`.
  - `RiskNet` is **frozen**, and `G_{j->h} = exp(-q_hat)` gates `CorrectionNet` (`3x3_8` stem -> zero-init `1x1` linear `d_theta`, `delta = c_{j->h} * tanh(d_theta)`):
    `r_{j->h}^* = r_j + (1 - G_{j->h}) * delta_{j->h}` (**no clipping**; out-of-`[0,1]` excursions logged).
- **Consumer Joint Optimization & Caching (Sec 2.4.7)**:
  - Sequential training `W1 -> W2 -> W3 -> W4`. At consumer `h`, preceding stages & `RiskNet` remain frozen (BN in inference mode), while **consumer backbone `F_h` + active SCPW + incoming CorrectionNets optimize jointly** (`Adam lr=1e-4`, `batch=66`, `max 40 epochs`, `RLROP(0.5, 3, 1e-6)`, `ES(8)`, seeds `[42, 123, 456]`, matched initial weights, cached `D=10` stochastic prefix trajectories).
  - Baseline `W1` is reused identically by `MRGR-only`; `SCPW-only` `W1` is reused identically by `Combined`.

### 1.5 Evaluation, Ablations & Statistical Comparison Suite (Sec 2.5–2.6, Table 6 & 7)
- **Stochastic Inference**: `11` ECMWF members $\times$ `D = 10` stochastic MC dropout trajectories = **`110` predictions** per issue/seed/lead (BN stats fixed, backbone dropout active).
- **Metrics**: Per-case predictive-mean (over 110 draws) `ACC`, `MAE`, `RMSE`; empirical `CRPS` Eq. (15) over `110` draws; population spread & 80%/95% PI coverage; dry-event `tau(s, k)` = training 20th percentile by cell `s` & midpoint season `k` with **strict `< tau`** (`p_hat = N_dry / 110`), binary Brier Score & 10-bin reliability diagram.
- **Model Lineup (8 Configurations)**:
  1. `Persistence` (external benchmark: carry `t-1` weekly RZSM)
  2. `Adapted RISE-UNet Baseline` (`modelRzsmReluExtraConv`, SCPW=off, MRGR=off)
  3. `RISE-UNet with MRGR` (SCPW=off, MRGR=on)
  4. `RISE-UNet with SCPW` (SCPW=on, MRGR=off) + **`Constant-Calendar Control ([0,0])`**
  5. `Season-Aware MRGR–RISE-UNet (Combined)` (SCPW=on, MRGR=on)
  6. `SCPW + Ungated Residual Correction Ablation` (`r* = r + delta`)
  7. `U-Net Regression Benchmark` (external)
  8. `UNet++ Regression Benchmark` (external)
- **Primary Inferential Family**: Seed-averaged paired case-level `CRPS` difference (`Combined - Baseline`) at `W3` and `W4`, using **28-calendar-day Paired Moving-Block Bootstrap** (`10,000` replicates, actual dates/gaps retained, mean-centered null `p`, percentile 95% CI, Holm-Bonferroni over W3 & W4) + `56-day` sensitivity check.

---

## 2. Historical Progress Ledger (Pre-Revision Phases 0–23)

| Phase Range | Summary of Work Completed | Historical Status | Post-Audit Disposition |
| :--- | :--- | :---: | :--- |
| **Phase 0 – Gate 1 (1A–1B)** | Archived parent `EX29` repo (`4af8e8c8`), inspected `02_run_model...ipynb`, drafted `freeze/` contracts. | `[PASS / VERIFIED]` (legacy) | Preserved; `freeze/parent_architecture_contract.yaml` & `parent_training_contract.yaml` flagged with `modelRzsmRelu` vs `ExtraConv` & `B=66` reconciliation note. |
| **Sub-Phase 21A** | Spatial foundation: PSA boundary (`99,948.76 km^2`), Candidate A `32x48`, `126`-cell mask (`86.46%`). | `[PASS / ACCEPTED]` | **Retained unchanged** — matches Updated Thesis Sec 2.1.5 identically. |
| **Sub-Phase 21B–21F** | Acquired `265` ERA5-Land NCs, `132+132` raw ERA5 atmos NCs, `1,084` valid S2S cycles; built 11-yr RZSM cube & pure-Python GRIB2 decoder. | `[PASS]` | Raw archives retained (need 7d Dec 2014 ERA5 atmos + 26 derived atmos monthly files); RZSM climatology/bounds require refit on purged 80% train. |
| **Sub-Phase 21G–21K & Ph 22** | Built `1,154`-cycle calendar, `677/194/202` split, `src/models/a0_unet.py` (`modelRzsmRelu` 1.63M), trained 3 seeds (`B=33`, `D=1`). | `[SUPERSEDED]` | **Superseded** by Thesis v3.0 due to wrong backbone (`modelRzsmRelu`), `tcw` `ch10` saturation, un-anomalized predictors, and old split. |
| **Phase 23 (Gate 2 Diagnosis)** | Demonstrated recursive gap (`~20%` masked / `35–44%` unmasked) & Amihan/Habagat spatial dipole; produced Thesis 1 problem motivation. | `[PRELIMINARY ONLY]` | Motivational evidence only; must be re-run in Phase 23R under `ExtraConv`, `D=10`, purged split, and `28d` MBB. |
| **Phase 24 Draft (`v2.0.0`)** | Initial MRGR-only contract (frozen A0, DOY concat, clipped `r*`). | `[SUPERSEDED]` | Replaced cleanly by `contracts/A1/mrgr_model_architecture_contract.yaml` (**v3.0.0**). |

---

## 3. Master Execution Roadmap Going Forward (Phases 21R–26R)

```text
Phase 21R (Data, Storage, Split & Backbone Remediation)
    ├── 21R.1: Kaggle Storage Consolidation & Loader [6,795/6,795 VERIFIED; CODE & DOCS SYNCED] (cloud_lake.py + logs/kaggle_migration_verification_report.json)
    ├── 21R.2: Missing Raw/Processed Atmospheric Acquisition [READY ON KAGGLE; DERIVATION PENDING] (Dec 25-31 2014 warmup + 26 months)
    ├── 21R.3: Purged 80/10/10 Split Manifests & Train RZSM Contract [DONE OFFLINE] (manifests/splits_v3/, rzsm_normalization_v3_purged.yaml)
    ├── 21R.4: Pipeline Remediation [CODE READY] (src/data/preprocessing_v3.py: 7d lag-1d + seasonal anomalies + dynamic ch10 tcw)
    ├── 21R.5: ExtraConv Backbone Hook & SCPW-MRGR Code [CODE READY; GPU VRAM PROFILE PENDING] (src/models/scpw_mrgr.py)
    └── 21R.6: Unit Tests Update & Verification [CPU TESTS PASSING; TF-GPU DEFERRED] (tests/test_18_thesis_v3_modules.py)
         ↓
Phase 23R (Re-Baselined Adapted RISE-UNet Baseline & Sensitivity Reference)
    ├── Train Baseline RISE-UNet (Seeds 42, 123, 456; B=66; 40 epochs; [0.2,0.3,0.5]; shared init weights saved)
    ├── Execute D=10 MC Dropout Inference (110 draws/case; fixed BN stats; active Dropout)
    └── Verify Valid Baseline Metrics (ACC/MAE/RMSE mean-110, CRPS Eq. 15, dry-event BS < tau, m-sensitivity)
         ↓
Phase 24R (SCPW–MRGR Implementation, 9-Year Nested OOF Banks & Matched Ablations)
    ├── 24R.1: SCPW Branch (Conv3x3_128 + MLP_16_128 zero-init) + SCPW-only & Constant-Calendar [0,0] Control
    ├── 24R.2: 9-Year Interval-Purged Nested OOF Prefix Banks (unordered subset dedup: 675 -> 183 fits/seed/config)
    ├── 24R.3: RiskNet_{j->h} Training & Freezing (6 pairs, Huber delta=1.0, c_{j->h} = max(P95, 1e-6))
    ├── 24R.4: Joint Unfrozen Consumer + CorrectionNet Training (MRGR-only [reuses Baseline W1] & Combined [reuses SCPW W1])
    └── 24R.5: SCPW + Ungated Residual Correction Ablation (r* = r + delta)
         ↓
Phase 25R (External Benchmarks & Locked Validation Gate)
    ├── Persistence Benchmark (t-1 weekly RZSM carried to W1-W4)
    ├── Independent U-Net [24] & UNet++ [25] Regression Benchmarks (matched 851/100 split & hyperparameters)
    └── Validation Selection Gate: Lock checkpoints, RiskNet decile bin edges & configs BEFORE unsealing Test
         ↓
Phase 26R (Sealed Final Test Evaluation, Statistical Inference & Thesis 2 Manuscript Integration)
    ├── Single-Access Unsealing of N=108 Test Cases (2024-11-25 .. 2025-12-04, 1,188 samples/lead)
    ├── Primary Inference: 28-day & 56-day Paired MBB (10,000 resamples) + Holm-Bonferroni on W3/W4 CRPS
    ├── Secondary Diagnostics: MRGR-SCPW interaction contrast, DJF/MAM/JJA/SON seasonal stratification,
    │   RiskNet Spearman/deciles, out-of-[0,1] frequency, m in {0,0.5,1,1.5,2} perturbation slopes
    └── Publication Packaging & Thesis 2 Results/Discussion Integration
```

---

## 4. Detailed Phase Specifications & Remediation Gates

### Phase 21R — Engineering Remediation & Infrastructure Alignment (`[GATE BEFORE TRAINING]`)

1. **Step 21R.1 — Kaggle-First Storage Adapter & 6,795-File Consolidation (`src/data/cloud_lake.py`, `scripts/21_sync_kaggle_docs_dataset.py`)** `[DONE & LIVE ON KAGGLE]`:
   - Decoupled resolver from GCS-only assumptions (`gs://rise-unet-rzsm/` `HTTP 403`). Resolution priority: (1) local repo / `data_cache/` $\to$ (2) consolidated `kirkishere/rise-unet-rzsm` (auto-extracted `<bundle_stem>/` tree or 7 topic `ZIP_STORED` bundles) $\to$ (3) legacy 4-dataset Backup Store (`object-locations.json` loose + `.zip.bin` MD5/CRC32C) $\to$ (4) optional GCS fallback.
   - Cloud verification kernel `kirkishere/rise-unet-rzsm-verify-and-consolidate` (`2,443.03 s`) verified **`6,795 / 6,795` objects (`31,092,944,230` B)** across Aaron's 4 Kaggle datasets (`COMPLETE_ALL_6795_VERIFIED`, Archive A/B SHA-256 + per-file MD5/CRC32C, `logs/kaggle_migration_verification_report.json`) and consolidated the `3,826` active files (`6,448,640,613` B) into `7` topic bundles (`14.49 GB` free in `/kaggle/working/`), leaving `10.02 GB` `reference/` and `14.63 GB` duplicated/intermediate legacy checkpoints parked while retaining all `12` canonical `best_model.weights.h5` + `training_history.json` (`86.07 MB`).
   - Consolidated Private dataset **`kirkishere/rise-unet-rzsm`** is now **`ready` (`3,828` files / `6,449,251,345` B**, auto-extracted across the 7 topic folders + manifests/report), paired with Private code/docs dataset **`kirkishere/rise-unet-rzsm-docs`** via `scripts/21_sync_kaggle_docs_dataset.py`.
2. **Step 21R.2 — Complete Atmospheric Archive & Warmup** `[PENDING KAGGLE DERIVATION]`:
   - Acquire/handle `2014-12-25` to `2014-12-31` hourly ERA5 single & pressure levels so `2015-01-01` and `2015-01-04` possess full 7-day trailing windows at `lag=-1d` (`2014-12-25..31`).
   - Run `scripts/03_derive_era5_atmospheric_daily.py` on the `26` missing monthly NetCDFs (`2015_02`, `2015_03`, and `2024_01..2025_12`) inside Kaggle using the verified mounted raw `era5` bundles.
3. **Step 21R.3 — Repartition & Purge `80/10/10` Split (`851 / 100 / 108`) & Train-Only RZSM Contract** `[EXECUTED NON-DESTRUCTIVELY]`:
   - Preserved legacy `manifests/splits/` (`split_summary.json`) for backward compatibility while emitting v3.0 purged manifests (`manifests/splits_v3/`: `851` Train / `100` Val / `108` Sealed Test) via `scripts/20_generate_v3_purged_splits_and_rzsm_bounds.py`.
   - Fitted `contracts/A0/rzsm_normalization_v3_purged.yaml` and `processed/rzsm/v3_purged_train_climatology_and_thresholds.nc` strictly on Training `<= 2023-10-20` (`min=-0.185106, max=+0.157441, std=0.039847`; zero Val/Test access).
4. **Step 21R.4 — Remediate Preprocessing, `ch10` Saturation & Seasonal Anomalies** `[IMPLEMENTED & UNIT-TESTED IN src/data/preprocessing_v3.py]`:
   - Implemented 7-day trailing rolling summaries on ERA5 obs sampled strictly at `lag=-1d`, training-fitted `DJF/MAM/JJA/SON` seasonal anomaly subtraction across both ERA5 and ECMWF S2S (`t2m, d2m, tcw`), empirical training-only scaling with `eps_floor=1e-6`, and zero hardcoded `tcw` clipping. Full production tensor assembly awaits Kaggle raw data mount.
5. **Step 21R.5 — ExtraConv Backbone Hook, SCPW–MRGR Implementation & Evaluation Suite** `[CODE IMPLEMENTED; GPU VRAM PROFILING PENDING]`:
   - Implemented `src/models/scpw_mrgr.py` (`build_extraconv_unet_v3` using authentic `modelRzsmReluExtraConv.py` blocks with `128`-ch SCPW injection + `RiskNet` + `CorrectionNet` builders) and `src/evaluation/v3_metrics_and_bootstrap.py` (`110`-draw CRPS Eq. 15, strict `< tau` Brier/reliability, `28d/56d` calendar MBB + Holm-Bonferroni, RiskNet deciles).
   - Pending GPU runtime: verify Colab/Kaggle T4 VRAM at `B=66` (FP32 vs `mixed_float16`) and execute the TF-gated graph tests.

### Phase 23R — Adapted RISE-UNet Baseline & Baseline Sensitivity Reference

- Train Adapted `RISE-UNet` (`modelRzsmReluExtraConv`) sequentially across `W1..W4` for seeds `[42, 123, 456]`, saving exact initial layer weights per seed so all downstream configurations share identical initial backbone weights.
- Restore each stage's best `val_loss` (`0.2*L1 + 0.3*L2 + 0.5*L3`) checkpoint before freezing BN in inference mode (`training=False`) while keeping `Dropout`/`SpatialDropout2D` active (`training=True`) to generate cached `D=10` stochastic prefix trajectories (`110` draws/issue).
- Evaluate Baseline on Validation (`N=100`): `110`-draw deterministic mean `ACC/MAE/RMSE`, full `110`-draw empirical `CRPS`, `80%/95%` spread/PI calibration, dry-event Brier & reliability (`< tau`), and controlled `m in {0, 0.5, 1, 1.5, 2}` recursive sensitivity curves.

### Phase 24R — SCPW–MRGR Implementation & Matched Ablation Matrix

- **Compute & Cross-Fitting Governance (Sec 2.3.1)**:
  - Implement `(stage_lead, excluded_year_set)` subset caching (`W1=129, W2=45, W3=9` = `183` unique stage fits across `129` unique year-subsets, vs. `675` naive fits per seed per MRGR-bearing config). Exact fold retained counts in `manifests/splits_v3/split_summary_v3.json` span **`750–770` issues** (`81–101` excluded/fold including `4` boundary-crossing December issues in `2016..2023`). Log wall-clock/GPU compute cost per configuration.
  - Monitor Colab GPU quota upfront before launching `MRGR-only` & `Combined` nested banks; keep inner-fold hyperparameters cleanly parameterized (fixed `40`-epoch budget, no checkpointing on outer held-out year).
- **Module Compliance Checks (Sec 2.4–2.5)**:
  - Verify `W1` predictions from `Baseline` and `MRGR-only` agree bitwise under matched seeds & dropout streams; verify `SCPW` `W1` and `Combined` `W1` agree bitwise.
  - Verify zero initial residual on `SCPW` (`MLP` output weights/bias = `0.0`) and `CorrectionNet` (`1x1` head weights/bias = `0.0`).
  - Verify `r*` is **never clipped to `[0,1]`**; log out-of-range frequency separately.

### Phase 25R & Phase 26R — Benchmarks, Sealed Final Evaluation & Thesis 2 Delivery

- Run `Persistence`, `U-Net`, and `UNet++` benchmarks under identical `851/100/108` purged splits and `126`-cell active mask.
- Freeze RiskNet evaluation decile bin edges and model selection rules on Validation (`N=100`) prior to single-pass evaluation on Sealed Final Test (`N=108`, `2024-11-25..2025-12-04`).
- Execute **28-calendar-day MBB (`10,000` replicates)** + **Holm-Bonferroni** on W3/W4 `Combined - Baseline` CRPS + **56-day sensitivity**, plus MRGR–SCPW interaction contrast, `DJF/MAM/JJA/SON` seasonal stratification, and full mechanism diagnostics (`Table 7`).

---

## 5. Synchronization & Document Governance Matrix

- **Tier 1 (Contracts)**: `contracts/A0/thesis_revised_baseline_and_data_contract.yaml`, `contracts/A1/mrgr_model_architecture_contract.yaml`, `contracts/A0/VERIFICATION_STATUS.yaml`, `contracts/spatial/spatial_grid_contract.yaml`.
- **Tier 2 (Master Plans)**: [`mindanao_adaptation_master_plan.md`](mindanao_adaptation_master_plan.md) and [`mindanao_remediation_and_optimal_model_masterplan.md`](mindanao_remediation_and_optimal_model_masterplan.md) (binding Before/During/After remediation & optimality playbook per [`AGENTS.md`](AGENTS.md)).
- **Tier 3 (Registry & Audits)**: [`reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md), [`reproduction_audit/26_updated_thesis_alignment_and_comprehensive_audit.md`](reproduction_audit/26_updated_thesis_alignment_and_comprehensive_audit.md), and [`reproduction_audit/27_end_to_end_verification_and_redo_verdict.md`](reproduction_audit/27_end_to_end_verification_and_redo_verdict.md) (synced cleanly to Kaggle `kirkishere/rise-unet-rzsm-docs`).
