<!-- markdownlint-disable -->
# Mindanao RISE-UNet Remediation & Optimal Model Master Plan (Before, During, and After Adaptation & Enhancement)

```text
Document Authority : Mandatory Execution & Optimality Plan (Enforced by AGENTS.md)
Created            : 2026-11-20
Grounded In Audit  : reproduction_audit/27_end_to_end_verification_and_redo_verdict.md
Paired Roadmap     : mindanao_adaptation_master_plan.md (v3.0.0-thesis-revised)
Governing Specs    : contracts/A0/thesis_revised_baseline_and_data_contract.yaml
                     contracts/A1/mrgr_model_architecture_contract.yaml
                     contracts/A0/rzsm_normalization_v3_purged.yaml
                     contracts/spatial/spatial_grid_contract.yaml
Primary Lake       : kirkishere/rise-unet-rzsm (3,828 files / 6.45 GB) + kirkishere/rise-unet-rzsm-docs
```

> [!IMPORTANT]
> **Purpose of This Master Plan**: Every step below converts the verified findings of Audit 27 into a concrete, ordered, fail-closed engineering and modeling playbook so that **both** your **Adapted Parent Baseline (`modelRzsmReluExtraConv`)** and your **Enhanced Model (`SCPW–MRGR`)** achieve maximum empirical optimality, reproducibility, and academic defensibility from data prep through final thesis evaluation.

---

## 0. Authoritative Thesis Timeline & Purged Split Map (Single Source of Truth)

Do **not** confuse the superseded Legacy Year Split (`2015–2021 / 2022–2023 / 2024–2025`, `677 / 194 / 202` unpurged) with the active **Oct 12, 2026 Revised Thesis v3.0 Split (`80/10/10` Purged)**:

```text
2014-12-12 .. 2014-12-31  │ Warmup Buffer ONLY (not trained or evaluated):
                          │   • 2014-12-12..31 ERA5-Land RZSM -> supports 7d mean at lag=-14d for 2015-01-01
                          │   • 2014-12-25..31 ERA5 Atmos      -> supports 7d mean at lag=-1d  for 2015-01-01
──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────
2015-01-01 .. 2023-09-23  │ TRAINING SET (80% purged = 851 issues / 9,361 member-samples per lead)
  (Max W4 target:         │   • All climatologies (DJF/MAM/JJA/SON), active-cell min-max bounds, and
   2023-10-20)            │     20th-percentile drought thresholds tau(s,k) fitted STRICTLY <= 2023-10-20
                          │   • Purged gap: 7 issues (2023-09-26..2023-10-17) dropped so W4 <= 2023-10-20
                          │   • Inside this span, SCPW-MRGR also runs 9-yr OOF cross-fitting (2015..2023)
──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────
2023-10-21 .. 2024-10-28  │ VALIDATION SET (10% purged = 100 issues / 1,100 member-samples per lead)
  (Max W4 target:         │   • Checkpoint selection, VRAM/hyperparameter checks, RiskNet decile calibration,
   2024-11-24)            │     and Phase 23R gap/monsoon diagnostics
                          │   • Purged gap: 7 issues (2024-10-31..2024-11-21) dropped so W4 <= 2024-11-24
──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────
2024-11-25 .. 2025-12-04  │ SEALED FINAL TEST (10% = 108 issues / 1,188 member-samples per lead)
  (Max W4 target:         │   • STRICTLY QUARANTINED: zero peeking, fitting, tuning, or plotting until single-pass
   2025-12-31)            │     Phase 26R final evaluation after all model choices are locked on Validation
```

---

## PHASE I — BEFORE TRAINING: Fix Verified Bugs, Seal Leakage & Build Optimal Data (`Phase 21R`)

Zero GPU baseline or enhancement training may start until **all four gates below** pass cleanly.

### Step 1.1 — Fix Critical Code Bugs Locally (`[BLOCKER CLEARANCE]`)
1. **Fix Keras 3 / TensorFlow 2.20 Crash in `src/models/scpw_mrgr.py`**:
   - Replace `tf.tanh(mlp_out)` (`L249`) and `tf.abs(e_x - e_r)` (`L326`) with Keras 3-native `keras.ops` (`keras.ops.tanh`, `keras.ops.abs`) or Keras layers so graph construction works identically under TF `2.20.0`.
   - Build a unified **differentiable TensorFlow joint `MRGR` graph** (`build_mrgr_joint_consumer_model`) linking `r_j -> frozen RiskNet -> G = exp(-q_hat) -> CorrectionNet -> r* = mask * (r_j + (1 - G)*c*tanh(d_theta)) -> concat(x_nonrec, r*) -> consumer ExtraConv F_h` so `CorrectionNet` optimizes end-to-end with `F_h`.
   - Enforce **`eval_mask` zero-filling (`~eval_mask = 0.0`) inside `apply_mrgr_gate_and_refinement`** before computing excursion counts (`below_zero_count`, `above_one_count`, `gate_mean`) and before feeding recursive channels downstream.
   - Implement exact **volumetric reconstruction helpers** (`scaled anomaly [0, 1] -> unscaled anomaly -> + climatology_rolling_7d(season, cell) -> m^3/m^3`) so predictions can be compared accurately against seasonal 20th-percentile drought thresholds `dry_threshold_p20` (`tau`).
2. **Fix Remote Data Resolution Bugs in `src/data/cloud_lake.py`**:
   - Fix `resolve_rzsm_target_cube` (`L565`) to request `"processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc"` (matching both repo and `kirkishere/rise-unet-rzsm`).
   - Fix `resolve_s2s_cycle` (`L515`) to resolve raw Kaggle GRIB pairs (`raw/ecmwf_s2s/production/<date>/s2s_{cf,pf}_<date>.grib`) and route through `harmonize_s2s_cycle(..., allow_step0_fallback=False)`.
3. **Clean Up `src/data/preprocessing_v3.py` Interface**:
   - Decouple default RZSM input from the legacy `2015–2021` `rzsm_0_100_seasonal_anomaly` inside `era5_land_rzsm_production_2015_2025.nc`; compute v3 RZSM anomalies dynamically from `rzsm_0_100_rolling_7d` using the `<= 2023-10-20` climatology in `v3_purged_train_climatology_and_thresholds.nc`.
   - Wire `atm_climatology` cleanly inside `assemble_v3_case_tensors` and require `min_check_date="2015-01-01"` when fitting ERA5 atmospheric climatologies/bounds so Dec 2014 warmup days never enter training statistics.
4. **Remediate Credibility Hazards Across Notebooks, Stubs & Audits**:
   - Move/quarantine the fabricated Gaussian Panel D in `figures/mindanao_s2s_pilot_case_and_recursive_inference.png` (`NB 07` Cell 19) and mark `NB 04` (`0/8` unexecuted), `NB 08` (`229K` MAE toy), and `documentation/09` (`FAIL`) clearly in the registry.
   - Remove nonexistent `"static soil clay/sand/bulk density"` references from legacy prose, sync script registry naming (`18_verify_training_numerics.py`), and mark `scripts/19 --mode execute` as legacy stub until replaced by the clean v3 Phase 23R script.

### Step 1.2 — Complete Missing Raw Warmup & 26 Derived Atmospheric Months (`Phase 21R.2`)
1. **Acquire Late-Dec 2014 ERA5 Atmospheric Warmup (`2014-12-25` to `2014-12-31`)**:
   - Run `scripts/02_download_era5_atmospheric.py --year 2014 --month 12` (single + pressure levels) and stage cleanly; verify no NaNs over Mindanao's `126` active cells. *(Fallback only if CDS API access is permanently unavailable: formally amend contract to document `min_periods=1` truncated trailing mean strictly on `2015-01-01..2015-01-06` and verify effect on Train-only W1).*
2. **Derive All `26` Missing Monthly Atmospheric NetCDFs (`scripts/03_derive_era5_atmospheric_daily.py`)**:
   - Derive `2015_01`, `2015_02`, `2015_03`, and `2024_01..2025_12` directly from `kirkishere/rise-unet-rzsm/raw_era5/` on Kaggle.
   - Upgrade coordinate assertions in `scripts/03` & `src/data/atmospheric.py` to check **both** `ds_single` and `ds_pressure` coordinates (`lat, lon`), squeeze any CDS `expver` dimension cleanly, and verify continuous `2014-12-25..2025-12-31` daily coverage (`0` gaps, `0` NaNs on `126` active cells).

### Step 1.3 — Assemble Unsaturated `1,059`-Case V3 Production Tensor Store (`Phase 21R.4`)
- Write production CLI `scripts/22_build_v3_cases_and_bounds.py` invoking `preprocessing_v3`:
  - Fit ALL scaling bounds & `DJF/MAM/JJA/SON` climatologies strictly on Train (`2015-01-01..2023-10-20`).
  - Verify empirically: `ch10 s2s_tcw` has full variance (std > 0, no clipping at `[0.5, 15.0]`), `atm` variables reflect 7-day trailing means ending at `lag = -1d`, all `1,410` ocean pixels are exact `0.0`.
  - Write `processed/cases_v3/{train,val,sealed_test}/CASE_*.npz` (`851 / 100 / 108`) and freeze `contracts/A0/predictor_normalization_v3_purged.yaml` with SHA-256 ledger. Keep `sealed_test/` isolated and untouched.

### Step 1.4 — Pre-GPU Verification & Pre-Training Design Optimization (`Phase 21R.5`)
Before committing compute to `183`-fit OOF and multi-seed training, resolve these **exact engineering & modeling trade-offs** cleanly via Validation (`N=100`, never Sealed Test):
1. **Live GPU Graph & `B=66` VRAM Profile (`10.24M ExtraConv`)**:
   - Create `scripts/23_profile_v3_extraconv_gpu.py` and run **all 127 unit tests including the 7 TensorFlow-gated tests** inside Kaggle/Colab GPU (`tensorflow==2.20.0`).
   - Profile `build_extraconv_unet_v3` (`10.24M`) & `SCPW–MRGR` at `B=66` (`6` cases × `11` members) in **FP32 vs `mixed_float16`** (mirroring Parent `EX29` NB 02 Cell 1, keeping output heads & `crps2d_tf` in `float32`). Lock the fastest non-OOM exact `B=66` configuration.
2. **Pre-Register `183`-Fit OOF Early-Stopping & LR Rule (Anti-Leakage)**:
   - Outer held-out year `y_out` **must never** be monitored by `EarlyStopping` or `ReduceLROnPlateau`. Lock an exact deterministic rule in code before running Phase 24R (recommended: hold out the most recent retained year of the `8` in-fold years as inner early-stop monitor, OR use a fixed epoch + cosine/step-decay schedule calibrated solely on full-Train `A0` convergence).
3. **Diagnostic Sensitivity Checks to Optimize Both Models Defensibly (Validation-Only)**:
   - **Check A — `91.80%` Ocean `0.0` Drought Collision & SE/BN Dilution**: Evaluate baseline `ExtraConv` `[11, 12, 5, 6]` faithfully first, and compare coastal vs inland cell errors on Validation (`N=100`). Only consider adding an explicit binary land-mask channel or masked normalization if documented as a controlled v3.1 ablation so Parent `EX29` parity remains pure for the baseline!
   - **Check B — `RiskNet` Seasonal Conditioning**: Test whether supplying SCPW calendar vector `s_h(t)` to `RiskNet` at `W3/W4` improves Huber error correlation (`Spearman rho`) across `Amihan`/`Habagat` on OOF/Validation.
   - **Check C — ReLU Output Floor (`0.0`) Monitoring**: Log % of active cells hitting exact `0.0` at extreme dry tails across Validation cases so CRPS & drought calibration (`< tau_20`) remain transparent.

---

## PHASE II — DURING TRAINING: Optimal Baseline Redo & Enhancement Execution (`Phase 23R–24R`)

### Step 2.1 — Authentic Adapted Baseline Redo (`Phase 23R`, `scripts/24_train_v3_baseline_and_diagnostic.py`)
1. **Matched Initialization Vault**: For each seed in `[42, 123, 456]`, save exact initial weights `w0(seed)` of `build_extraconv_unet_v3(lead=1..4)` before gradient step 1 so `Baseline`, `SCPW`, `MRGR`, `SCPW–MRGR`, and ablations share identical starting weights.
2. **Sequential Training (`W1 -> W2 -> W3 -> W4`)**:
   - `Adam(lr=1e-4)`, `B=66`, `max_epochs=40`, `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`, `EarlyStopping(patience=8, restore_best_weights=True)` monitoring masked active-cell `val_loss` (`[0.2, 0.3, 0.5]` deep supervision on differentiable `crps2d_tf`).
   - Before every downstream step (`W2, W3, W4`), enforce `eval_mask` (`~eval_mask = 0.0`), run **`D=10` MC Dropout inference passes** (BN frozen `training=False`, `Dropout`/`SpatialDropout2D` active `training=True`), cache the `M=110` draw trajectories, and never use GCS `cp -r` path nesting bugs.
   - Plot training curves for **all 3 seeds (`42, 123, 456`)** cleanly and log both member-wise/mean per-case metrics (`ACC` computed strictly per case over 110-draw predictive mean, not pooled across all dates × cells!).
3. **Clean Phase 23R Research Gap Re-Verification (`N=100` Val)**:
   - Re-run Protocol 1 (properly ocean-masked!) vs Protocol 2 (Oracle), Protocol 4 (`m`-perturbation sensitivity), and Gap 2 (monsoon `Amihan/Habagat`, `DJF/MAM/JJA/SON`, East-West spatial maps) directly from saved `N=100` predictions into `logs/v3_phase23r_diagnostic.json` (**no stubs, no unexecuted notebook cells, no Gaussian plots**).

### Step 2.2 — `183`-Fit OOF Bank & `SCPW–MRGR` Training (`Phase 24R`, `scripts/25_train_v3_scpw_mrgr.py`)
1. **Execute `183`-Fit Deduplicated OOF Cache**:
   - Using `plan_nested_yearly_oof_folds` (`W1=129, W2=45, W3=9` unique stage fits across `129` year-subsets per seed), generate clean `D=10` OOF predictions for both `Baseline` (`MRGR-only` bank) and `SCPW` (`Combined` bank) with strict `eval_mask` zeroing and the pre-registered OOF inner stopping rule.
   - Verify **bitwise identity**: `Baseline W1 == MRGR W1` and `SCPW W1 == Combined W1` for each seed & MC stream.
2. **Train & Gate-Verify `RiskNet` (`6` Handoffs)**:
   - Verify `RiskNet` converges smoothly under active-cell Huber loss (`delta=1.0`), verify zero BN/Dropout variance, compute `c_{j->h} = max(P95(|r_{j,OOF}-y_j|), 1e-6)`, and verify positive **Spearman correlation (`rho`)** & rising Validation MAE across predicted risk deciles before locking `RiskNet` weights.
3. **Joint `CorrectionNet` + Consumer Training & Matched Ablations**:
   - Verify exact zero residual at initialization (`scpw_mlp_out == 0.0`, `CorrectionNet 1×1 == 0.0`).
   - Freeze `RiskNet` & upstream stages; train consumer `F_h` + active `SCPW` + incoming `CorrectionNet`s jointly (`B=66`, `max 40 ep`), preserving **unclipped `r*`** (`~eval_mask = 0.0`) and logging active-cell out-of-`[0,1]` excursions.
   - Execute full matched ablation lineup:
     1. `Adapted RISE-UNet Baseline`
     2. `SCPW-only`
     3. `Constant-Calendar Control ([0, 0])`
     4. `MRGR-only`
     5. `Combined SCPW–MRGR`
     6. `Ungated Residual Ablation (r* = r + delta)`

---

## PHASE III — AFTER TRAINING: External Benchmarks, Statistical Certification & Delivery (`Phase 25R–26R`)

### Step 3.1 — External Benchmarks & Validation Lock (`Phase 25R`, `scripts/26_run_v3_benchmarks_and_selection.py`)
1. Implement and evaluate **Persistence** (carry latest `t-1` observed weekly RZSM anomaly), **Standard U-Net Regression**, and **UNet++ Regression** under the exact same `851/100` splits, active `126`-cell mask, and `B=66` protocol.
2. Freeze all model checkpoints, RiskNet evaluation decile bin edges, calibration summaries, and interpretation hypotheses on **Validation (`N=100`)** before ever opening `sealed_test_v3`. Document any El Niño (`2023–2024`) vs 9-year OOF generalization behavior transparently.

### Step 3.2 — Single-Pass Sealed Final Test & Publication Statistics (`Phase 26R`, `scripts/27_evaluate_sealed_test_v3.py`)
1. Unseal `manifests/splits_v3/test_cases_sealed_v3.csv` (`N = 108`, `2024-11-25..2025-12-04`) **once**.
2. Compute and persist complete active-cell metrics across all 8 configurations:
   - Predictive mean (`110` draws) `ACC`, `MAE`, `RMSE`
   - Full `110`-draw empirical `CRPS` (Eq. 15) + population spread + `80% / 95%` PI coverage & width
   - Physical `m³/m³` reconstructed dry-event probability (`p_hat = N_dry / 110` using strict `< tau(s, k)`), binary **Brier Score**, and **10-bin reliability diagrams**
   - **28-calendar-day circular Moving-Block Bootstrap** (`10,000` replicates preserving exact calendar gaps) + **`56-day` sensitivity check** + **Holm–Bonferroni** over W3 & W4 `Combined - Baseline` CRPS + MRGR–SCPW interaction contrast + `DJF/MAM/JJA/SON` & `Amihan/Habagat` stratifications.

### Step 3.3 — Clean Synchronization, Publication Package & Optional UI Prototype
1. Generate all manuscript figures and tables strictly from saved JSON telemetry and real model outputs (**zero synthetic curves**), run full unit tests on GPU & CPU (`127/127` passing on TF environment), verify all `reproduction_audit/` entries match logs SHA-256 for SHA-256, and push updated code/docs cleanly to `kirkishere/rise-unet-rzsm-docs` and checkpoints/results cleanly to Kaggle.
2. *(Optional Showcase)* Export final locked weights + seasonal `tau(s, k)` thresholds to a standalone **Streamlit prototype UI** that visualizes antecedent RZSM, 7d ERA5 atmosphere, S2S ensemble inputs, RiskNet gates, W1–W4 predicted soil moisture, and drought probability maps cleanly for thesis defense.
