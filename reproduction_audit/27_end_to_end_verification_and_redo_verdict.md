<!-- markdownlint-disable -->
# Independent End-to-End Verification Audit & Redo Verdict (Parent `EX29` -> Mindanao Adaptation -> Thesis v3.0 `SCPW–MRGR`)

**Document Identifier**: `reproduction_audit/27_end_to_end_verification_and_redo_verdict.md`  
**Audit Date**: 2026-11-20  
**Scope**: Complete empirical check of parent `EX29` (`parent_study_ex29/`), spatial grid & boundary masks, processed NetCDF cubes, all `14` Mindanao notebooks (`notebooks/00..13`) + `documentation/` notebooks, all `23` scripts (`scripts/01..21`), all `8` logs (`logs/*.json`), all `26` prior audits & contracts, and Thesis v3.0 modules (`Phase 21R`, `SCPW–MRGR`).  
**Sealed Test Policy**: Strictly zero validation/test target leakage; all Sealed Final Test (`2024-11-25` to `2025-12-04`, `N = 108`) target values remained untouched.

---

## 1. Executive Verdict (Plain-Language Bottom Line)

| Question | Factual Verdict | Plain-Language Summary |
| :--- | :---: | :--- |
| **Did we pick the best parent model to adapt?** | **YES** | Out of **16** model variants and **29** experiment configurations in `parent_study_ex29/`, the published *Nature Communications* authors explicitly selected **`EX29` + `modelRzsmReluExtraConv.py`** (`10.24M` params, standard `Conv2D`) as their sole final baseline (`losses.py:L466-467`). |
| **Was Legacy Phase 21–23 (`checkpoints/A0`) adapted and trained properly?** | **NO — REDO REQUIRED** | **7 verified defects** invalidate the 12 saved legacy models as an authentic thesis baseline: wrong backbone (`1.63M` Depthwise vs `10.24M` ExtraConv), `100%` constant-clipped moisture channel (`ch10 s2s_tcw = 1.0`), raw `t0` single-day ERA5 & un-anomalized predictors, unpurged splits (`677/194/202`), wrong batch/draw settings (`B=33, D=1`), pooled ACC metric, and an **unmasked-ocean leak** in `notebooks/13` that artificially inflated W4 recursive error from `0.0627` to `0.0898 ± 0.0288`. |
| **Should we redo adaptation and training?** | **PARTIAL REDO (Keep Data Foundation; Redo Pipeline & Models)** | **Keep** the spatial grid/mask (`32×48`, `126` active cells), 11-year RZSM target cube (`4,018` days), cleaned Kaggle data lake (`kirkishere/rise-unet-rzsm`), and purged `80/10/10` splits (`851/100/108`). **Redo** the predictor tensors, all baseline training (`Phase 23R`), all diagnostics, and enhancement/benchmark runs (`Phase 24R–26R`). |
| **Are we heading on the right path with Thesis v3.0 (`SCPW–MRGR`)?** | **CONCEPT YES; CODE & DATA INCOMPLETE** | The Oct 12 **Thesis v3.0** design fixes the major legacy flaws scientifically, **but zero GPU training should start until critical runtime bugs are fixed**: `src/models/scpw_mrgr.py` will crash on Keras 3 / TF 2.20 (`tf.tanh` / `tf.abs` on symbolic tensors), joint `CorrectionNet + Consumer` TF graph & volumetric drought inverse-scaling are unbuilt, `cloud_lake.py` has 2 remote path mismatches, `26` atmospheric months + `2014-12` warmup remain un-derived, and `scripts/22+` execution scripts do not yet exist. |

---

## 2. End-to-End Scorecard by Pipeline Component

| Pipeline Layer | Current Status | Verdict | Key Empirical Evidence |
| :--- | :---: | :---: | :--- |
| **1. Spatial Grid, Boundary & Mask** (`processed/grid/`, `scripts/01`) | `32×48` (`0.25°`), `126` land cells (`f >= 0.50`), `1,410` ocean | **`[SOLID WITH CAVEATS]`** | Exact N->S/W->E coordinate alignment across grid, mask, RZSM, pilot ERA5, & S2S; `0.0` NaNs. *Defense caveats*: `91.80%` ocean `0.0` padding; small Sulu/Tawi-Tawi islands excluded by `>=0.50` threshold (only 1 cell at Jolo `0.5969` west of `121°E`). |
| **2. 11-Yr ERA5-Land RZSM Cube** (`era5_land_rzsm_production_2015_2025.nc`) | `4,018` daily steps (`2015-01-01..2025-12-31`), `0` gaps | **`[DATA SOUND; V3 REFIT NEEDED]`** | Exact daily continuity (`min/max step = 1 day`), `0` active NaNs, `0` ocean nonzeros on inspected `<=2023-10-20` slice (`[0.1013, 0.5114] m³/m³`). However, embedded normalized variable uses legacy `<=2021` bounds; v3 (`<=2023-10-20`) bounds exist in `v3_purged_train_climatology_and_thresholds.nc`. |
| **3. Raw Archive & Kaggle Data Lake** (`kirkishere/rise-unet-rzsm`) | `3,828` files (`6.45 GB`), `6,795/6,795` GCS parity verified | **`[COMPLETE RAW; 26 DERIVED MOS & 2014-12 MISSING]`** | `265` raw ERA5 (`2015-01..2025-12`), `267` raw ERA5-Land, `2,170` S2S GRIB files intact. **Zero** `2014-12` ERA5 atmospheric files exist; only `106` derived monthly atmospheric files exist (`2015_02`, `2015_03`, `2024_01..2025_12` + prod `2015_01` missing). |
| **4. Legacy Baseline A0 (`checkpoints/A0`, `NB 12`, `scripts/16`)** | 12 weights (`42, 123, 456 × W1..W4`), `B=33`, `40` ep | **`[FAILED PARITY — SUPERSEDED]`** | Used `1.63M` `modelRzsmRelu` instead of `10.24M` `ExtraConv`, trained on saturated `ch10=1.0`, un-anomalized `t0` inputs, unpurged `677/194`, `B=33`, `training=False` (`D=1`). Keep **only** for temporary UI prototyping, never final thesis metrics. |
| **5. Legacy Diagnostic (`NB 13`, `scripts/19`, Gap 1 & 2)** | Executed inline (`N=194`), `scripts/19` stub | **`[INVALID NUMBERS — CONFOUNDED]`** | Forgot to zero `1,410` ocean cells between recursive steps `W1->W2->W3->W4` in `NB 13 Cell 11`, inflating W4 error (`0.0627 -> 0.0898`). `scripts/19 --mode execute` returns `0` without running anything; `NB 13` Cells `21, 23, 24` (Gap 2 monsoon/dipole) are unexecuted (`None`). |
| **6. Notebooks `00..11` & Verification Logs (`logs/*.json`)** | Mixed provenance & mocked/synthetic artifacts | **`[NEEDS CLEANUP]`** | `NB 04` `0/8` unexecuted; `NB 07 Panel D` plots a **fake Gaussian blob** instead of real model sensitivity; `NB 08` trains a `229K` toy U-Net on MAE labeled CRPS; `documentation/09` reports `FAIL`; local `logs/validation_atmospheric_execution.json` (`mode: "mock"`) & `a0_production_smoke_test.json` conflict with NBs `10` & `11`. |
| **7. Thesis v3.0 Modules (`splits_v3`, `preprocessing_v3`, `scpw_mrgr`, `v3_metrics`)** | Offline CPU modules + `test_18` (`7` CPU pass, `1` TF skipped) | **`[DESIGN SOUND; BUGS & NO EXECUTION SCRIPTS]`** | `splits_v3.py` (`851/100/108` + `183`-fit OOF dedup) & `v3_metrics_and_bootstrap.py` pass math checks. **Blocking bugs**: `scpw_mrgr.py` Keras 3 `tf.tanh`/`tf.abs` crash, missing TF MRGR graph, missing `eval_mask` in MRGR refinement, `cloud_lake.py` wrong `rzsm`/`s2s` paths, zero `scripts/22+` runners. |

---

## 3. Parent Baseline Review: Was `EX29` (`modelRzsmReluExtraConv`) the Best Model, & Was It Adapted Properly?

### 3.1 Did We Choose the Best Model from the Parent Paper? **YES.**
Direct inspection of `parent_study_ex29/function/` and `parent_study_ex29/notebooks/02_run_model_EXPERIMENTS_ECMWF_and_GEFSv12_v5.ipynb` confirms there are **16 architecture files** (`modelRzsmRelu`, `ExtraConv`, `ExtraConvDepthwise`, `ExtraConvResidual`, `2Ddropout`, `2DdropoutRelu`, `modelRzsm`, `modelRzsmTmax`, `DenseRelu`, `Attention`, `PixelSuper`, `Transformer`, `basicUNET`, `Backbone4SubPixel`, `Backbone4SubPixelNoPlusPlus`, `Backbone4TESTING`) and **29 experiment definitions (`EX0..EX29`)**.

- **Author Selection**: In `02_run_model...` Cell 6 & Cell 25, the authors explicitly enforce `testing_scenario = 'regular'` (**`modelRzsmReluExtraConv.py` + `crps2d_tf`**) and state in `losses.py:L466-467`:
  > *"This is the final one that I am sticking with. None of the other tests were significantly better"*
- **Why `EX29` Won (`loadDataAllWeeks.py:L681-846`)**: Earlier experiments (`EX9..EX28`) included noisy raw NWP soil moisture reforecasts (`soilw_bgrnd`) and all `3` lags (`-1, -7, -14`) of `5` ERA5 atmospheric fields (`15` noisy channels). **`EX29`** pruned both noise sources: keeping `3` RZSM memory lags (`-1, -7, -14`), **only `lag=-1` ERA5 atmospheric obs at W1–W2**, dynamic S2S forecasts (`t2m, d2m, tcw` for ECMWF) at W1–W2, and pure soil memory + recursive predictions at W3–W4.
- **Conclusion**: Choosing **`EX29` + `modelRzsmReluExtraConv`** is **100% scientifically right and defendable**. The problem was **not** picking `EX29`; the problem was that Legacy Phase 21 accidentally cloned the wrong parent file (`modelRzsmRelu.py`) due to errors captured in `freeze/parent_architecture_contract.yaml` and `freeze/parent_input_contract.json`!

### 3.2 Three-Way Comparison: Parent `EX29` vs. Legacy Phase 21 vs. Thesis v3.0

| Technical Choice | Parent `EX29` Actual Code | Legacy Phase 21 (DO NOT CITE) | Thesis v3.0 Target (`Phase 21R–26R`) |
| :--- | :--- | :--- | :--- |
| **Backbone Network** | `modelRzsmReluExtraConv` (`10.24M`, `[32,64,128,256]`, standard `Conv2D`) | `modelRzsmRelu` (`1.63M`, `[32,32,64,128]`, `DepthwiseConv2D`) | `modelRzsmReluExtraConv` (`10.22M–10.24M`) + Stage 1 `SCPW` (`+41,664`) + Stage 2 `MRGR` (`+25,740`) |
| **Domain / Mask Ratio** | CONUS `48×96`, `83.85%` active land | Mindanao `32×48`, `8.20%` (`126/1,536`) active land | Same `32×48`, `126` active cells |
| **Atmospheric Obs (`pwat, spfh, tmax, diff_temp, hgt_pres`)** | **7-day trailing mean at `lag = -1d`**, DJF/MAM/JJA/SON anomalies (`W1–W2` only) | **BUG**: Single-day raw values at **`t0`**, un-anomalized, **hardcoded bounds** | Restores **7-day trailing mean at `lag = -1d`**, train-only DJF/MAM/JJA/SON anomalies |
| **ECMWF S2S (`t2m, d2m, tcw`)** | `7d` lead mean, seasonal anomalies, domain min-max | **BUG**: Raw values (no anomalies); hardcoded `tcw=[0.5, 15.0]` **clipped 100% of Mindanao `tcw` (`27.7–53.8 kg/m²`) to `1.0`** | Dynamic train-only min-max on seasonal anomalies (`eps_floor=1e-6`), zero `tcw` saturation |
| **Loss & Gradients** | `crps2d_tf`: `MAE - 0.08*std` (**std detached via `.numpy()`!**), unmasked full grid, weights `[1,1,1]` | `crps2d_tf`: **differentiable** `sqrt(var+1e-7)`, **masked to `126` land cells**, weights `[0.2, 0.3, 0.5]` | Same masked differentiable `CRPS_exp` surrogate `[0.2, 0.3, 0.5]` + Huber (`RiskNet`) |
| **Batch, Epochs, EarlyStop** | `B=66`, `100` ep, `RLROP(0.1, 5)`, `ES(10)`, `restore_best=False` bug | `B=33`, `40` ep, `RLROP(0.5, 3)`, `ES(8)`, restores min val proxy | `B=66`, `max 40` ep, `RLROP(0.5, 3)`, `ES(8)` (need OOF rule) |
| **MC Dropout at Inference** | `SpatialDropout2D`/`Dropout` `training=True` (`D=1` default / `D=25` extra test) | `model(x, training=False)` / `predict()` -> `D=1` (though `training=True` hardcoded inside layers) | `D=10` stochastic passes (`11 × 10 = 110` draws) with BN frozen |
| **Chronological Splits** | `2000–15 / 16–19 / 20–22` (unpurged) | `2015–21 (677) / 22–23 (194) / 24–25 (202)` (**unpurged**: 8 W4 overlaps) | **Purged `80/10/10`**: `851 Train (<=2023-10-20) / 100 Val / 108 Test` + `183`-fit OOF |

### 3.3 Parent `EX29` Quirks You Must Disclose Honestly During Defense
Three genuine quirks exist inside the **published authors' own code** (`parent_study_ex29/`):
1. `parent_study_ex29/function/losses.py:L28-30` called `y_pred.numpy()` inside `_crps_tf`, severing gradients through the `-0.08 * std` spread term so Parent `EX29` trained only on whole-grid MAE!
2. `parent_study_ex29/notebooks/02_run_model...` Cell 19/23 saved the post-patience final epoch (`restore_best_weights=False`).
3. `loadDataAllWeeks.py:L815` accidentally masked `test_RZSM` with `RZSM_validation_obs==0`.

Mindanao's codebase **fixed all three** (differentiable `sqrt(var + 1e-7)`, masking the loss strictly to the `126` active Mindanao land cells instead of `1,410` ocean zeros, restoring best validation checkpoints, and using Mindanao's exact land mask). In your defense, present those three deltas explicitly as **documented fidelity fixes to numerical bugs in the reference script**, not accidental drift.

---

## 4. What Went Wrong in Legacy Phase 21–23, Notebooks, Plots, and Audits

1. **`src/data/s2s.py` & `processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc` Pilot Artifact Mismatch**: Inspecting `s2s_pilot_reforecast_w1_w2.nc` directly shows `issue_date: '2016-01-01'` and `tcw` ranging only `[2.238, 6.079] kg/m²` with identical Lead 1 and Lead 2 means (`0.361988` across full grid), confirming it came from an early fallback/partial GRIB fixture, unlike the verified production ECDS files (`27.66–53.77 kg/m²`). Production runs must strictly enforce `allow_step0_fallback=False`.
2. **Legacy Phase 21 (`scripts/16`, `NB 12`) Training & Metric Deficiencies**:
   - All 3 seeds (`42, 123, 456`) × 4 leads (`12` models) trained cleanly on Colab T4 (`epochs` converged cleanly; `seed_456` did complete despite missing a curve PNG in `figures/`), **but** trained the wrong `1.63M` `modelRzsmRelu` on broken `ch10=1.0`, un-anomalized `t0` ERA5 inputs, `B=33`, and unpurged `677/194`.
   - `scripts/16:L231-236` calculated `val_acc` (`0.854 / 0.682 / 0.563 / 0.495`) by pooling all `194 × 11 × 126` cells together (mixing spatial climatology gradients across Mindanao with forecast timing skill), rather than computing per-case spatial ACC over ensemble means.
   - `NB 12` GCS upload commands (`cp -r checkpoints/A0/ gs://.../A0/`) created triple/double-nested copies (`A0/A0/`, `seed_42/seed_42/seed_42/`), which combined with intermediate per-epoch optimizer `.npz` saves accounted for the `14.61 GB` legacy checkpoint bloat.
3. **Phase 23 Diagnostic Confounder (`NB 13` vs. `scripts/16`)**:
   - `scripts/16:L292-293` explicitly zeroed out ocean pixels (`p3[:, ~eval_mask, :] = 0.0`) before feeding W1->W2->W3->W4, achieving **W3 MAE `0.0593 ± 0.0007` and W4 MAE `0.0627 ± 0.0002`**.
   - `NB 13` Cell 11 forgot that line! Feeding **unmasked non-zero marine predictions** (`1,410` cells) into recursive channels trained on `0.0` ocean inflated W3 MAE to `0.0719 ± 0.0168` and W4 MAE to **`0.0898 ± 0.0288`** — meaning roughly half of the headline **`44.12%` W4 recursive gap** (`logs/recursive_degradation_diagnostic_results.json`, `THESIS_GAP`, `POST_A0`) was caused by the unmasked-ocean distribution shift vs. the properly zero-ocean Oracle (`0.0502`)!
   - Meanwhile, **`scripts/19 --mode execute` is an empty stub returning `0`**, and `NB 13` Cells `21, 23, 24` (Gap 2 monsoon/quarter/dipole analysis) have `execution_count = None`, so the local JSON lacks Gap 2 entirely.
4. **Notebook & Documentation Credibility Hazards (Clean Up Before Defense/Publication)**:
   - **`NB 07` Cell 19 Panel D (`figures/mindanao_s2s_pilot_case_and_recursive_inference.png`)**: Hardcoded a synthetic Gaussian surface centered at `(125.0°E, 7.6°N)` (`gaussian_filter`) while labeling it `"Downstream Sensitivity Response: Δ_W2 from W1 Perturbation"` — and `16_ecmwf_s2s_pilot_audit.md` certified it. **Delete or replace this panel immediately with real perturbation output.**
   - **`NB 04`**: Saved completely unexecuted (`0/8` cells `None`).
   - **`NB 08` & `scripts/12, 13`**: Used a `229,347`-param 2-layer Conv2D surrogate trained on unmasked MAE, yet printed `"Mean CRPS Loss"` (`20_audit` & `22_audit`).
   - **`documentation/09_mindanao_a0_vram_profiling.ipynb`**: Reports `HARDWARE_BENCHMARK_FAILED` in Cell 14 stdout, diverging from `notebooks/09` (`PASS` on 10.2–10.6 GB, which still only tested `1.63M`, not `10.24M ExtraConv`).
   - **False/Stale Claims in Legacy Audits**: `THESIS_GAP:L65` and `21_vram_audit:L67` invent static "soil clay, sand, bulk density" predictors that **do not exist** in `EX29` or Mindanao; `18_audit:L127` claims `264` atmospheric months complete when `26` are missing; `logs/validation_atmospheric_execution.json` is an old `mode="mock"` log (`3ffa9d3`); `OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md:L454` lists a non-existent script name (`18_verify_production_training_artifacts.py` vs actual `18_verify_training_numerics.py`) and recent Nov 19 v3 headers in `MATRIX`/`POST_A0`/`THESIS_GAP` misstated legacy epochs (`120` instead of `40`) and test count (`194` instead of `202`).

---

## 5. Thesis v3.0 Audit (`Phase 21R`, `SCPW–MRGR`): What Must Be Fixed Before GPU Training

### 5.1 Blocking Code & Data Bugs Verified Right Now

1. **Keras 3 / TF 2.20 Symbolic Crash in [`src/models/scpw_mrgr.py`](../src/models/scpw_mrgr.py)**:
   - Line `249`: `w_h = 1.0 + 0.25 * tf.tanh(mlp_out)` calls raw `tf.tanh` on a `KerasTensor`.
   - Line `326`: `diff = tf.abs(e_x - e_r)` calls raw `tf.abs` on `KerasTensor` subtraction.
   - Under TF `2.20.0` (Keras 3), both raise `ValueError: A KerasTensor cannot be used as input to a TensorFlow function`. Must use `keras.ops` (`keras.ops.tanh`, `keras.ops.abs`) or Keras layers (`Activation("tanh")`, `Lambda`/`Subtract`).
2. **Missing End-to-End Differentiable MRGR Training Graph & Volumetric Drought Inverse Unscaler**:
   - `apply_mrgr_gate_and_refinement` (`scpw_mrgr.py:L161-194`) is NumPy-only; no TF layer/model connects `r_j -> frozen RiskNet -> G = exp(-q_hat) -> CorrectionNet -> r* -> T_{h<-j}(r*) -> consumer ExtraConv` for joint optimization of `CorrectionNet` + consumer `F_h`.
   - Neither `scpw_mrgr.py` nor `preprocessing_v3.py` implements `T_{h<-j}` nor converts `[0, 1]` model anomaly outputs back into volumetric RZSM (`m³/m³`) by unscaling `[rzsm_min, rzsm_max]` and adding `climatology_rolling_7d(season, cell)` — which is **mandatory** to evaluate `y_hat < dry_threshold_p20` (`tau_20` is in `m³/m³`).
3. **Unmasked Ocean Leakage in MRGR (`scpw_mrgr.py:L161-194`)**:
   - `apply_mrgr_gate_and_refinement` accepts no `eval_mask`. Because `CorrectionNet` has bias and `3×3` convs, `r*` becomes nonzero on `1,410` ocean cells, corrupting `below_zero_count`/`gate_mean` logs and downstream `prepare_case_lead_tensors`. Must apply `r*[:, ~eval_mask, :] = 0.0` and compute excursion stats strictly over the `126` land cells.
4. **Remote Data Loader Path Bugs in [`src/data/cloud_lake.py`](../src/data/cloud_lake.py)**:
   - `resolve_rzsm_target_cube` (`L565`) requests `"processed/rzsm/era5_land_rzsm_production_2015_2025.nc"`, missing the `production/` folder (`"processed/rzsm/production/..."`), so cloud fallback fails when local cache is absent.
   - `resolve_s2s_cycle` (`L515`) requests a non-existent `.nc` path (`s2s_mindanao_YYYYMMDD.nc`), whereas Kaggle stores raw `raw/ecmwf_s2s/production/<date>/s2s_{cf,pf}_<date>.grib` needing `harmonize_s2s_cycle`.
5. **Interface Gap in [`src/data/preprocessing_v3.py`](../src/data/preprocessing_v3.py)**:
   - `assemble_v3_case_tensors` (`L187`) defaults to `"rzsm_0_100_seasonal_anomaly"`, which in `era5_land_rzsm_production_2015_2025.nc` was fitted on Legacy `2015–2021` (not `<=2023-10-20`!), and `V3PredictorNormalizationParams.atm_climatology` is never consumed during inference assembly.
6. **Missing Data & Zero `21R.2–26R` Execution CLI Scripts**:
   - Missing **`2014-12-25..31` hourly ERA5 raw atmospheric files** (`0` files in `kirkishere/rise-unet-rzsm`) + **`26` missing derived monthly atmospheric NetCDFs** (`2015_02`, `2015_03`, `2024_01..2025_12`, plus `2015_01` prod path).
   - **Zero CLI runners (`scripts/22+`) exist** for atmospheric completion, v3 case assembly (`1,059` cases), `10.24M ExtraConv` VRAM profiling (`B=66`), Phase 23R baseline training, Phase 24R `183`-fit OOF + `SCPW–MRGR`, Phase 25R (`U-Net`, `UNet++`, `Persistence` — which do not even exist in `src/` yet!), or Phase 26R Sealed Test.

### 5.2 Architectural & Statistical Risks to Tighten Before GPU Runs (High-Defensibility Checklist)

1. **`91.80%` Ocean `0.0` = Extreme Drought Ambiguity (`1,410 / 1,536` Zero Cells)**:
   - Scaling `[anom_min, anom_max] -> [0, 1]` maps the **worst drought in history** (`-0.1851`) to **`0.0`**, the exact same number assigned to **1,410 ocean pixels** surrounding Mindanao's narrow islands!
   - Furthermore, `modelRzsmReluExtraConv` contains **`86` `BatchNormalization` layers** and **`17` `SqueezeAndExcite2D` GlobalAveragePooling layers** computed across all `1,536` grid points — so **91.8% of every GAP/BN average is constant ocean zero bleed!**
   - *Actionable safeguard*: Either pass an explicit static binary land-mask input channel (or test center-padding / masked SE-GAP vs vanilla `ExtraConv` on Validation `N=100` in Phase 23R) so coastal cells aren't pulled toward false dry states!
2. **T4 `15.36 GB` VRAM Ceiling for `10.24M ExtraConv` at `B=66`**:
   - Legacy `1.63M` `modelRzsmRelu` already peaked at **`10,565 MB`** (`~10.3 GiB`) at `B=66` FP32 on Colab T4 (`logs/A0_gpu_benchmark.json`). `modelRzsmReluExtraConv` has **`6.3×` more parameters and wider `[32, 64, 128, 256]` standard `Conv2D` feature maps**. It will almost certainly OOM on T4 in FP32 at `B=66` unless **`mixed_float16` (just like Parent NB 02 Cell 1!)** or gradient accumulation (`2 × B=33` or `3 × B=22` — note BN differs for smaller micro-batches!) or a P100/L4/A100 GPU is used. Profile immediately in Phase 21R!
3. **Undefined Early-Stopping Monitor for `183` OOF Fits**:
   - Holding out `y_out` leaves `750–770` training cases/fold, with no designated validation split inside `splits_v3.py`. You must **pre-register** either a fixed epoch schedule derived from full Baseline convergence OR an inner held-out validation year inside each `8`-year fold so `y_out` is never leaked via early stopping / `ReduceLROnPlateau`.
4. **ReLU Head Clamping (`0.0` Floor) & `RiskNet` Calendar Blindness**:
   - `activation="relu"` on output heads clips extreme-dry predictions to exact `0.0` (collapsing uncertainty `< tau_20`). Monitor floor-clamping rate alongside `r* < 0` excursions.
   - `RiskNet` at `W3/W4` currently receives `(x_lags, r_j)` **without** SCPW's `[sin, cos]` seasonal vector `s_h(t)`, making it blind to `Amihan`/`Habagat` monsoon reliability differences. Adding `s_h(t)` (or testing it in Val ablation) is a small, clean win.
5. **Single-Year El Niño Validation (`2023-10-21..2024-10-28`, `N=100`) vs La Niña Test (`2024-11..2025-12`) + `74` Missing Nov–Dec MARS Slots**:
   - Your `N=100` Validation year spans the **2023–2024 El Niño**, while `N=108` Sealed Test shifts toward wetter 2025 conditions (`La Niña`), and `74` missing ECMWF reforecast dates fall systematically in **Nov–Dec (`Amihan` onset)** (`2015–2023`). Disclose this explicitly in Thesis Ch. 3 Limitations & rely on the 9-year OOF bank (`2015–2023`) for robust internal hyperparameter calibration rather than tuning heavily on single-year El Niño Validation!

---

## 6. Clear Action Plan: What to Keep, Fix Now, Re-Do Next, and Do Last

### Step A — Keep As-Is (Zero Redo Needed)
- Spatial boundary, `32×48` Candidate A grid, and `126`-cell geodesic evaluation mask ([`processed/grid/`](../processed/grid/)).
- 11-year daily ERA5-Land RZSM cube (`era5_land_rzsm_production_2015_2025.nc`, using the **raw/7d rolling layers** to compute clean v3 anomalies under `contracts/A0/rzsm_normalization_v3_purged.yaml`).
- Consolidated Kaggle datasets [`kirkishere/rise-unet-rzsm`](https://www.kaggle.com/datasets/kirkishere/rise-unet-rzsm) (`3,828` files / `6.45 GB`) & [`kirkishere/rise-unet-rzsm-docs`](https://www.kaggle.com/datasets/kirkishere/rise-unet-rzsm-docs).
- Purged `80/10/10` split manifests ([`manifests/splits_v3/`](../manifests/splits_v3/): `851 / 100 / 108`).
- The 12 legacy weights in `checkpoints_legacy_a0_best12.zip` **strictly for an optional Streamlit UI prototype demo** (clearly tagged as Legacy Phase 21 v1 prototype, never for final Thesis 2 tables).

### Step B — Immediate Engineering Fixes (Do Now Locally Before Any GPU Work)
1. Fix `src/models/scpw_mrgr.py` Keras 3 `keras.ops` compatibility (`tanh`, `abs`), implement the differentiable TF joint `MRGR` graph, add `eval_mask` zeroing inside MRGR refinement, and add `[0,1]` <-> volumetric `m³/m³` helper for `< tau_20`.
2. Fix `src/data/cloud_lake.py` paths in `resolve_rzsm_target_cube` (`processed/rzsm/production/...`) and `resolve_s2s_cycle` (raw GRIB + `harmonize_s2s_cycle`), plus `preprocessing_v3.py` default variable alignment.
3. Quarantine/remediate fake/contradictory artifacts: replace or quarantine `NB 07 Panel D` (`mindanao_s2s_pilot_case_and_recursive_inference.png`), stop `scripts/19` stub ambiguity, mark `NB 04` / `08` / `13 Gap 2` / `documentation/09` caveats clearly, and purge stale static soil-clay text from legacy docs.

### Step C — Data Completion & Hardware Check (Phase `21R.2–21R.5` on Kaggle/Colab)
1. Download `2014-12-25..31` hourly ERA5 (`single` + `pressure`) via `scripts/02_download_era5_atmospheric.py --year 2014 --month 12` (or explicitly lock clipped warmup rule if CDS unavailable), derive all `26` missing atmospheric monthly NetCDFs (`2015_02`, `2015_03`, `2024_01..2025_12`, plus `2015_01`) via `scripts/03_derive_era5_atmospheric_daily.py`, and verify `2015–2025` daily continuity.
2. Build v3.0 unsaturated `.npz` case tensors (`1,059` total: `851` Train fitted `<=2023-10-20`, `100` Val, `108` Sealed Test quarantined).
3. Run a GPU preflight script executing all TF-gated unit tests (`test_12, 13, 15, 18`), verifying `scpw_mrgr.py` parameter counts live on TensorFlow `2.20.0`, and profiling `10.24M ExtraConv` + `SCPW` VRAM at `B=66` (`mixed_float16` vs FP32).

### Step D — Authentic Baseline Redo (`Phase 23R`) & Enhancement Execution (`Phase 24R–26R`)
1. **Redo Baseline (`Phase 23R`)**: Train 3-seed (`42, 123, 456`) `10.24M ExtraConv` baseline (`B=66`, `max 40 epochs`, shared initial weights saved, `D=10` MC dropout -> `110` draws/case), and re-run ocean-masked recursive-vs-oracle + perturbation + monsoon dipole diagnostics on `N=100` Validation.
2. **Pre-register OOF Inner Policy & Execute `SCPW–MRGR` Matrix (`Phase 24R`)**: Run `183`-fit deduplicated nested OOF bank and train matched ablations (`Baseline`, `SCPW`, `Constant-Calendar`, `MRGR`, `SCPW–MRGR`, `Ungated`).
3. **External Benchmarks & Single-Pass Sealed Test (`Phase 25R–26R`)**: Implement and train `U-Net`, `UNet++`, and `Persistence` on identical `851/100` splits, freeze all model selection & RiskNet bin edges on Validation (`N=100`), then unseal Test (`N=108`) **once** for `28d`/`56d` circular calendar MBB + Holm-Bonferroni + drought Brier/reliability (`< tau_20`).
