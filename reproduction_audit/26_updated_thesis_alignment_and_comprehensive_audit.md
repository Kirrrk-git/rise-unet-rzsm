<!-- markdownlint-disable -->
# Comprehensive Alignment Audit: Updated Thesis Proposal (SCPW–MRGR), Baseline Recreation, Regional Adaptation & GCS-to-Kaggle Migration

```text
Document Authority : Operational Execution Matrix & Artifact Registry
Audit Date         : 2026-11-19
Reference Proposal : Improving DL-DM RISE-UNet: Seasonal Conditioning and Reliability-Gated
                     Recursive Refinement for Probabilistic Subseasonal Root-Zone
                     Soil-Moisture Forecasting (Thesis 1 — Outline / 12 Oct 2026 Defense Revision)
Parent Study       : Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3
Contract Suite     : contracts/A0/thesis_revised_baseline_and_data_contract.yaml (v3.0.0)
                     contracts/A1/mrgr_model_architecture_contract.yaml (v3.0.0)
Status             : [AUDIT COMPLETE — THESIS 1 OUTLINE / PRE-REMEDIATION]
```

---

## 1. Executive Summary

An exhaustive, code-and-data-verified audit was conducted across the repository, local data caches, Google Cloud Storage inventories, the migrated Kaggle migration index, and the revised 24-page Thesis 1 outline proposal (`RISE-UNet_Thesis_—_Original_and_Revised_Proposal.pdf`, including the 12 October 2026 outline-defense minutes in Appendix D).

### High-Level Verdicts

1. **Thesis Progress vs. Claims Alignment**:
   - The updated thesis explicitly classifies the work at **Thesis 1 (outline/proposal stage)** with zero empirical model accuracy results claimed. Existing Phase 21–23 baseline/diagnostic runs in the repository (`modelRzsmRelu.py`, 2015–2021 unpurged split, hardcoded non-anomaly predictor bounds, `D=1`, `B=33`) are **methodologically superseded** by the revised thesis protocol and **cannot** be reported directly as Thesis 2 baseline comparisons without remediation and retraining.
2. **Five Major Code & Data Bugs Uncovered in Legacy Baseline Runs**:
   - **Wrong Backbone Imported**: `src/models/a0_unet.py` imported `modelRzsmRelu.py` (`[32, 32, 64, 128]`, DepthwiseConv2D, `1.61M–1.63M` params) instead of **`modelRzsmReluExtraConv.py`** (`[32, 64, 128, 256]`, standard `Conv2D`, `10.22M–10.24M` params) used by parent `EX29` (`losses.py:469`, `02_run_model...ipynb:78`) and required for SCPW insertion.
   - **Channel 10 (`s2s_tcw`) 100% Saturated**: `scripts/08_derive_training_normalization.py` hardcoded `tcw` bounds to `[0.5, 15.0]` from a buggy pilot file, whereas real ECDS `tcw` spans `27.66–53.77 kg/m²`. Every single active grid cell in all production `.npz` tensors had `x_w1[..., 10] == 1.0000` (`std = 0.0`).
   - **Missing 7-Day Summaries, Seasonal Anomalies & Empirical Scaling on Atmos/S2S**: `case_builder.py` extracted un-anomalized single-day `t0` atmospheric fields (plus un-anomalized S2S fields) with hardcoded bounds, rather than parent `EX29`'s & Thesis Section 2.2.2–2.2.3's 7-day trailing summaries (`lag=-1d` for ERA5 obs), DJF/MAM/JJA/SON seasonal anomaly subtraction, and empirical training min-max normalization.
   - **Incomplete Atmospheric Derivation & Missing 2014 Warmup**: `gs://rise-unet-rzsm/processed/atmospheric/` contains only 106 NetCDFs (`2015_04..2023_12` + `2015_01` pilot), missing **26 monthly files** (`2015_02`, `2015_03`, `2024_01..2025_12`) that must be derived from raw ERA5 archives. Additionally, **zero 2014 ERA5 atmospheric files** exist in GCS/local (`2014-12-25..31` is required if computing 7-day trailing means at `2015-01-01` and `2015-01-04`).
   - **Metric & Split Drift**: Phase 21 `scripts/16` pooled space-time ACC (`r` over `194*11*126` elements) and member-wise MAE on `M=11` draws instead of per-case 110-draw predictive-mean metrics; legacy splits had `4` Train->Val and `4` Val->Test target-window overlaps; Phase 23 `notebooks/13` vs `scripts/16` diverged (`35–44%` unmasked vs `~20%` ocean-masked gap).
3. **GCS vs. Kaggle Migration Verification**:
   - **GCS Read/Write Blocked**: `gcloud billing projects describe mindanao-drought-aaron-jalapon` shows `billingEnabled: false`, and `gcloud storage objects describe` fails with **`HTTP 403`** (*billing account disabled*).
   - **Exact Byte-for-Byte Inventory Match**: Kaggle index `aaronjalapon02/mindanao-gcs-migration-index` (Version 2, updated `2026-11-02`, verified `2026-11-02T09:41:01Z`) accounts for all **`7,789` files / `45,788,363,491` bytes** (`45.79 GB`) across the 4 GCS buckets, including **`6,795` files / `31,092,944,230` bytes** from `gs://rise-unet-rzsm/` split cleanly across 3 private Kaggle datasets (`6,477` loose files + `299` in `remaining-a.zip.bin` + `19` in `remaining-b.zip.bin`). All checksums passed; only the 150 GB and 100 GB VM notebook disks were intentionally excluded by owner request.

---

## 2. Progress Inventory: Completed vs. Superseded vs. Missing

| Pillar | What Was Completed | Current Validity Against Updated Thesis PDF | Required Action |
| :--- | :--- | :--- | :--- |
| **Spatial Foundation (`21A`)** | Candidate A `32x48`, `0.25°`, `126` cells (`86.46%` area), WGS84 masks | **`[VALID]`** — Exact match to Thesis Sec 2.1.5 | Retain unchanged |
| **Raw Archives & RZSM Cube (`21B, 21D`)** | `265` raw ERA5-Land NCs, `132+132` raw ERA5 single/pressure NCs, `1,084` usable S2S cycles, 11-yr RZSM cube | **`[PARTIAL]`** — Raw files complete (except 2014-12 ERA5 atmos); RZSM climatology/norm fitted to `<=2021` | Acquire 7d Dec 2014 ERA5 atmos; derive `26` missing processed atmos months; refit climatology/norm to purged `80%` train |
| **Baseline Recreation & A0 (`Gate 1, 21K`)** | `modelRzsmRelu` (1.63M) trained 3 seeds on 677/194 split (`0.0341..0.0627` masked MAE) | **`[SUPERSEDED]`** — Wrong backbone (`modelRzsmRelu` vs `ExtraConv`), saturated `tcw`, un-anomalized predictors, old split | Refactor `a0_unet.py` to `modelRzsmReluExtraConv` (`10.24M`), fix data pipeline, re-profile T4 VRAM at `B=66`, retrain |
| **Recursive Diagnostic (`23`)** | Verified recursive gap exists (`~20%` masked / `35–44%` unmasked) + monsoon East-West dipole | **`[PRELIMINARY ONLY]`** — Proved conceptual motivation only; contains 104 vs 88 Amihan count bug in text & `L=4` MBB | Re-run under `ExtraConv`, `D=10` (`M=110`), `28d/56d` MBB, midpoint season on purged split |
| **Enhancement (`24+`)** | Drafted old MRGR-only v2.0.0 contract (frozen A0, DOY concat, clipped `r*`) | **`[SUPERSEDED]`** — Updated Thesis uses SCPW + MRGR, unfrozen consumer backbone, OOF RiskNets, unclipped `r*` | Implement from `contracts/A0/v3.0.0` and `contracts/A1/v3.0.0` |

---

## 3. Forensic Code & Documentation Defects Uncovered

### 3.1 Backbone Mismatch: `modelRzsmRelu.py` vs. `modelRzsmReluExtraConv.py`
- Parent `losses.py:466-469` and `02_run_model_EXPERIMENTS_ECMWF_and_GEFSv12_v5.ipynb:78,182` select **`modelRzsmReluExtraConv.py`** when `testing_scenario == 'regular'` (EX29). Updated Thesis Sec 2.4.1–2.4.2 explicitly specifies `modelRzsmReluExtraConv.py` (`[32, 64, 128, 256]`) and places **SCPW** between the initial 128-ch `inception_block` output (`u_h`) and its immediately following 64-ch `conv_batchnorm_relu_block_SE_and_residual_connections` composite.
- However, `freeze/parent_architecture_contract.yaml`, `01a_parent_baseline_execution_log.md`, and `notebooks/00_parent_freeze_and_inspection.ipynb` mixed `[32, 64, 128, 256]` metadata from Table S3 with `modelRzsmRelu.py` (`nb_filter=[32, 32, 64, 128]`, 18 `DepthwiseConv2D` layers, `1,630,307` params at `C=12`), and `src/models/a0_unet.py` imported `modelRzsmRelu`.
- **Verified Exact Parameter Counts (matched against KerasCV `SqueezeAndExcite2D(ratio=0.25, bias=True)`, validated to `0` error against `1,627,139 / 1,630,307 / 1,608,131 / 1,611,299`)**:
  - **Legacy `modelRzsmRelu.py`**: W1 `1,627,139` | W2 `1,630,307` | W3 `1,608,131` | W4 `1,611,299`
  - **Authentic `modelRzsmReluExtraConv.py` (`32x48`)**: W1 **`10,240,723`** | W2 **`10,243,539`** | W3 **`10,223,827`** | W4 **`10,226,643`** (~6.3x larger -> T4 GPU VRAM at `B=66` must be re-profiled, likely with `mixed_float16` matching parent notebook).
  - **SCPW Module Overhead**: W1 `15,024` | W2 `15,024` | W3 `5,808` | W4 `5,808` (`41,664` total across W1–W4).
  - **MRGR Module Overhead (RiskNet + CorrectionNet)**: `W1->W2` (`x_2=11ch`): `2,625 + 2,625 = 5,250`; each of the 5 handoffs entering W3/W4 (`x_h=3ch`): `2,049 + 2,049 = 4,098` (`25,740` total across 6 pairs). Total SCPW + MRGR combined overhead = **`67,404` params** (~0.16% of 4-stage `~40.93M` backbone).

### 3.2 `s2s_tcw` Saturation & Predictor Preprocessing Defects
- `scripts/08_derive_training_normalization.py:125-205` only computed RZSM bounds empirically, while hardcoding constants for all 5 ERA5 atmospheric variables and all 6 S2S channels (`s2s_tcw` hardcoded to `[0.5, 15.0]`). Real Mindanao ECDS `tcw` (`~27.66–53.77 kg/m²`) exceeded `15.0` everywhere, saturating `x_w1[..., 10]` and `x_w2_base[..., 10]` to constant `1.0000` (`std = 0.0`) across all 5 inspected production `.npz` archives (`CASE_20150101_0001.npz`..`CASE_20150115_0005.npz`). Unit test `test_08` missed this because it passed synthetic `tcw = 0.02`.
- `src/data/case_builder.py:164` extracted single-day `t0` raw atmospheric fields rather than trailing 7-day rolling seasonal anomalies at `lag = -1d` (`[t0-7 .. t0-1]`), introducing day-of-issue timing mismatch and omitting seasonal anomaly subtraction for both ERA5 and S2S predictors.

### 3.3 Other Verified Discrepancies Across Repo Artifacts
- **Season Count Bug in `THESIS_RESEARCH_GAP_EMPIRICAL_EVIDENCES.md`**: Section 8 reported `Amihan=104, Habagat=88, Transition=18` and `DJF=54, MAM=52, JJA=52, SON=52` (summing to `210` scheduled cycles including 16 Nov–Dec MARS gaps), whereas actual evaluated validation cases (`N=194`) are `88 Amihan + 88 Habagat + 18 Transition` and `44 DJF + 52 MAM + 52 JJA + 46 SON`.
- **Pressure Level Typo in `thesis_research_paper_package/README.md:71`**: Wrote `$Z_{500}$` instead of `$Z_{200}$` (200-hPa geopotential height `~12.4 km`).
- **Stale Manifest & Test Count Drift**: `manifests/production_case_calendar.csv` still reflects an older 527-ready snapshot (while `cases_availability_audit.csv` has all `1,073` usable issues); `raw/ecmwf_s2s/production/` in GCS contains `2015-01-16` and off-by-one `2015-11-15/19/29` alongside the `1,073` usable cycles + `7` Dec 2025 OOB cycles; `OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md` cited `114` unit tests (`108 passed, 6 skipped`), whereas actual execution on 2026-11-19 ran **`119` unit tests** (`113 passed, 6 TF-gated skipped`).

---

## 4. Independent Audit & Verification of the Updated Thesis PDF

### 4.1 Data Split Verification & Boundary Purging Math
Applying Thesis Section 2.3 (`floor(0.80 * 1073) = 858`, `floor(0.90 * 1073) - 858 = 107`, remaining `108`) to `cases_availability_audit.csv` (`1,073` usable issues from `2015-01-01` to `2025-12-04`) and purging boundary-crossing target windows (`t0 + 27d >= next_split.min_issue_date`) yields:
- **Training (Purged)**: **`851` issues** (`9,361` member-samples/lead), `2015-01-01` .. `2023-09-23` (`max W4 end = 2023-10-20`; `7` trailing issues `2023-09-26..2023-10-17` purged).
- **Validation (Purged)**: **`100` issues** (`1,100` member-samples/lead), `2023-10-21` .. `2024-10-28` (`max W4 end = 2024-11-24`; `7` trailing issues `2024-10-31..2024-11-21` purged). Covers one full 4-season annual cycle (`W1 mid: DJF=21, MAM=26, JJA=26, SON=27`), though concentrated in the 2023–2024 El Niño.
- **Sealed Final Test**: **`108` issues** (`1,188` member-samples/lead), `2024-11-25` .. `2025-12-04` (`max W4 end = 2025-12-31`; `W1 mid: DJF=28, MAM=26, JJA=27, SON=27`).
- **RZSM Out-of-Range Check**: Fitting seasonal climatology and active-cell bounds on `<= 2023-10-20` yields `[-0.185106, +0.157441]` and **`0.000%`** out-of-`[0, 1]` excursions across all `50,526` Val and `50,652` Test cell-days.

### 4.2 Critical Issues Flagged Inside the Updated Thesis Specification
1. **Triple-Nested OOF Cross-Fitting Compute Bottleneck (`Sec 2.3.1`)**: Leaving out 1 calendar year across 9 years (`2015..2023`; `81–101` issues excluded per outer fold, including `4` boundary-crossing December issues in `2016..2023`, leaving **`750–770` issues/fold** in `manifests/splits_v3/split_summary_v3.json`) for `h=4` with nested inner cross-fitting of `W3` and `W2` requires **`675` full 10.24M-param U-Net trainings per seed per MRGR config** (`4,050` across 3 seeds × 2 configs) if implemented naively. With `(stage_lead, excluded_year_set)` deduplication (`W1=129, W2=45, W3=9` across `129` year-subsets), this shrinks cleanly to **`183` fits per seed** (`1,098` total across 3 seeds × 2 configs)—still substantial GPU demand requiring explicit Colab Pro/Kaggle quota budgeting.
2. **Illustrative Case Date in Table 4 (`Sec 2.1.6 & 2.4.6`)**: Uses `2015-01-16`, which is the legacy pilot fixture date and **not** in the `1,073` operational CY48R1 issue list (closest actual cycles: `2015-01-15`, `2015-01-18`).
3. **`RiskNet` Context Blindness on `W3–W4` (`Sec 2.4.2 & 2.4.4`)**: At `h=3, 4`, `RiskNet_{j->h}(x_h, r_j)` receives only `3` antecedent RZSM channels (no meteorological predictors and no SCPW calendar signal), meaning it cannot directly condition source-error risk on Amihan vs. Habagat regimes. Worth noting explicitly during Thesis 2 diagnostic interpretation.
4. **Literature Fact-Check**: Key references `[1]–[25]` verified (note `[10]` Pagulayan et al. 2024 DRJ v15(1) title is `"The impacts and resilience of upland farming communities in Davao Oriental, Philippines through vulnerabilities and adaptation to climate change"` vs `"Vulnerability, adaptation and resilience..."` in Ref `[10]`).

---

## 5. GCS vs. Kaggle Migration Reconciliation Ledger

| GCS Source Bucket | Live `gsutil` Objects | Live `gsutil` Bytes | Kaggle Destination Dataset(s) | Migration Verified Objects / Bytes | Status |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `gs://rise-unet-rzsm/` | `6,795` | `31,092,944,230` | `mindanao-drought-data-gcs-migration/versions/2` (`6,477` loose: `16,977,826,455` B) + `rise-remaining-a-gcs-migration` (`299` zip: `4,096,398,106` B) + `rise-remaining-b-gcs-migration` (`19` zip: `10,018,719,669` B) | `6,795` / `31,092,944,230` B | `[VERIFIED EXACT]` |
| `gs://mindanao-drought-aaron-jalapon-drought-data/` | `508` | `4,339,058,006` | `mindanao-drought-main-gcs-migration` (`.zip.bin`) | `508` / `4,339,058,006` B | `[VERIFIED EXACT]` |
| `gs://rise-unet-rzsm-copy-20260909/` | `486` | `10,356,361,255` | `rise-backup-gcs-migration` (`.zip.bin`) | `486` / `10,356,361,255` B | `[VERIFIED EXACT]` |
| `gs://mindanao-drought-aaron-jalapon-drought-restricted/` | `0` | `0` | Recorded empty in index | `0` / `0` B | `[VERIFIED EXACT]` |
| **Grand Total** | **`7,789`** | **`45,788,363,491`** | Indexed in `aaronjalapon02/mindanao-gcs-migration-index` v2 (`4752b3fd...`) | **`7,789` / `45,788,363,491` B** | `[VERIFIED EXACT]` |

> [!NOTE]
> The Kaggle migration index page (`4752b3fdd87a48af418fe26afdd867ed31384eb87333125224d4b3c973f14d8b`, Version 2, 5.83 MB uncompressed / 665 kB download) loads via its private share link and contains `README.md` (`3.64 kB`), `kaggle_restore.py` (`3.19 kB`), `migration-summary.json` (`37.78 kB`), `object-locations.json` (`3.91 MB`), `source-final-recheck.json` (`1.07 kB`), and `source-inventories.json` (`1.87 MB`). Checksum receipts and live GCS metadata match byte-for-byte (`cloud_sources_modified: false`). Note that GCS VMs (`150 GB` and `100 GB` disks) were intentionally excluded by owner request, and read access to the child data datasets inside Kaggle kernels enabled the cloud verification & consolidation run documented in Section 7.

---

## 6. Offline Phase 21R Engineering Completed (2026-11-19)

Without exceeding data/quarantine boundaries (preserving all legacy v1 manifests/contracts untouched and never reading Validation/Sealed Test targets during parameter fitting), the following Thesis v3.0 foundations have been implemented and verified offline:

- **Kaggle Consolidated & Backup Store Adapter ([`src/data/cloud_lake.py`](../src/data/cloud_lake.py))**: Added `find_consolidated_kaggle_dir` + `restore_from_consolidated_kaggle` (resolving unpacked trees or the `7` topic `ZIP_STORED` bundles from `kirkishere/rise-unet-rzsm` / `rise-unet-rzsm-verify-and-consolidate`) with priority ahead of legacy `find_kaggle_index_dir`, `open_kaggle_backup_object`, and `restore_from_kaggle_backup` (loose + `.zip.bin` with MD5/CRC32C/size checks) and optional GCS fallback, plus [`scripts/21_sync_kaggle_docs_dataset.py`](../scripts/21_sync_kaggle_docs_dataset.py) to sync living code/docs to Private `kirkishere/rise-unet-rzsm-docs`.
- **Purged `80/10/10` Split & Train-Only RZSM Contract ([`src/data/splits_v3.py`](../src/data/splits_v3.py), [`scripts/20_generate_v3_purged_splits_and_rzsm_bounds.py`](../scripts/20_generate_v3_purged_splits_and_rzsm_bounds.py))**: Generated `manifests/splits_v3/` (`851` Train / `100` Val / `108` Sealed Test + `183`-fit OOF deduplication plan) and fitted `contracts/A0/rzsm_normalization_v3_purged.yaml` + `processed/rzsm/v3_purged_train_climatology_and_thresholds.nc` strictly on `<= 2023-10-20` RZSM (`min=-0.185106`, `max=+0.157441`, `std=0.039847`, plus per-cell × season 20th-percentile thresholds).
- **Predictor Pipeline Remediation ([`src/data/preprocessing_v3.py`](../src/data/preprocessing_v3.py))**: Replaced hardcoded `[0.5, 15.0]` `s2s_tcw` bounds with dynamic empirical training extrema (`eps_floor=1e-6`), added `lag = -1d` 7-day trailing ERA5 atmospheric rolling means, warmup checks, and DJF/MAM/JJA/SON seasonal anomaly subtraction.
- **SCPW–MRGR Model & Evaluation Modules ([`src/models/scpw_mrgr.py`](../src/models/scpw_mrgr.py), [`src/evaluation/v3_metrics_and_bootstrap.py`](../src/evaluation/v3_metrics_and_bootstrap.py))**: Implemented `build_extraconv_unet_v3` (hooking `SCPW` between `128`-ch inception and `64`-ch composite), `RiskNet` + `CorrectionNet` builders and NumPy gate math (unclipped `r*`), exact `110`-draw CRPS Eq. (15), strict `< tau` drought Brier/reliability, `28d/56d` calendar-block paired Moving-Block Bootstrap + Holm-Bonferroni, and RiskNet decile/perturbation slope diagnostics.
- **Automated Verification ([`tests/test_18_thesis_v3_modules.py`](../tests/test_18_thesis_v3_modules.py))**: Added unit tests verifying consolidated `rise-unet-rzsm` ZIP & unpacked restoration, Kaggle `.zip.bin` + loose restoration, purged splits (`851/100/108`) & `183`-fit OOF dedup, `ch10` non-saturation, SCPW–MRGR parameter counts (`67,404` overhead), exact CRPS Eq. (15), strict `< tau` Brier, and 28-day MBB. (Full GPU model instantiation awaits Colab/Kaggle GPU execution.)

---

## 7. Live 6,795-File Kaggle Verification & Consolidated Lake Execution (2026-11-20)

Authoritative cloud verification kernel `kirkishere/rise-unet-rzsm-verify-and-consolidate` (`KernelWorkerStatus.COMPLETE`, `2,443.03 s` elapsed, archived in [`logs/kaggle_migration_verification_report.json`](../logs/kaggle_migration_verification_report.json)) mounted all 4 Aaron Kaggle datasets inside Kaggle and performed single-pass cryptographic verification and selective consolidation:

- **100% Byte-for-Byte Cloud Verification (`status: COMPLETE_ALL_6795_VERIFIED`, `byte_for_byte_match: true`)**:
  - Verified **`6,795 / 6,795` objects (`31,092,944,230` bytes)** against `aaronjalapon02/mindanao-gcs-migration-index` (`object-locations.json`) via exact byte size, MD5, and CRC32C, plus full archive SHA-256 checks on `rise-remaining-a-gcs-migration.zip.bin` (`4,096,577,509` B, `299` objects, `sha256_verified: true`) and `rise-remaining-b-gcs-migration.zip.bin` (`10,018,733,738` B, `19` objects, `sha256_verified: true`).
- **Consolidated Active Lake (`3,826` files / `6,448,640,613` B in `7` `ZIP_STORED` bundles; `14,491,152,384` B working free)**:
  - `raw_era5_land.zip`: `267` files (`3,942,260,828` B)
  - `raw_era5.zip`: `265` files (`1,120,570,506` B)
  - `raw_ecmwf_s2s.zip`: `2,170` files (`372,006,659` B)
  - `raw_boundaries_and_other.zip`: `4` files (`522,389,890` B)
  - `processed_all.zip`: `1,003` files (`360,647,083` B)
  - `checkpoints_legacy_a0_best12.zip`: `24` files (`86,072,963` B — all 12 canonical `checkpoints/A0/seed_{42,123,456}/lead_{1..4}/best_model.weights.h5` + `training_history.json`)
  - `meta_docs_and_audits.zip`: `93` files (`44,692,684` B)
- **Intentional Bloat Parked in Aaron's Backup (Zero Active Loss)**:
  - `parked_reference_bytes`: `10,018,580,801` B (`reference/` — primarily the `9.95 GB` US CONUS `6y4kh-osfstorage-archive.zip`).
  - `parked_redundant_or_epoch_ckpt_bytes`: `14,625,722,816` B (`checkpoints/` 5x recursive upload nesting such as `seed_X/seed_X/seed_X` & `A0/A0/` plus `1,770` per-epoch `12 MB` `.npz` optimizer/intermediate checkpoints).
- **Cross-Check Summary**: `manifest_active_files.json` (`3,826` entries) reconciles against `object-locations.json` with `0` missing raw/processed files, `0` size mismatches, `0` CRC32C mismatches, `0` duplicates, and `0/12` missing canonical A0 weights.
- **Standalone Consolidated Dataset Verified Live (`kirkishere/rise-unet-rzsm`, status `ready`)**: Promoted cleanly from the verification kernel output into Private `kirkishere/rise-unet-rzsm` (`3,828` files / `6,449,251,345` B verified across all pages via Kaggle API: `checkpoints_legacy_a0_best12` [`24`], `meta_docs_and_audits` [`93`], `processed_all` [`1,003`], `raw_boundaries_and_other` [`4`], `raw_ecmwf_s2s` [`2,170`], `raw_era5` [`265`], `raw_era5_land` [`267`], `manifest_active_files.json` [`1`], and `verification_report.json` [`1`]), alongside Private `kirkishere/rise-unet-rzsm-docs` (`193` files, synced via [`scripts/21_sync_kaggle_docs_dataset.py`](../scripts/21_sync_kaggle_docs_dataset.py)). Note that Kaggle automatically unpacked the 7 topic ZIPs into `<bundle_stem>/<blob_name>`, matching Case 1 in [`src/data/cloud_lake.py`](../src/data/cloud_lake.py).
