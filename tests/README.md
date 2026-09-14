<!-- markdownlint-disable -->
# Automated Unit Test Suite & Verification Hierarchy Guide

This directory contains the automated test suite for the **Mindanao Tropical RISE-UNet Adaptation (Track B)**.

All unit tests are self-contained, execute without requiring external network access or cloud credentials, and validate the physical, mathematical, and algorithmic integrity of the pipeline across 81 tests.

To provide clear operational order and verification hierarchy, the test suite is structured into a **4-Tier Verification Ladder**, advancing from low-level coordinate invariants to full deep learning architectures.

---

## 1. Quick Start: Running Tests

To run the complete test suite (81 tests across 12 test modules):

```bash
# From repository root (dl_dm_rzsm_subseasonal_forecast/)
python -m unittest discover -s tests -p "test_*.py"
```

To execute tests by specific verification tier:

```bash
# Tier 1: Spatial & Infrastructure Foundation
python -m unittest tests.test_01_cloud_lake tests.test_02_rzsm tests.test_03_temporal

# Tier 2: Observation Decoders & Target Cube Synthesis
python -m unittest tests.test_04_s2s tests.test_05_compile_cube tests.test_06_target_reconciliation

# Tier 3: Operational Calendar, Partitions & Case Assembly
python -m unittest tests.test_07_case_calendar tests.test_08_normalization_and_splits tests.test_09_pilot_ladder tests.test_10_case_builder

# Tier 4: TensorFlow Pipeline & Neural Architecture
python -m unittest tests.test_11_tf_dataset tests.test_12_a0_unet
```

---

## 2. 4-Tier Verification Hierarchy

```text
  Tier 4: Deep Learning Pipeline & Neural Architecture
  ├── test_11_tf_dataset.py                     <── Ensemble grouping (B mod 11 == 0), target broadcasting, checkpoint parity
  └── test_12_a0_unet.py                        <── Model A0 parameter count (1,630,307), deep supervision heads, forward pass
        ▲
  Tier 3: Calendar, Partitions & Case Assembly
  ├── test_07_case_calendar.py                  <── 1,154 CY48R1 operational cycle schedule, leap days, date math
  ├── test_08_normalization_and_splits.py       <── Train (735), Val (210), Sealed Test (209) partition counts & normalization
  ├── test_09_pilot_ladder.py                   <── Bit-for-bit array integrity & provenance of 8 pilot case npz tensors
  └── test_10_case_builder.py                   <── Input channel schedules (W1: 11, W2: 12, W3: 5, W4: 6) & ocean masking
        ▲
  Tier 2: Ingestion & Target Synthesis
  ├── test_04_s2s.py                            <── Pure-Python GRIB2 Section 7 decoding, ensemble M=11, 1.5° -> 0.25° remap
  ├── test_05_compile_cube.py                   <── 11-Year continuous daily NetCDF target cube compilation (4,018 nominal days)
  └── test_06_target_reconciliation.py          <── Exact mathematical parity of 7-day rolling targets against parent EX29
        ▲
  Tier 1: Spatial & Infrastructure Foundation
  ├── test_01_cloud_lake.py                     <── Local-first resolution engine, URI construction, offline execution
  ├── test_02_rzsm.py                           <── Candidate A grid (32x48), 126 active cells, depth-weighted RZSM (0-100 cm)
  └── test_03_temporal.py                       <── Rolling 7-day window formulas & antecedent lags (t0-1d, 7d, 14d)
```

---

## 3. Test Suite Catalog & Verification Scope

| Tier | Module | Scope / Component Tested | Key Invariants & Assertions Verified |
| :---: | :--- | :--- | :--- |
| **Tier 1** | **[`test_01_cloud_lake.py`](test_01_cloud_lake.py)** | Data Lake Access & Resolution | • Local-first priority for committed codebase artifacts.<br>• Zero-network execution for offline development.<br>• Correct GCS URI construction and actionable error messages when offline. |
| **Tier 1** | **[`test_02_rzsm.py`](test_02_rzsm.py)** | Spatial Grid & Soil Moisture Layering | • Bounding box coordinates ($4.00^\circ\text{N} - 11.75^\circ\text{N}, 116.00^\circ\text{E} - 127.75^\circ\text{E}$).<br>• Grid dimension $(32 \times 48)$ with exactly 126 active land cells.<br>• Root-zone depth-weighted integration of Layers 1, 2, 3 ($0\text{--}100\,\text{cm}$). |
| **Tier 1** | **[`test_03_temporal.py`](test_03_temporal.py)** | Temporal Aggregation & Lags | • Rolling 7-day target aggregation for Leads 1, 2, 3, 4.<br>• Antecedent lag retrieval ($t_0-1\text{d}, t_0-7\text{d}, t_0-14\text{d}$). |
| **Tier 2** | **[`test_04_s2s.py`](test_04_s2s.py)** | ECMWF S2S Reforecast Decoder | • GRIB2 Section 7 decoding without external binary dependencies.<br>• Ensemble dimension $(M=11)$, bilinear spatial remapping ($1.5^\circ \to 0.25^\circ$).<br>• Strict hard-fail default when forecast lead steps are missing. |
| **Tier 2** | **[`test_05_compile_cube.py`](test_05_compile_cube.py)** | NetCDF RZSM Target Datacube | • 11-year daily continuous time dimension ($4,018$ nominal days).<br>• Monotonic coordinate validation ($0.25^\circ$ spacing).<br>• Finite value checks and unit consistency ($m^3/m^3$). |
| **Tier 2** | **[`test_06_target_reconciliation.py`](test_06_target_reconciliation.py)** | Target Calculation & Offset Parity | • Verifies 7-day rolling average target definitions against parent EX29.<br>• Validates explicit day offset formulas for $W_1$ through $W_4$. |
| **Tier 3** | **[`test_07_case_calendar.py`](test_07_case_calendar.py)** | Operational Forecast Calendar | • Reconciles 1,154 operational cycles against ECMWF CY48R1 reference calendar.<br>• Validates date parsing, leap year handling, and Mon/Thu scheduling. |
| **Tier 3** | **[`test_08_normalization_and_splits.py`](test_08_normalization_and_splits.py)** | Normalization Contract & Splits | • Train (735), Val (210), Sealed Test (209) partition counts.<br>• Verifies target-horizon boundary extension semantics.<br>• Asserts active consumption of `normalization_parameters.yaml`. |
| **Tier 3** | **[`test_09_pilot_ladder.py`](test_09_pilot_ladder.py)** | Pilot Case Provenance | • Provenance trace of 8 pilot case tensors (`CASE_20150115_W01.npz` to `CASE_20150304_W08.npz`).<br>• Bit-for-bit array integrity and metadata consistency. |
| **Tier 3** | **[`test_10_case_builder.py`](test_10_case_builder.py)** | Full Forecast Case Construction | • Input channel schedules ($W_1: 11, W_2: 12, W_3: 5, W_4: 6$).<br>• Correct alignment of 1d, 7d, 14d antecedent soil moisture lags.<br>• Application of binary land evaluation mask (zero-fill ocean buffer). |
| **Tier 4** | **[`test_11_tf_dataset.py`](test_11_tf_dataset.py)** | TensorFlow Data Pipeline | • Ensemble grouping constraint ($B \pmod{11} == 0$).<br>• Target broadcasting: $Y_{Wk} (1, 32, 48, 1) \to (11, 32, 48, 1)$.<br>• Multi-head CRPS loss calculation and checkpoint save/restore parity. |
| **Tier 4** | **[`test_12_a0_unet.py`](test_12_a0_unet.py)** | Model A0 UNET_RZSM Architecture | • Instantiation across Leads 1, 2, 3, 4.<br>• Exact parameter count (1,630,307 for Lead 2).<br>• Output shapes `(B, 32, 48, 1)` across 3 deep supervision heads.<br>• Finite forward pass with zero NaN/Inf activations. |

---

## 4. Why Tests Have Both Numbers and Functional Names

In Python testing best practices (pytest / unittest / PEP 8):
1. **1-to-1 Module Parity**: Standard naming directly links the test module to the source code module it validates (e.g., `tests/test_02_rzsm.py` tests `src/data/rzsm.py`, `tests/test_10_case_builder.py` tests `src/data/case_builder.py`).
2. **Test Independence & Hermeticity**: Unit tests are designed to be order-independent so they can be run in any sequence or executed in parallel across CPU cores using `pytest-xdist`.
3. **Traceability**: Audit dossiers, peer review logs, and the scientific master plan reference tests by their canonical functional names.
